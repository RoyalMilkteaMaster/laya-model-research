import sys
import threading
import time
from types import SimpleNamespace

import pytest

from multihop_benchmark.evaluation.gpu_memory_sampler import GpuMemorySampler

MIB = 1024 * 1024


class FakeNvml:
    """模擬 pynvml：依序回傳 used bytes，用完後維持最後一個值。"""

    def __init__(self, used_mib, fail_init=False, fail_after=None):
        self.used = [value * MIB for value in used_mib]
        self.fail_init = fail_init
        self.fail_after = fail_after
        self.calls = 0
        self.shutdowns = 0
        self.lock = threading.Lock()

    def nvmlInit(self):
        if self.fail_init:
            raise RuntimeError("NVML Shared Library Not Found")

    def nvmlShutdown(self):
        self.shutdowns += 1

    def nvmlDeviceGetHandleByIndex(self, index):
        return f"gpu{index}"

    def nvmlDeviceGetMemoryInfo(self, handle):
        with self.lock:
            self.calls += 1
            if self.fail_after is not None and self.calls > self.fail_after:
                raise RuntimeError("GPU is lost")
            used = self.used[min(self.calls - 1, len(self.used) - 1)]
        return SimpleNamespace(used=used, total=24564 * MIB, free=24564 * MIB - used)


def wait_for_calls(fake, count, timeout=5.0):
    deadline = time.monotonic() + timeout
    while fake.calls < count:
        assert time.monotonic() < deadline, f"只取樣 {fake.calls} 次"
        time.sleep(0.005)


def test_stop_returns_peak_mib_over_interval():
    fake = FakeNvml([1024, 3072, 2048, 512])
    sampler = GpuMemorySampler(interval_s=0.01, nvml=fake)

    sampler.start()
    wait_for_calls(fake, 4)
    peak = sampler.stop()

    assert peak == 3072
    assert fake.shutdowns == 1


def test_samples_in_background_at_interval():
    fake = FakeNvml([100])
    sampler = GpuMemorySampler(interval_s=0.02, nvml=fake)

    sampler.start()
    time.sleep(0.2)
    sampler.stop()

    assert 3 <= fake.calls <= 20  # 約 0.2 / 0.02 + 2 次；下限放寬以容忍負載


def test_short_interval_still_measured_at_start_and_stop():
    fake = FakeNvml([700, 900])
    sampler = GpuMemorySampler(interval_s=60.0, nvml=fake)

    sampler.start()
    started = time.monotonic()
    peak = sampler.stop()

    assert time.monotonic() - started < 1.0  # stop 不必等滿 interval
    assert peak == 900


def test_nvml_init_failure_returns_none_without_raising():
    sampler = GpuMemorySampler(nvml=FakeNvml([100], fail_init=True))

    sampler.start()
    assert sampler.stop() is None
    assert "NVML" in sampler.error


def test_missing_pynvml_returns_none_without_raising(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)  # import pynvml → ImportError
    sampler = GpuMemorySampler()

    sampler.start()
    assert sampler.stop() is None
    assert sampler.error


def test_sampling_error_keeps_peak_of_successful_samples():
    fake = FakeNvml([400, 800], fail_after=2)
    sampler = GpuMemorySampler(interval_s=0.01, nvml=fake)

    sampler.start()
    wait_for_calls(fake, 3)
    assert sampler.stop() == 800
    assert "GPU is lost" in sampler.error


def test_stop_without_start_returns_none():
    assert GpuMemorySampler(nvml=FakeNvml([100])).stop() is None


def test_sampler_can_be_restarted_for_next_interval():
    fake = FakeNvml([5000, 100])
    sampler = GpuMemorySampler(interval_s=60.0, nvml=fake)

    sampler.start()
    assert sampler.stop() == 5000
    sampler.start()
    assert sampler.stop() == 100


def test_interval_must_be_positive():
    with pytest.raises(ValueError):
        GpuMemorySampler(interval_s=0)


@pytest.mark.integration
def test_real_gpu_peak_is_positive():
    sampler = GpuMemorySampler(interval_s=0.1)

    sampler.start()
    time.sleep(0.5)
    peak = sampler.stop()

    assert sampler.error is None
    assert peak is not None and peak > 0
