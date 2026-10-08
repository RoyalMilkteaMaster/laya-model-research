"""以 pynvml 在背景執行緒取樣整張 GPU 的已用記憶體，回傳 start()～stop() 區間峰值（MiB）。

量的是整張卡的 used（含 Ollama 等其他程序），因為 LLM 在 Ollama 程序內執行。
沒有 GPU、找不到 NVML 或初始化失敗時 stop() 回 None、不拋例外，原因放在 `error`，
由呼叫端決定是否寫 Log。
"""

from __future__ import annotations

import threading
import warnings
from typing import Any

_MIB = 1024 * 1024


def _import_pynvml() -> Any:
    with warnings.catch_warnings():
        # PyPI 的 pynvml 包裝套件匯入時提示改用 nvidia-ml-py（實際模組相同）。
        warnings.simplefilter("ignore", FutureWarning)
        import pynvml
    return pynvml


class GpuMemorySampler:
    def __init__(self, interval_s: float = 1.0, *, device_index: int = 0, nvml: Any = None) -> None:
        if interval_s <= 0:
            raise ValueError(f"interval_s 必須 > 0，目前為 {interval_s!r}")
        self.interval_s = interval_s
        self.device_index = device_index
        self.error: str | None = None
        self._nvml = nvml
        self._handle: Any = None
        self._peak_bytes: int | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """開始新區間：初始化 NVML、立即取樣一次並啟動背景執行緒。失敗時只記錄 error。"""
        if self._thread is not None:
            raise RuntimeError("GpuMemorySampler 已在取樣中，請先 stop()")
        self.error = None
        self._peak_bytes = None
        self._handle = None
        try:
            if self._nvml is None:
                self._nvml = _import_pynvml()
            self._nvml.nvmlInit()
        except Exception as exc:  # 無 GPU／無驅動／無套件：一律降級為 None
            self.error = f"NVML 初始化失敗：{exc!r}"
            return
        try:
            self._handle = self._nvml.nvmlDeviceGetHandleByIndex(self.device_index)
        except Exception as exc:
            self.error = f"取得 GPU {self.device_index} 失敗：{exc!r}"
            self._shutdown()
            return

        self._sample()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="gpu-memory-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> int | None:
        """結束區間並回傳峰值 MiB；未取得任何樣本時回 None。"""
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join()
            self._thread = None
            self._sample()
            self._shutdown()
        return None if self._peak_bytes is None else self._peak_bytes // _MIB

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_s):
            if not self._sample():
                return

    def _sample(self) -> bool:
        if self._handle is None:
            return False
        try:
            used = int(self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used)
        except Exception as exc:
            self.error = f"讀取 GPU 記憶體失敗：{exc!r}"
            self._handle = None
            return False
        if self._peak_bytes is None or used > self._peak_bytes:
            self._peak_bytes = used
        return True

    def _shutdown(self) -> None:
        try:
            self._nvml.nvmlShutdown()
        except Exception:
            pass
