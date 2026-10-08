"""run／probe-models 實機煙霧（需 Ollama＋GPU＋已 prepare-data／build-index／train-laya）：`pytest -m integration`。

暫存 Data Root 以符號連結共用正式 `datasets/` 與 `indexes/`，records 與 Log 寫在暫存目錄，不污染正式 runs/。
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from multihop_benchmark.settings import load_settings

pytestmark = pytest.mark.integration

SMALL_MODEL = "qwen3.5:0.8b"


@pytest.fixture
def data_root(tmp_path):
    real = load_settings().data_root
    root = tmp_path / "laya_data"
    root.mkdir()
    for name in ("datasets", "indexes"):
        (root / name).symlink_to(real / name)
    return root


def cli(data_root: Path, *args: str) -> list[str]:
    return [sys.executable, "-m", "multihop_benchmark.cli", *args]


def env_for(data_root: Path) -> dict[str, str]:
    return {**os.environ, "PROJECT_DATA_ROOT": str(data_root)}


def read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def events(data_root: Path) -> list[dict]:
    return read_lines(data_root / "logs" / "benchmark.jsonl")


def test_timeout_one_second_marks_timeout_and_run_continues(data_root):
    result = subprocess.run(
        cli(data_root, "run", "--run-id", "it-timeout", "--limit", "2", "--models", SMALL_MODEL,
            "--datasets", "hotpotqa", "--conditions", "recall_llm", "rag_laya", "--timeout-s", "1"),
        env=env_for(data_root), capture_output=True, text=True, timeout=1800,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    records_dir = data_root / "runs" / "it-timeout" / "records"
    rows = [row for path in sorted(records_dir.iterdir()) for row in read_lines(path)]
    assert len(rows) == 4
    assert any(row["status"] == "timeout" for row in rows)
    assert [e["event"] for e in events(data_root)].count("combo_finished") == 2


def test_kill_minus_9_then_resume_skips_done_questions(data_root):
    run_args = ("run", "--run-id", "it-resume", "--limit", "3", "--models", SMALL_MODEL,
                "--datasets", "hotpotqa", "--conditions", "recall_llm", "rag_llm")
    records_dir = data_root / "runs" / "it-resume" / "records"
    first = records_dir / "hotpotqa__qwen3.5-0.8b__recall_llm.jsonl"

    process = subprocess.Popen(cli(data_root, *run_args), env=env_for(data_root),
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline and not (first.is_file() and len(read_lines(first)) >= 2):
        assert process.poll() is None, "run 在 kill 前就結束了"
        time.sleep(0.2)
    os.kill(process.pid, signal.SIGKILL)
    process.wait()
    before = {path.name: read_lines(path) for path in records_dir.iterdir()}
    done_before = {(name, row["question_id"]) for name, rows in before.items() for row in rows
                   if row["status"] == "done"}
    assert done_before

    result = subprocess.run(cli(data_root, *run_args), env=env_for(data_root), capture_output=True, text=True,
                            timeout=1800)
    assert result.returncode == 0, result.stdout + result.stderr
    after = {path.name: read_lines(path) for path in records_dir.iterdir()}
    for name, rows in before.items():
        assert after[name][: len(rows)] == rows  # 既有行不變
        rerun = {row["question_id"] for row in after[name][len(rows):]}
        assert not rerun & {qid for (n, qid) in done_before if n == name}  # 已 done 的題不重跑
    assert len(after) == 2 and all(len({row["question_id"] for row in rows}) == 3 for rows in after.values())
    resumed = [e for e in events(data_root) if e["event"] == "resumed"]
    assert resumed and sum(e["skipped"] for e in resumed) == len(done_before)


def test_probe_models_single_model(data_root):
    result = subprocess.run(cli(data_root, "probe-models", "--run-id", "it-probe", "--models", SMALL_MODEL),
                            env=env_for(data_root), capture_output=True, text=True, timeout=1800)
    assert result.returncode == 0, result.stdout + result.stderr
    probe = json.loads((data_root / "runs" / "it-probe" / "probe_results.json").read_text(encoding="utf-8"))
    entry = probe["models"][SMALL_MODEL]
    assert entry["n"] == 20 and 0 <= entry["valid_tool_call_rate"] <= 1 and entry["mean_latency_ms"] > 0
