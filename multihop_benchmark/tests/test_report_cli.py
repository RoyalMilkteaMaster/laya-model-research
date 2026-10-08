import json
import time

import pytest

from multihop_benchmark.cli import main
from test_progress_report_writer import FIXTURE_ORDER, install_fixture, table_row


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "laya_data"
    monkeypatch.setenv("PROJECT_DATA_ROOT", str(root))
    monkeypatch.setenv("PROJECT_RUNTIME_ROOT", str(tmp_path / "laya_runtime"))
    return root


def test_report_writes_progress_and_dashboard_once(data_root, capsys):
    install_fixture(data_root)

    exit_code = main(["report", "--run-id", "fixture"])

    assert exit_code == 0
    assert (data_root / "reports" / "PROGRESS.md").is_file()
    assert (data_root / "reports" / "dashboard.html").is_file()
    out = capsys.readouterr().out
    assert "完成 9 / 12000 題（0.08%），已嘗試 12" in out
    assert "目前組別 musique__qwen3.5:0.8b__recall_llm" in out
    log_lines = (data_root / "logs" / "benchmark.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(log_lines[-1])["event"] == "report_written"


def test_report_without_records_prints_no_data(data_root, capsys):
    exit_code = main(["report", "--run-id", "missing-run"])

    assert exit_code == 0
    assert "尚無資料" in capsys.readouterr().out
    assert "尚無資料" in (data_root / "reports" / "PROGRESS.md").read_text(encoding="utf-8")
    assert "尚無資料" in (data_root / "reports" / "dashboard.html").read_text(encoding="utf-8")


def test_report_every_rewrites_on_schedule_and_picks_up_new_records(data_root, monkeypatch):
    records_dir = install_fixture(data_root)
    progress = data_root / "reports" / "PROGRESS.md"
    snapshots, sleeps = [], []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        snapshots.append(progress.read_text(encoding="utf-8"))
        with (records_dir / FIXTURE_ORDER[2]).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "run_id": "fixture", "dataset": "musique", "model": "qwen3.5:0.8b", "condition": "recall_llm",
                "question_id": "mq3", "gold": "Oslo", "prediction": "Bergen", "em": 0, "f1": 0.0,
                "status": "done", "hops": 3, "steps": [], "latency_ms": 2500,
                "finished_at": "2026-09-29T03:00:03+00:00",
            }) + "\n")

    monkeypatch.setattr(time, "sleep", fake_sleep)

    exit_code = main(["report", "--run-id", "fixture", "--every", "600", "--count", "2"])

    assert exit_code == 0
    assert len(sleeps) == 1 and 599 <= sleeps[0] <= 600
    before = table_row(snapshots[0], "musique", "qwen3.5:0.8b", "recall_llm")
    after = table_row(progress.read_text(encoding="utf-8"), "musique", "qwen3.5:0.8b", "recall_llm")
    assert before[3:6] == ["2", "2", "1.000"]
    assert after[3:6] == ["3", "3", "0.667"]  # 新增 mq3（EM 0）後：(1+1+0)/3


def test_report_rejects_non_positive_every(data_root, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["report", "--run-id", "fixture", "--every", "0"])
    assert excinfo.value.code != 0
    assert "--every" in capsys.readouterr().err


def test_report_logs_one_warning_per_file_with_corrupt_lines(data_root):
    records_dir = install_fixture(data_root)
    for name in FIXTURE_ORDER[:2]:
        with (records_dir / name).open("a", encoding="utf-8") as handle:
            handle.write("garbage one\ngarbage two\n")

    assert main(["report", "--run-id", "fixture"]) == 0

    logs = [json.loads(line) for line in (data_root / "logs" / "benchmark.jsonl").read_text(encoding="utf-8").splitlines()]
    warnings = [entry for entry in logs if entry["level"] == "WARNING"]
    assert [entry["event"] for entry in warnings] == ["record_line_invalid", "record_line_invalid"]
    # hotpotqa 檔原有 4 行、2wiki 檔 6 行，殘行接在其後
    assert {entry["file"]: entry["lines"] for entry in warnings} == {FIXTURE_ORDER[0]: [5, 6], FIXTURE_ORDER[1]: [7, 8]}
    assert all(entry["errorCode"] == "RECORD_LINE_INVALID" and entry["run_id"] == "fixture" for entry in warnings)
