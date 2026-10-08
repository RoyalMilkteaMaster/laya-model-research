import json
import re
from datetime import date, datetime

import pytest

from multihop_benchmark.benchmark_logger import get_logger

REQUIRED_FIELDS = {"timestamp", "level", "event", "source", "message"}


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_each_line_has_required_fields_and_optional_context(tmp_path):
    logger = get_logger("setup", log_dir=tmp_path)

    logger.info("開始檢查", event="setup_started")
    logger.error(
        "Ollama 連線失敗",
        event="ollama_unreachable",
        run_id="run-1",
        combo="hotpotqa__qwen3.5:4b__rag_laya",
        question_id="q-7",
        errorCode="OLLAMA_CONNECT",
    )

    lines = read_lines(tmp_path / "benchmark.jsonl")
    assert len(lines) == 2
    for line in lines:
        assert REQUIRED_FIELDS <= line.keys()
        assert line["source"] == "setup"
        datetime.fromisoformat(line["timestamp"])
    assert lines[0]["level"] == "INFO"
    assert lines[0]["event"] == "setup_started"
    assert lines[0]["message"] == "開始檢查"
    assert lines[1]["level"] == "ERROR"
    assert lines[1] | {
        "run_id": "run-1",
        "combo": "hotpotqa__qwen3.5:4b__rag_laya",
        "question_id": "q-7",
        "errorCode": "OLLAMA_CONNECT",
    } == lines[1]


def test_context_cannot_overwrite_core_fields(tmp_path):
    logger = get_logger("run", log_dir=tmp_path)
    try:
        raise ValueError("boom")
    except ValueError:
        logger.exception(
            "actual message",
            event="actual_event",
            timestamp="not-a-timestamp",
            source="spoofed",
            message="spoofed message",
            exception="spoofed exception",
            run_id="run-1",
        )

    (line,) = read_lines(tmp_path / "benchmark.jsonl")
    datetime.fromisoformat(line["timestamp"])
    assert line["level"] == "ERROR"
    assert line["event"] == "actual_event"
    assert line["source"] == "run"
    assert line["message"] == "actual message"
    assert "ValueError: boom" in line["exception"]
    assert line["run_id"] == "run-1"
    # 撞名的 context 不遺失，改放在巢狀 context 欄位（level 由 logging.log(level, ...) 簽名擋下，無法傳入）
    assert line["context"] == {
        "timestamp": "not-a-timestamp",
        "source": "spoofed",
        "message": "spoofed message",
        "exception": "spoofed exception",
    }


def test_event_defaults_when_not_given(tmp_path):
    get_logger("setup", log_dir=tmp_path).warning("沒有 event")

    (line,) = read_lines(tmp_path / "benchmark.jsonl")
    assert REQUIRED_FIELDS <= line.keys()
    assert line["event"]


@pytest.mark.parametrize(
    ("source", "log_file"),
    [
        ("run", "benchmark.jsonl"),
        ("report", "benchmark.jsonl"),
        ("train-laya", "train_laya.jsonl"),
        ("prepare-data", "prepare_data.jsonl"),
        ("build-index", "prepare_data.jsonl"),
        ("make-laya-data", "prepare_data.jsonl"),
    ],
)
def test_source_selects_program_log_file(tmp_path, source, log_file):
    get_logger(source, log_dir=tmp_path).info("hi", event="probe")

    assert [p.name for p in tmp_path.iterdir()] == [log_file]


def test_explicit_log_name_overrides_source_mapping(tmp_path):
    get_logger("dense_index", log_name="prepare_data", log_dir=tmp_path).info("hi")

    assert (tmp_path / "prepare_data.jsonl").exists()


def test_exception_is_recorded(tmp_path):
    logger = get_logger("run", log_dir=tmp_path)
    try:
        raise ValueError("boom")
    except ValueError:
        logger.exception("失敗", event="question_failed")

    (line,) = read_lines(tmp_path / "benchmark.jsonl")
    assert "ValueError: boom" in line["exception"]


def test_rotated_file_is_named_with_date(tmp_path):
    logger = get_logger("run", log_dir=tmp_path)
    logger.info("第一天", event="combo_started")

    (handler,) = logger.logger.handlers
    handler.doRollover()
    logger.info("第二天", event="combo_started")

    rotated = sorted(p.name for p in tmp_path.iterdir() if p.name != "benchmark.jsonl")
    assert rotated == [f"benchmark-{date.today():%Y-%m-%d}.jsonl"]
    assert re.fullmatch(r"benchmark-\d{4}-\d{2}-\d{2}\.jsonl", rotated[0])
    assert read_lines(tmp_path / rotated[0])[0]["message"] == "第一天"
    assert read_lines(tmp_path / "benchmark.jsonl")[0]["message"] == "第二天"


def test_rollover_after_another_process_rotated_writes_to_current_file(tmp_path):
    # run 與 report 是兩個程序共用 benchmark.jsonl：另一程序先把檔案改名輪替後，
    # 本程序輪替時不得繼續寫進已改名的舊檔。
    logger = get_logger("report", log_dir=tmp_path)
    logger.info("前一天", event="report_written")
    rotated = tmp_path / f"benchmark-{date.today():%Y-%m-%d}.jsonl"
    (tmp_path / "benchmark.jsonl").rename(rotated)

    (handler,) = logger.logger.handlers
    handler.doRollover()
    logger.info("今天", event="report_written")

    assert [line["message"] for line in read_lines(rotated)] == ["前一天"]
    assert [line["message"] for line in read_lines(tmp_path / "benchmark.jsonl")] == ["今天"]


def test_default_log_dir_is_data_root_logs(tmp_path, monkeypatch):
    monkeypatch.setenv("PROJECT_DATA_ROOT", str(tmp_path / "laya_data"))
    monkeypatch.setenv("PROJECT_RUNTIME_ROOT", str(tmp_path / "laya_runtime"))

    get_logger("setup").info("hi", event="setup_started")

    assert (tmp_path / "laya_data" / "logs" / "benchmark.jsonl").exists()
