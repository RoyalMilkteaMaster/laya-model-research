import copy
import json

import pytest

from multihop_benchmark.runs import record_store
from multihop_benchmark.runs.record_store import RecordValidationError, append, completed_ids, validate_record

RUN_ID = "run-test"
COMBO = "hotpotqa__qwen3.5:4b__rag_laya"


def make_record(question_id="q1", status="done", **overrides):
    record = {
        "run_id": RUN_ID,
        "dataset": "hotpotqa",
        "model": "qwen3.5:4b",
        "condition": "rag_laya",
        "question_id": question_id,
        "gold": "United States",
        "gold_aliases": ["USA"],
        "prediction": "USA",
        "em": 1,
        "f1": 1.0,
        "status": status,
        "hops": 1,
        "steps": [
            {
                "hop": 1,
                "sub_question": "Which country?",
                "evidence_added": ["p1", "p2"],
                "laya": {"sufficient_p": 0.3, "next_action": "B", "remaining": 1, "latency_ms": 12.5},
                "llm": {"prompt_tokens": 120, "output_tokens": 9, "latency_ms": 350.0},
            },
            {
                "hop": 2,
                "sub_question": "Final?",
                "evidence_added": [],
                "llm": {"prompt_tokens": 200, "output_tokens": 4, "latency_ms": 210.0},
            },
        ],
        "latency_ms": 900.0,
        "vram_peak_mb": 5120,
    }
    record.update(overrides)
    return record


def records_file(data_root, combo=COMBO):
    return data_root / "runs" / RUN_ID / "records" / f"{combo}.jsonl"


def read_jsonl(path):
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    return [json.loads(line) for line in text.splitlines()]


def test_append_writes_valid_jsonl_under_data_root(tmp_path):
    append(RUN_ID, COMBO, make_record("q1"), data_root=tmp_path)

    lines = read_jsonl(records_file(tmp_path))
    assert lines == [make_record("q1")]
    for line in lines:
        validate_record(line)


def test_consecutive_appends_do_not_overwrite(tmp_path):
    for qid in ("q1", "q2", "q3"):
        append(RUN_ID, COMBO, make_record(qid), data_root=tmp_path)
    append(RUN_ID, "musique__qwen3.5:4b__recall_llm", make_record("m1"), data_root=tmp_path)

    assert [line["question_id"] for line in read_jsonl(records_file(tmp_path))] == ["q1", "q2", "q3"]
    assert [line["question_id"] for line in read_jsonl(records_file(tmp_path, "musique__qwen3.5:4b__recall_llm"))] == ["m1"]


def test_data_root_defaults_to_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("PROJECT_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("PROJECT_RUNTIME_ROOT", str(tmp_path / "runtime"))

    append(RUN_ID, COMBO, make_record("q1"))

    assert records_file(tmp_path).is_file()
    assert completed_ids(RUN_ID, COMBO) == {"q1"}


def test_completed_ids_ignores_non_done_status(tmp_path):
    append(RUN_ID, COMBO, make_record("q1"), data_root=tmp_path)
    for qid, status in [("q2", "invalid_tool"), ("q3", "timeout"), ("q4", "oom"), ("q5", "error")]:
        append(RUN_ID, COMBO, make_record(qid, status=status, error="boom"), data_root=tmp_path)

    assert completed_ids(RUN_ID, COMBO, data_root=tmp_path) == {"q1"}


def test_completed_ids_for_missing_file_is_empty(tmp_path):
    assert completed_ids(RUN_ID, COMBO, data_root=tmp_path) == set()


def test_simulated_interruption_resumes_with_done_ids(tmp_path):
    for qid in ("q1", "q2", "q3"):
        append(RUN_ID, COMBO, make_record(qid), data_root=tmp_path)
    append(RUN_ID, COMBO, make_record("q4", status="timeout", error="timeout after 300s"), data_root=tmp_path)

    assert completed_ids(RUN_ID, COMBO, data_root=tmp_path) == {"q1", "q2", "q3"}


def test_truncated_last_line_is_skipped_and_next_append_starts_new_line(tmp_path):
    for qid in ("q1", "q2"):
        append(RUN_ID, COMBO, make_record(qid), data_root=tmp_path)
    path = records_file(tmp_path)
    with path.open("a", encoding="utf-8") as fh:  # 模擬寫到一半被強制終止
        fh.write(json.dumps(make_record("q3"))[:40])

    assert completed_ids(RUN_ID, COMBO, data_root=tmp_path) == {"q1", "q2"}

    append(RUN_ID, COMBO, make_record("q3"), data_root=tmp_path)

    assert completed_ids(RUN_ID, COMBO, data_root=tmp_path) == {"q1", "q2", "q3"}
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 4
    assert json.loads(raw_lines[-1])["question_id"] == "q3"
    warnings = [json.loads(line) for line in (tmp_path / "logs" / "benchmark.jsonl").read_text("utf-8").splitlines()]
    assert any(w["event"] == "record_line_invalid" and w["combo"] == COMBO for w in warnings)


def test_append_rejects_invalid_record_without_writing(tmp_path):
    with pytest.raises(RecordValidationError, match="status"):
        append(RUN_ID, COMBO, make_record(status="finished"), data_root=tmp_path)
    assert not records_file(tmp_path).exists()


def test_append_rejects_record_from_another_run(tmp_path):
    with pytest.raises(RecordValidationError, match="run_id"):
        append(RUN_ID, COMBO, make_record(run_id="other-run"), data_root=tmp_path)


@pytest.mark.parametrize("bad", ["", "a/b", "..", "a\\b"])
def test_append_rejects_unsafe_path_parts(tmp_path, bad):
    with pytest.raises(ValueError):
        append(RUN_ID, bad, make_record(), data_root=tmp_path)
    with pytest.raises(ValueError):
        completed_ids(bad, COMBO, data_root=tmp_path)


def test_validate_accepts_optional_fields():
    record = make_record(status="error", error="ollama crashed", vram_peak_mb=None, prediction="", em=0, f1=0.0)
    validate_record(record)
    record["steps"][0]["laya"] = None
    validate_record(record)
    validate_record(make_record(steps=[], hops=0))


def _set(record, path, value):
    target = record
    for key in path[:-1]:
        target = target[key]
    if value is _DELETE:
        del target[path[-1]]
    else:
        target[path[-1]] = value


_DELETE = object()


@pytest.mark.parametrize(
    ("path", "value", "field"),
    [
        (("question_id",), _DELETE, "question_id"),
        (("gold",), None, "gold"),
        (("gold_aliases",), "USA", "gold_aliases"),
        (("gold_aliases",), ["USA", 3], "gold_aliases"),
        (("prediction",), None, "prediction"),
        (("em",), 2, "em"),
        (("em",), True, "em"),
        (("f1",), 1.5, "f1"),
        (("f1",), "0.5", "f1"),
        (("status",), "finished", "status"),
        (("hops",), -1, "hops"),
        (("hops",), 1.5, "hops"),
        (("latency_ms",), -3, "latency_ms"),
        (("vram_peak_mb",), "big", "vram_peak_mb"),
        (("error",), 42, "error"),
        (("surprise",), 1, "surprise"),
        (("steps",), {}, "steps"),
        (("steps", 0), "hop", "steps[0]"),
        (("steps", 0, "hop"), _DELETE, "steps[0].hop"),
        (("steps", 0, "sub_question"), 5, "steps[0].sub_question"),
        (("steps", 0, "evidence_added"), "p1", "steps[0].evidence_added"),
        (("steps", 0, "laya", "sufficient_p"), 1.2, "steps[0].laya.sufficient_p"),
        (("steps", 0, "laya", "next_action"), _DELETE, "steps[0].laya.next_action"),
        (("steps", 0, "laya", "latency_ms"), _DELETE, "steps[0].laya.latency_ms"),
        (("steps", 0, "laya", "next_action"), "NOT_A_CHOICE", "steps[0].laya.next_action"),
        (("steps", 0, "laya", "next_action"), "answer", "steps[0].laya.next_action"),
        (("steps", 0, "laya", "next_action"), "a", "steps[0].laya.next_action"),
        (("steps", 0, "laya", "remaining"), 2.5, "steps[0].laya.remaining"),
        (("steps", 0, "laya", "remaining"), 3, "steps[0].laya.remaining"),
        (("steps", 0, "laya", "remaining"), -1, "steps[0].laya.remaining"),
        (("steps", 0, "laya", "remaining"), 1.0, "steps[0].laya.remaining"),
        (("steps", 0, "laya", "remaining"), True, "steps[0].laya.remaining"),
        (("steps", 0, "laya", "remaining"), _DELETE, "steps[0].laya.remaining"),
        (("steps", 1, "llm"), _DELETE, "steps[1].llm"),
        (("steps", 1, "llm", "prompt_tokens"), -1, "steps[1].llm.prompt_tokens"),
        (("steps", 1, "llm", "output_tokens"), "9", "steps[1].llm.output_tokens"),
        (("steps", 0, "extra"), 1, "steps[0].extra"),
    ],
)
def test_validate_reports_offending_field(path, value, field):
    record = copy.deepcopy(make_record())
    _set(record, path, value)

    with pytest.raises(RecordValidationError) as excinfo:
        validate_record(record)

    assert field in str(excinfo.value)


@pytest.mark.parametrize("action", ["A", "B", "C"])
@pytest.mark.parametrize("remaining", [0, 1, 2])
def test_validate_accepts_every_laya_choice_and_score(action, remaining):
    record = make_record()
    record["steps"][0]["laya"].update(next_action=action, remaining=remaining)
    validate_record(record)


def test_truncated_multibyte_utf8_line_is_skipped(tmp_path):
    append(RUN_ID, COMBO, make_record("q1"), data_root=tmp_path)
    path = records_file(tmp_path)
    partial = json.dumps(make_record("q2", prediction="臺灣"), ensure_ascii=False).encode("utf-8")
    cut = partial.index("臺".encode("utf-8")) + 1  # 中止在多位元組字元的第一個 byte 之後
    with path.open("ab") as fh:
        fh.write(partial[:cut])

    assert completed_ids(RUN_ID, COMBO, data_root=tmp_path) == {"q1"}

    append(RUN_ID, COMBO, make_record("q2", prediction="臺灣"), data_root=tmp_path)

    assert completed_ids(RUN_ID, COMBO, data_root=tmp_path) == {"q1", "q2"}
    warnings = [json.loads(line) for line in (tmp_path / "logs" / "benchmark.jsonl").read_text("utf-8").splitlines()]
    assert any(w["event"] == "record_line_invalid" and w["combo"] == COMBO for w in warnings)


def test_validate_rejects_non_mapping():
    with pytest.raises(RecordValidationError, match="record"):
        validate_record(["not", "a", "record"])


def test_module_exposes_valid_statuses():
    assert record_store.VALID_STATUSES == ("done", "invalid_tool", "timeout", "oom", "error")


# ---- finished_at（Coordinator 2026-09-30：選填 UTC ISO 8601 字串，Ticket 09 寫入）----
@pytest.mark.parametrize("stamp", ["2026-09-30T00:12:34.567Z", "2026-09-30T00:12:34+00:00"])
def test_validate_accepts_optional_finished_at(stamp):
    validate_record(make_record(finished_at=stamp))


@pytest.mark.parametrize("stamp", [None, 1727654400, "yesterday", "", "2026-09-30T00:12:34"])
def test_validate_rejects_bad_finished_at(stamp):
    with pytest.raises(RecordValidationError, match="finished_at"):
        validate_record(make_record(finished_at=stamp))
