import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from multihop_benchmark.reporting.progress_report_writer import collect_progress, write_progress_report

FIXTURE_RECORDS = Path(__file__).parent / "fixtures" / "report_run" / "records"
# 寫入順序（finished_at 與 mtime 皆由舊到新）：hotpotqa 組 → 2wiki 組 → musique 組（最新＝進行中）
FIXTURE_ORDER = (
    "hotpotqa__qwen3.5-4b__rag_laya.jsonl",
    "2wiki__qwen3.5-27b__rag_llm.jsonl",
    "musique__qwen3.5-0.8b__recall_llm.jsonl",
)
NOW = datetime(2026, 9, 29, 12, 0, 0, tzinfo=timezone(timedelta(hours=8)))
APPROXIMATE = "順序為近似（record 無 finished_at）"


@pytest.fixture
def data_root(tmp_path):
    return tmp_path / "laya_data"


def install_fixture(data_root, run_id="fixture", *, finished_at=True):
    """複製 fixture records；finished_at=False 時移除該欄位（模擬舊版 runner 的 record）。"""
    records_dir = data_root / "runs" / run_id / "records"
    records_dir.mkdir(parents=True)
    for offset, name in enumerate(FIXTURE_ORDER):
        target = records_dir / name
        if finished_at:
            shutil.copyfile(FIXTURE_RECORDS / name, target)
        else:
            rows = [json.loads(line) for line in (FIXTURE_RECORDS / name).read_text(encoding="utf-8").splitlines()]
            target.write_text("".join(json.dumps({k: v for k, v in row.items() if k != "finished_at"}) + "\n"
                                      for row in rows), encoding="utf-8")
        os.utime(target, (1_700_000_000 + offset * 100, 1_700_000_000 + offset * 100))
    return records_dir


def record(dataset, model, condition, question_id, status, *, em=0, f1=0.0, latency_ms=1000, finished_at=None, **extra):
    row = {
        "run_id": "fixture", "dataset": dataset, "model": model, "condition": condition,
        "question_id": question_id, "gold": "g", "gold_aliases": [], "prediction": "p", "em": em, "f1": f1,
        "status": status, "hops": 1, "steps": [], "latency_ms": latency_ms, "vram_peak_mb": 1, **extra,
    }
    if finished_at:
        row["finished_at"] = finished_at
    return row


def section(markdown, heading):
    """回傳 `## heading` 到下一個 `## ` 之間的內容。"""
    start = markdown.index(f"## {heading}")
    end = markdown.find("\n## ", start + 1)
    return markdown[start : end if end != -1 else len(markdown)]


def table_row(markdown, dataset, model, condition):
    prefix = f"| {dataset} | {model} | {condition} |"
    rows = [line for line in markdown.splitlines() if line.startswith(prefix)]
    assert len(rows) == 1, rows
    return [cell.strip() for cell in rows[0].strip("|").split("|")]


def error_ids(markdown):
    rows = [line for line in section(markdown, "最近 10 筆錯誤").splitlines() if line.startswith("| 20")]
    return [row.split("|")[3].strip() for row in rows]


def recent_ids(markdown):
    recent = section(markdown, "最近 3 題摘要")
    return [line.split("/ `")[1].split("`")[0] for line in recent.splitlines() if line[:2] in ("1.", "2.", "3.")]


def test_progress_report_has_every_section(data_root):
    install_fixture(data_root)

    path = write_progress_report(data_root, "fixture", now=NOW)

    assert path == data_root / "reports" / "PROGRESS.md"
    markdown = path.read_text(encoding="utf-8")
    assert "產生時間：2026-09-29T12:00:00+08:00（UTC 2026-09-29T04:00:00+00:00）" in markdown
    for heading in ("目前進行中的組別", "整體完成率與 ETA", "各組即時指標", "最近 10 筆錯誤", "最近 3 題摘要"):
        assert f"## {heading}" in markdown
    assert APPROXIMATE not in markdown  # 全部 record 都有 finished_at


def test_progress_table_lists_all_60_combos_and_marks_not_started(data_root):
    install_fixture(data_root)

    markdown = write_progress_report(data_root, "fixture", now=NOW).read_text(encoding="utf-8")

    table = section(markdown, "各組即時指標")
    rows = [line for line in table.splitlines() if line.startswith("| ") and "qwen3.5:" in line]
    assert len(rows) == 60
    assert sum("未開始" in row for row in rows) == 57
    assert table_row(markdown, "hotpotqa", "qwen3.5:9b", "recall_llm")[3:] == ["0", "0", "—", "—", "—", "0", "未開始"]


def test_progress_numbers_match_hand_computation(data_root):
    install_fixture(data_root)

    markdown = write_progress_report(data_root, "fixture", now=NOW).read_text(encoding="utf-8")

    # 欄位：done、已嘗試、EM、F1、延遲 p50、非 done、狀態
    # hotpotqa 組：done 2（hq1,hq2）／嘗試 4；EM (1+0+0+0)/4；F1 (1+0.5+0+0)/4；延遲中位數 (2000+3000)/2 ms
    assert table_row(markdown, "hotpotqa", "qwen3.5:4b", "rag_laya")[3:] == [
        "2", "4", "0.250", "0.375", "2.50 s", "2", "未完成",
    ]
    # musique 組：EM 1.0、F1 (1+0.8)/2、延遲中位數 (500+1500)/2 ms；最新寫入 → 進行中
    assert table_row(markdown, "musique", "qwen3.5:0.8b", "recall_llm")[3:] == [
        "2", "2", "1.000", "0.900", "1.00 s", "0", "進行中",
    ]
    # 2wiki 組：done 5／嘗試 6；EM 3/6；F1 (1+1+0.4+0+1+0)/6；延遲中位數 (6000+7000)/2 ms
    assert table_row(markdown, "2wiki", "qwen3.5:27b", "rag_llm")[3:] == [
        "5", "6", "0.500", "0.567", "6.50 s", "1", "未完成",
    ]
    overall = section(markdown, "整體完成率與 ETA")
    # done 9 / (60 × 200) = 0.075% → 0.08%；done 題平均延遲 38000/9 ms；剩 11991 題 × 4.2222 s = 50628.67 s
    assert "已完成（status=done）9 / 12000 題（0.08%）；已嘗試 12 題，其中非 done 3 題" in overall
    assert "已完成題平均延遲：4.22 s" in overall
    assert "ETA：0 天 14 小時 3 分" in overall
    assert "預計完成：2026-09-30T02:03:49+08:00" in overall
    current = section(markdown, "目前進行中的組別")
    assert "musique__qwen3.5:0.8b__recall_llm（done 2 / 200，已嘗試 2）" in current


def test_recent_errors_are_non_done_records_newest_first(data_root):
    install_fixture(data_root)

    markdown = write_progress_report(data_root, "fixture", now=NOW).read_text(encoding="utf-8")

    assert error_ids(markdown) == ["wq4", "hq4", "hq3"]
    errors = section(markdown, "最近 10 筆錯誤")
    assert "| 2026-09-29T02:00:04+00:00 | 2wiki__qwen3.5:27b__rag_llm | wq4 | oom | CUDA out of memory" in errors
    assert "| invalid_tool | unknown tool call: search_web |" in errors
    assert "| timeout |" in errors


def test_recent_questions_show_prediction_gold_correctness_and_hops(data_root):
    install_fixture(data_root)

    markdown = write_progress_report(data_root, "fixture", now=NOW).read_text(encoding="utf-8")

    assert recent_ids(markdown) == ["mq2", "mq1", "wq6"]
    recent = section(markdown, "最近 3 題摘要")
    assert "預測：Curie" in recent and "gold：Marie Curie" in recent
    assert "對錯：對（EM 1，F1 0.800）；跳數：2" in recent
    assert "對錯：錯（EM 0，F1 0.000）；跳數：4" in recent


def test_rerun_appended_to_old_combo_is_newest_by_finished_at(data_root):
    """續跑回到最舊的 hotpotqa 組重做 hq3：最近清單依 finished_at 跨檔排序，不因整檔 mtime 變新而錯序。"""
    records_dir = install_fixture(data_root)
    hotpot = records_dir / FIXTURE_ORDER[0]
    rerun = record("hotpotqa", "qwen3.5:4b", "rag_laya", "hq3", "done", em=1, f1=1.0, latency_ms=4000,
                   finished_at="2026-09-29T04:00:00+00:00")
    with hotpot.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(rerun) + "\n")
    os.utime(hotpot, (1_700_001_000, 1_700_001_000))

    markdown = write_progress_report(data_root, "fixture", now=NOW).read_text(encoding="utf-8")

    assert recent_ids(markdown) == ["hq3", "mq2", "mq1"]
    assert error_ids(markdown) == ["wq4", "hq4", "hq3"]  # 舊的 timeout 仍留在錯誤歷史
    assert "hotpotqa__qwen3.5:4b__rag_laya（done 3 / 200，已嘗試 4）" in section(markdown, "目前進行中的組別")
    # hq3 改為 done：EM 2/4、F1 (1+0.5+1+0)/4、延遲 {1000,3000,4000,2000} 中位數 2500 ms
    assert table_row(markdown, "hotpotqa", "qwen3.5:4b", "rag_laya")[3:] == [
        "3", "4", "0.500", "0.625", "2.50 s", "1", "進行中",
    ]
    assert APPROXIMATE not in markdown


def test_records_without_finished_at_fall_back_to_mtime_and_are_marked_approximate(data_root):
    install_fixture(data_root, finished_at=False)

    markdown = write_progress_report(data_root, "fixture", now=NOW).read_text(encoding="utf-8")

    assert recent_ids(markdown) == ["mq2", "mq1", "wq6"]
    assert APPROXIMATE in section(markdown, "最近 10 筆錯誤")
    assert APPROXIMATE in section(markdown, "最近 3 題摘要")
    assert "| — | 2wiki__qwen3.5:27b__rag_llm | wq4 | oom |" in section(markdown, "最近 10 筆錯誤")


def test_target_reached_with_non_done_is_not_finished(data_root):
    """達到每組題數但含 non-done：續跑仍會重做，不得顯示已完成或 ETA 0。"""
    run_dir = data_root / "runs" / "fixture"
    (run_dir / "records").mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({
        "datasets": ["hotpotqa"], "models": ["qwen3.5:4b"], "conditions": ["rag_laya", "rag_llm"],
        "questions_per_combo": 2,
    }), encoding="utf-8")
    rows = [
        record("hotpotqa", "qwen3.5:4b", "rag_llm", "q1", "done", em=1, f1=1.0, latency_ms=60000,
               finished_at="2026-09-29T01:00:01+00:00"),
        record("hotpotqa", "qwen3.5:4b", "rag_llm", "q2", "done", latency_ms=60000,
               finished_at="2026-09-29T01:00:02+00:00"),
        record("hotpotqa", "qwen3.5:4b", "rag_laya", "q1", "done", em=1, f1=1.0, latency_ms=120000,
               finished_at="2026-09-29T02:00:01+00:00"),
        record("hotpotqa", "qwen3.5:4b", "rag_laya", "q2", "timeout", latency_ms=300000,
               finished_at="2026-09-29T02:00:02+00:00", error="timeout"),
    ]
    (run_dir / "records" / "mixed.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    markdown = write_progress_report(data_root, "fixture", now=NOW).read_text(encoding="utf-8")

    assert table_row(markdown, "hotpotqa", "qwen3.5:4b", "rag_llm")[3:] == [
        "2", "2", "0.500", "0.500", "60.00 s", "0", "已完成",
    ]
    assert table_row(markdown, "hotpotqa", "qwen3.5:4b", "rag_laya")[3:] == [
        "1", "2", "0.500", "0.500", "210.00 s", "1", "進行中",
    ]
    overall = section(markdown, "整體完成率與 ETA")
    # done 3 / 4；done 題平均延遲 (60+60+120)/3 = 80 s；剩 1 題 → ETA 80 s
    assert "已完成（status=done）3 / 4 題（75.00%）；已嘗試 4 題，其中非 done 1 題" in overall
    assert "已完成題平均延遲：80.00 s" in overall
    assert "ETA：0 天 0 小時 1 分" in overall
    assert "所有題目已有結果" not in markdown  # 仍有進行中組別：非終態
    assert "hotpotqa__qwen3.5:4b__rag_laya（done 1 / 2，已嘗試 2）" in section(markdown, "目前進行中的組別")


def test_question_text_is_looked_up_from_prepared_sample(data_root):
    install_fixture(data_root)
    sample = data_root / "datasets" / "musique" / "sample_200.jsonl"
    sample.parent.mkdir(parents=True)
    sample.write_text(json.dumps({"id": "mq2", "question": "Who discovered polonium?"}) + "\n", encoding="utf-8")

    markdown = write_progress_report(data_root, "fixture", now=NOW).read_text(encoding="utf-8")

    assert "問題：Who discovered polonium?" in section(markdown, "最近 3 題摘要")


@pytest.mark.parametrize("create_dir", [False, True], ids=["records-missing", "records-empty"])
def test_missing_or_empty_records_report_no_data(data_root, create_dir):
    if create_dir:
        (data_root / "runs" / "fixture" / "records").mkdir(parents=True)

    markdown = write_progress_report(data_root, "fixture", now=NOW).read_text(encoding="utf-8")

    assert "尚無資料" in markdown
    assert "產生時間：2026-09-29T12:00:00+08:00" in markdown


def test_trailing_partial_line_is_skipped_silently(data_root):
    records_dir = install_fixture(data_root)
    with (records_dir / FIXTURE_ORDER[2]).open("a", encoding="utf-8") as handle:
        handle.write('{"run_id": "fixture", "dataset": "musi')  # run 正在 append，尚無換行

    snapshot = collect_progress(data_root, "fixture", now=NOW)
    markdown = write_progress_report(data_root, "fixture", snapshot=snapshot).read_text(encoding="utf-8")

    assert snapshot.invalid_lines == []
    assert "無法解析" not in markdown
    assert table_row(markdown, "musique", "qwen3.5:0.8b", "recall_llm")[3] == "2"


def test_corrupt_lines_are_counted_and_warned_above_errors(data_root):
    records_dir = install_fixture(data_root)
    hotpot = records_dir / FIXTURE_ORDER[0]
    lines = hotpot.read_text(encoding="utf-8").splitlines(keepends=True)
    lines.insert(1, '{"run_id": "fixture", "dat\n')  # 中斷留下、之後又被 append 覆蓋在中間的殘行
    lines.append("not json at all\n")
    hotpot.write_bytes("".join(lines).encode("utf-8") + b'{"x": "\xe4\xb8\n')  # 截斷的 UTF-8，已換行
    os.utime(hotpot, (1_700_000_000, 1_700_000_000))

    snapshot = collect_progress(data_root, "fixture", now=NOW)
    markdown = write_progress_report(data_root, "fixture", snapshot=snapshot).read_text(encoding="utf-8")

    assert snapshot.invalid_lines == [(FIXTURE_ORDER[0], 2), (FIXTURE_ORDER[0], 6), (FIXTURE_ORDER[0], 7)]
    errors = section(markdown, "最近 10 筆錯誤")
    warning = errors.index("有 3 行 record 無法解析")
    assert warning < errors.index("| 時間 |")
    assert f"{FIXTURE_ORDER[0]}:2" in errors and f"{FIXTURE_ORDER[0]}:7" in errors
    assert table_row(markdown, "hotpotqa", "qwen3.5:4b", "rag_laya")[3:5] == ["2", "4"]  # 有效 record 不受影響


def test_manifest_limit_sets_questions_per_combo(data_root):
    install_fixture(data_root)
    (data_root / "runs" / "fixture" / "manifest.json").write_text(json.dumps({"limit": 2}), encoding="utf-8")

    markdown = write_progress_report(data_root, "fixture", now=NOW).read_text(encoding="utf-8")

    assert "已完成（status=done）6 / 120 題（5.00%）" in section(markdown, "整體完成率與 ETA")
    assert table_row(markdown, "musique", "qwen3.5:0.8b", "recall_llm")[-1] == "已完成"
    assert table_row(markdown, "hotpotqa", "qwen3.5:4b", "rag_laya")[-1] == "已完成"  # done 2 = 目標 2
    assert table_row(markdown, "2wiki", "qwen3.5:27b", "rag_llm")[-1] == "已完成"
    assert "無（最新一筆 record 所屬組別已完成：musique__qwen3.5:0.8b__recall_llm）" in markdown


def install_terminal_run(data_root):
    """每組已嘗試＝目標題數（所有題目已有結果）、最新一筆所屬組別全 done、另有兩組含非 done。"""
    run_dir = data_root / "runs" / "fixture"
    (run_dir / "records").mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({
        "datasets": ["hotpotqa"], "models": ["qwen3.5:4b"], "conditions": ["rag_laya", "rag_llm", "recall_llm"],
        "questions_per_combo": 2,
    }), encoding="utf-8")
    rows = [
        record("hotpotqa", "qwen3.5:4b", "rag_laya", "q1", "invalid_tool", latency_ms=1000,
               finished_at="2026-09-29T01:00:01+00:00", error="unknown tool call"),
        record("hotpotqa", "qwen3.5:4b", "rag_laya", "q2", "invalid_tool", latency_ms=1000,
               finished_at="2026-09-29T01:00:02+00:00", error="unknown tool call"),
        record("hotpotqa", "qwen3.5:4b", "rag_llm", "q1", "error", latency_ms=1000,
               finished_at="2026-09-29T01:00:03+00:00", error="boom"),
        record("hotpotqa", "qwen3.5:4b", "rag_llm", "q2", "done", em=1, f1=1.0, latency_ms=60000,
               finished_at="2026-09-29T01:00:04+00:00"),
        record("hotpotqa", "qwen3.5:4b", "recall_llm", "q1", "done", latency_ms=60000,
               finished_at="2026-09-29T01:00:05+00:00"),
        record("hotpotqa", "qwen3.5:4b", "recall_llm", "q2", "done", latency_ms=60000,
               finished_at="2026-09-29T01:00:06+00:00"),
    ]
    (run_dir / "records" / "all.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_all_questions_have_results_shows_terminal_eta(data_root):
    """run 結束（每組都已嘗試滿、無進行中組別）但含非 done：不再宣稱尚需時間或預計完成時間。"""
    install_terminal_run(data_root)

    markdown = write_progress_report(data_root, "fixture", now=NOW).read_text(encoding="utf-8")

    overall = section(markdown, "整體完成率與 ETA")
    assert (
        "已完成（status=done）3 / 6 題（50.00%）；已嘗試 6 / 6 題"
        "（所有題目已有結果；非 done 3 題：invalid_tool 2、error 1）"
    ) in overall
    assert "ETA：0（所有題目已有結果；續跑只會重做非 done 題）" in overall
    assert "預計完成" not in markdown
    assert "已完成題平均延遲：60.00 s" in overall
    assert "無（最新一筆 record 所屬組別已完成：hotpotqa__qwen3.5:4b__recall_llm）" in section(markdown, "目前進行中的組別")
    assert table_row(markdown, "hotpotqa", "qwen3.5:4b", "rag_laya")[-1] == "未完成"  # 組別狀態語意不變
