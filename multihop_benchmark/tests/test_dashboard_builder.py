import json
import re

from multihop_benchmark.reporting.dashboard_builder import write_dashboard
from test_progress_report_writer import APPROXIMATE, FIXTURE_ORDER, NOW, install_fixture, install_terminal_run, record


def embedded_data(html):
    match = re.search(r'<script id="dashboard-data" type="application/json">(.*?)</script>', html, re.S)
    assert match, "找不到內嵌 JSON"
    return json.loads(match.group(1))


def test_dashboard_is_self_contained_with_refresh_and_embedded_data(tmp_path):
    data_root = tmp_path / "laya_data"
    install_fixture(data_root)

    path = write_dashboard(data_root, "fixture", now=NOW)

    assert path == data_root / "reports" / "dashboard.html"
    html = path.read_text(encoding="utf-8")
    assert '<meta http-equiv="refresh" content="60">' in html
    assert "2026-09-29T12:00:00+08:00" in html
    # 不引用本機其他檔案或任何外部資源
    assert not re.search(r"""\b(src|href)\s*=|@import|url\(|file:|https?://""", html, re.I)
    data = embedded_data(html)
    assert len(data["combos"]) == 60
    hotpot = next(c for c in data["combos"] if c["combo"] == "hotpotqa__qwen3.5:4b__rag_laya")
    assert (hotpot["done"], hotpot["attempted"], hotpot["em"], hotpot["latency_p50_ms"]) == (2, 4, 0.25, 2500)
    assert (data["overall"]["done"], data["overall"]["attempted"], data["overall"]["target_total"]) == (9, 12, 12000)


def test_dashboard_has_table_and_two_charts(tmp_path):
    data_root = tmp_path / "laya_data"
    install_fixture(data_root)

    html = write_dashboard(data_root, "fixture", now=NOW).read_text(encoding="utf-8")

    assert html.count("<tr class=\"combo-row") == 60
    assert 'id="chart-cumulative-em"' in html and 'id="chart-latency"' in html
    data = embedded_data(html)
    # 累積 EM：2wiki 27b rag_llm 依序 EM 1,1,0,0(oom),1,0 → 1, 1, 2/3, 1/2, 3/5, 1/2
    curve = data["cumulative_em"]["qwen3.5:27b"]["rag_llm"]
    assert [round(value, 4) for value in curve] == [1.0, 1.0, 0.6667, 0.5, 0.6, 0.5]
    assert html.count("<polyline") >= 3  # 三個有資料的模型×條件各一條曲線


def test_dashboard_without_records_shows_no_data(tmp_path):
    html = write_dashboard(tmp_path / "laya_data", "fixture", now=NOW).read_text(encoding="utf-8")

    assert "尚無資料" in html
    assert '<meta http-equiv="refresh" content="60">' in html


def test_dashboard_recent_lists_follow_finished_at_after_rerun_of_old_combo(tmp_path):
    data_root = tmp_path / "laya_data"
    records_dir = install_fixture(data_root)
    rerun = record("hotpotqa", "qwen3.5:4b", "rag_laya", "hq3", "done", em=1, f1=1.0, latency_ms=4000,
                   finished_at="2026-09-29T04:00:00+00:00")
    with (records_dir / FIXTURE_ORDER[0]).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(rerun) + "\n")

    html = write_dashboard(data_root, "fixture", now=NOW).read_text(encoding="utf-8")

    data = embedded_data(html)
    assert [item["question_id"] for item in data["recent_questions"]] == ["hq3", "mq2", "mq1"]
    assert [item["question_id"] for item in data["recent_errors"]] == ["wq4", "hq4", "hq3"]
    assert data["order_is_approximate"] is False and APPROXIMATE not in html


def test_dashboard_marks_approximate_order_and_corrupt_lines(tmp_path):
    data_root = tmp_path / "laya_data"
    records_dir = install_fixture(data_root, finished_at=False)
    with (records_dir / FIXTURE_ORDER[1]).open("a", encoding="utf-8") as handle:
        handle.write("garbage\n")

    html = write_dashboard(data_root, "fixture", now=NOW).read_text(encoding="utf-8")

    data = embedded_data(html)
    assert data["order_is_approximate"] is True and APPROXIMATE in html
    assert data["invalid_lines"] == [[FIXTURE_ORDER[1], 7]]
    assert "有 1 行 record 無法解析" in html


def tile(html, label):
    match = re.search(rf'<div class="label">{label}</div><div class="value">(.*?)</div>', html)
    assert match, f"找不到卡片 {label}"
    return match.group(1)


def test_dashboard_terminal_state_shows_no_remaining_eta(tmp_path):
    data_root = tmp_path / "laya_data"
    install_terminal_run(data_root)

    html = write_dashboard(data_root, "fixture", now=NOW).read_text(encoding="utf-8")

    assert tile(html, "ETA") == "0（所有題目已有結果；續跑只會重做非 done 題）"
    assert tile(html, "預計完成") == "不適用（所有題目已有結果）"
    assert tile(html, "目前組別").startswith("無（")
    assert "已嘗試 6 / 6 題（所有題目已有結果；非 done 3 題：invalid_tool 2、error 1）" in tile(html, "整體完成率")
    overall = embedded_data(html)["overall"]
    assert overall["all_have_results"] is True
    assert (overall["eta_seconds"], overall["finish_at"]) == (0, None)
    assert overall["non_done_by_status"] == {"invalid_tool": 2, "error": 1}


def test_dashboard_in_progress_keeps_eta_and_finish_time(tmp_path):
    data_root = tmp_path / "laya_data"
    install_fixture(data_root)

    html = write_dashboard(data_root, "fixture", now=NOW).read_text(encoding="utf-8")

    assert tile(html, "ETA") == "0 天 14 小時 3 分"
    assert tile(html, "預計完成") == "2026-09-30T02:03:49+08:00"
    assert "所有題目已有結果" not in html
    assert embedded_data(html)["overall"]["all_have_results"] is False
