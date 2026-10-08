import csv
import json

import pytest

from multihop_benchmark.reporting import results_aggregator as agg

RUN_ID = "fixture"


def step(hop, *, prompt, output, laya_ms=None):
    entry = {
        "hop": hop,
        "sub_question": f"sub {hop}",
        "evidence_added": [],
        "llm": {"prompt_tokens": prompt, "output_tokens": output, "latency_ms": 5.0},
    }
    if laya_ms is not None:
        entry["laya"] = {"sufficient_p": 0.3, "next_action": "B", "remaining": 1, "latency_ms": laya_ms}
    return entry


def record(dataset, model, condition, qid, *, em, f1, status="done", hops, latency, vram, steps):
    return {
        "run_id": RUN_ID, "dataset": dataset, "model": model, "condition": condition,
        "question_id": qid, "gold": "g", "gold_aliases": [], "prediction": "p",
        "em": em, "f1": f1, "status": status, "hops": hops, "steps": steps,
        "latency_ms": latency, "vram_peak_mb": vram,
    }


# hotpotqa / qwen3.5:4b / rag_laya — 手算值見 test_summary_values_match_hand_calculation
LAYA_GROUP = [
    # 同一題較早的一次失敗嘗試：續跑重做後以最後一行為準
    record("hotpotqa", "qwen3.5:4b", "rag_laya", "q1", em=0, f1=0.0, status="timeout", hops=4,
           latency=300000, vram=9999, steps=[]),
    record("hotpotqa", "qwen3.5:4b", "rag_laya", "q1", em=1, f1=1.0, hops=2, latency=1000, vram=5000,
           steps=[step(1, prompt=100, output=10, laya_ms=10), step(2, prompt=200, output=20, laya_ms=20)]),
    record("hotpotqa", "qwen3.5:4b", "rag_laya", "q2", em=0, f1=0.5, hops=1, latency=2000, vram=6000,
           steps=[step(1, prompt=150, output=15, laya_ms=30)]),
    record("hotpotqa", "qwen3.5:4b", "rag_laya", "q3", em=0, f1=0.0, status="invalid_tool", hops=1,
           latency=3000, vram=5500, steps=[step(1, prompt=50, output=6, laya_ms=40)]),
    record("hotpotqa", "qwen3.5:4b", "rag_laya", "q4", em=0, f1=0.0, status="timeout", hops=3,
           latency=300000, vram=7000, steps=[]),
]

# musique / qwen3.5:27b / recall_llm — 無 Laya 呼叫
LLM_GROUP = [
    record("musique", "qwen3.5:27b", "recall_llm", "m1", em=1, f1=1.0, hops=1, latency=100, vram=20000,
           steps=[step(1, prompt=400, output=40)]),
    record("musique", "qwen3.5:27b", "recall_llm", "m2", em=1, f1=0.8, hops=2, latency=300, vram=21000,
           steps=[step(1, prompt=300, output=30), step(2, prompt=500, output=50)]),
]


def write_records(run_dir, groups):
    records_dir = run_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    for group in groups:
        first = group[0]
        name = f"{first['dataset']}__{first['model'].replace(':', '-')}__{first['condition']}.jsonl"
        with (records_dir / name).open("w", encoding="utf-8") as handle:
            for item in group:
                handle.write(json.dumps(item) + "\n")
    return run_dir


@pytest.fixture
def run_dir(tmp_path):
    return write_records(tmp_path / "runs" / RUN_ID, [LAYA_GROUP, LLM_GROUP])


def read_summary(path):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def row_for(rows, dataset, model, condition):
    (match,) = [r for r in rows if (r["dataset"], r["model"], r["condition"]) == (dataset, model, condition)]
    return match


def test_summary_has_one_row_per_combo_in_fixed_order_with_fixed_columns(run_dir, tmp_path):
    agg.aggregate_run(run_dir, tmp_path / "paper")

    fieldnames, rows = read_summary(run_dir / "summary.csv")
    assert fieldnames == [
        "dataset", "model", "condition", "n", "em", "f1", "latency_p50_ms", "latency_p95_ms",
        "latency_mean_ms", "vram_peak_mb", "prompt_tokens_mean", "output_tokens_mean", "hops_mean",
        "laya_calls_mean", "laya_latency_ms_mean", "invalid_tool_rate", "timeout_rate",
    ]
    assert len(rows) == 60
    keys = [(r["dataset"], r["model"], r["condition"]) for r in rows]
    assert keys[:5] == [
        ("hotpotqa", "qwen3.5:27b", "recall_llm"),
        ("hotpotqa", "qwen3.5:27b", "rag_llm"),
        ("hotpotqa", "qwen3.5:27b", "recall_laya"),
        ("hotpotqa", "qwen3.5:27b", "rag_laya"),
        ("hotpotqa", "qwen3.5:9b", "recall_llm"),
    ]
    assert keys[-1] == ("musique", "qwen3.5:0.8b", "rag_laya")
    assert [k[0] for k in keys[::20]] == ["hotpotqa", "2wiki", "musique"]
    assert [k[1] for k in keys[:20:4]] == [
        "qwen3.5:27b", "qwen3.5:9b", "qwen3.5:4b", "qwen3.5:2b", "qwen3.5:0.8b",
    ]
    assert len(set(keys)) == 60


def test_summary_values_match_hand_calculation(run_dir, tmp_path):
    agg.aggregate_run(run_dir, tmp_path / "paper")
    _, rows = read_summary(run_dir / "summary.csv")

    laya = row_for(rows, "hotpotqa", "qwen3.5:4b", "rag_laya")
    # q1 的最後一行取代較早的 timeout 行 → n=4
    # EM 1/4；F1 (1+0.5+0+0)/4；延遲 [1000,2000,3000,300000] 線性內插 p50=2500、
    # p95 = 3000 + 0.85*(300000-3000) = 255450、平均 76500；VRAM 取最大 7000；
    # prompt tokens (300+150+50+0)/4=125；output (30+15+6+0)/4=12.75；hops 7/4；
    # Laya 呼叫 (2+1+1+0)/4=1；單次 Laya 延遲 (10+20+30+40)/4=25；invalid 1/4、timeout 1/4
    assert {k: v for k, v in laya.items() if k not in ("dataset", "model", "condition")} == {
        "n": "4", "em": "25.0", "f1": "37.5",
        "latency_p50_ms": "2500", "latency_p95_ms": "255450", "latency_mean_ms": "76500",
        "vram_peak_mb": "7000", "prompt_tokens_mean": "125", "output_tokens_mean": "13",
        "hops_mean": "1.75", "laya_calls_mean": "1.00", "laya_latency_ms_mean": "25.0",
        "invalid_tool_rate": "0.250", "timeout_rate": "0.250",
    }

    llm = row_for(rows, "musique", "qwen3.5:27b", "recall_llm")
    # 延遲 [100,300]：p50=200、p95=100+0.95*200=290；tokens (400+800)/2、(40+80)/2；無 Laya 呼叫
    assert {k: v for k, v in llm.items() if k not in ("dataset", "model", "condition")} == {
        "n": "2", "em": "100.0", "f1": "90.0",
        "latency_p50_ms": "200", "latency_p95_ms": "290", "latency_mean_ms": "200",
        "vram_peak_mb": "21000", "prompt_tokens_mean": "600", "output_tokens_mean": "60",
        "hops_mean": "1.50", "laya_calls_mean": "0.00", "laya_latency_ms_mean": "",
        "invalid_tool_rate": "0.000", "timeout_rate": "0.000",
    }


def test_missing_combo_has_n_zero_and_blank_metrics(run_dir, tmp_path):
    agg.aggregate_run(run_dir, tmp_path / "paper")
    _, rows = read_summary(run_dir / "summary.csv")

    missing = row_for(rows, "2wiki", "qwen3.5:0.8b", "rag_llm")
    assert missing["n"] == "0"
    assert all(missing[c] == "" for c in agg.SUMMARY_COLUMNS[4:])
    assert sum(r["n"] != "0" for r in rows) == 2


# ---------------------------------------------------------------- LaTeX 表格

DATASET_ORDER = ("hotpotqa", "2wiki", "musique")
# 每個資料集一張效率表；latency_mean_ms 只在 summary.csv（單欄寬度放不下第 8 個數字欄）
EFFICIENCY_COLUMNS = (
    "latency_p50_ms", "latency_p95_ms", "vram_peak_mb",
    "prompt_tokens_mean", "output_tokens_mean", "hops_mean", "laya_latency_ms_mean",
)
EFFICIENCY_STEMS = tuple(f"efficiency_{d}" for d in DATASET_ORDER)
TABLE_STEMS = ("main_results", *EFFICIENCY_STEMS)
MISSING = "---"  # LaTeX 排版為「—」


def table_rows(tex):
    """回傳 (cells, meta)：每個資料列以行尾註解 `% key=value ...` 標示對應的 summary 組別。"""
    rows = []
    for line in tex.splitlines():
        if "%" not in line or "model=" not in line:
            continue
        body, comment = line.split("%", 1)
        cells = [c.strip() for c in body.strip().removesuffix("\\\\").split("&")]
        rows.append((cells, dict(kv.split("=", 1) for kv in comment.split())))
    return rows


def expected_cell(summary_row, column):
    return summary_row[column] or MISSING


@pytest.fixture
def generated(run_dir, tmp_path):
    paper = tmp_path / "paper"
    agg.aggregate_run(run_dir, paper)
    _, rows = read_summary(run_dir / "summary.csv")
    return paper, rows


def read_table(paper, stem):
    return (paper / "tables" / f"{stem}.tex").read_text(encoding="utf-8")


@pytest.mark.parametrize("stem", TABLE_STEMS)
def test_tables_are_bare_tabular_for_input(generated, stem):
    paper, _ = generated
    tex = read_table(paper, stem)
    body = "\n".join(line.split("%", 1)[0] for line in tex.splitlines()).strip()
    assert body.startswith("\\begin{tabular}") and body.endswith("\\end{tabular}")
    for forbidden in ("\\documentclass", "\\begin{table}", "\\caption", "\\label", "\\centering", "\\begin{document}"):
        assert forbidden not in tex
    assert "\\toprule" in tex and "\\bottomrule" in tex


def test_main_table_cells_trace_back_to_summary(generated):
    paper, rows = generated
    data = table_rows(read_table(paper, "main_results"))

    assert len(data) == 20  # 5 模型 × 4 條件；前兩格為 Model、Condition 標籤
    for cells, meta in data:
        expected = []
        for dataset in DATASET_ORDER:
            summary_row = row_for(rows, dataset, meta["model"], meta["condition"])
            expected += [expected_cell(summary_row, "em"), expected_cell(summary_row, "f1")]
        assert cells[2:] == expected, meta

    (laya_cells, _), = [(c, m) for c, m in data if (m["model"], m["condition"]) == ("qwen3.5:4b", "rag_laya")]
    assert laya_cells[2:] == ["25.0", "37.5", MISSING, MISSING, MISSING, MISSING]
    labels = {m["condition"]: cells[1] for cells, m in data}
    assert labels == {"recall_llm": "Recall+LLM", "rag_llm": "RAG+LLM", "recall_laya": "Recall+Laya", "rag_laya": "RAG+Laya"}


@pytest.mark.parametrize("dataset", DATASET_ORDER)
def test_efficiency_table_cells_trace_back_to_summary(generated, dataset):
    paper, rows = generated
    data = table_rows(read_table(paper, f"efficiency_{dataset}"))

    assert len(data) == 20  # 5 模型 × 4 條件；第一格為 Condition 標籤（模型為分段標題列）
    for cells, meta in data:
        assert meta["dataset"] == dataset
        summary_row = row_for(rows, dataset, meta["model"], meta["condition"])
        assert cells[1:] == [expected_cell(summary_row, c) for c in EFFICIENCY_COLUMNS], meta


def test_efficiency_values_for_known_combo(generated):
    paper, _ = generated
    data = table_rows(read_table(paper, "efficiency_musique"))
    (llm_cells, _), = [(c, m) for c, m in data if (m["model"], m["condition"]) == ("qwen3.5:27b", "recall_llm")]
    assert llm_cells[1:] == ["200", "290", "21000", "600", "60", "1.50", MISSING]


def test_tables_mark_every_missing_combo_with_dash(generated):
    paper, _ = generated
    present = {("hotpotqa", "qwen3.5:4b", "rag_laya"), ("musique", "qwen3.5:27b", "recall_llm")}
    missing = [cells for stem in EFFICIENCY_STEMS for cells, meta in table_rows(read_table(paper, stem))
               if (meta["dataset"], meta["model"], meta["condition"]) not in present]
    assert len(missing) == 58
    assert all(cell == MISSING for cells in missing for cell in cells[1:])


def test_empty_records_directory_still_produces_all_outputs(tmp_path):
    run_dir = tmp_path / "runs" / "empty"
    (run_dir / "records").mkdir(parents=True)

    agg.aggregate_run(run_dir, tmp_path / "paper")

    _, rows = read_summary(run_dir / "summary.csv")
    assert len(rows) == 60 and all(r["n"] == "0" for r in rows)
    data = table_rows((tmp_path / "paper" / "tables" / "main_results.tex").read_text(encoding="utf-8"))
    assert all(cell == MISSING for cells, _ in data for cell in cells[2:])


# ---------------------------------------------------------------- 探針表

PROBE = {
    "run_id": RUN_ID,
    "models": {
        "qwen3.5:27b": {"n": 20, "valid_tool_call_rate": 1.0, "mean_latency_ms": 812.4},
        "qwen3.5:9b": {"n": 20, "valid_tool_call_rate": 0.95, "mean_latency_ms": 401.6},
        "qwen3.5:4b": {"n": 20, "valid_tool_call_rate": 0.85, "mean_latency_ms": 250.0},
        "qwen3.5:0.8b": {"n": 20, "valid_tool_call_rate": 0.4, "mean_latency_ms": 90.2},
    },
}


def probe_rows(paper):
    tex = (paper / "tables" / "probe.tex").read_text(encoding="utf-8")
    return tex, {meta["model"]: cells for cells, meta in table_rows(tex)}


def test_probe_only_builds_probe_table_from_probe_results(tmp_path):
    run_dir = tmp_path / "runs" / RUN_ID
    run_dir.mkdir(parents=True)
    (run_dir / "probe_results.json").write_text(json.dumps(PROBE), encoding="utf-8")

    result = agg.aggregate_run(run_dir, tmp_path / "paper", probe_only=True)

    assert result.tables == [tmp_path / "paper" / "tables" / "probe.tex"]
    assert not (run_dir / "summary.csv").exists()
    tex, rows = probe_rows(tmp_path / "paper")
    assert "\\begin{tabular}" in tex and "\\caption" not in tex and "\\begin{table}" not in tex
    # 欄位：Model, n, 合法工具呼叫率, 平均延遲 (ms)；數值與 probe_results.json 一致
    assert rows["qwen3.5:27b"][1:] == ["20", "1.00", "812"]
    assert rows["qwen3.5:9b"][1:] == ["20", "0.95", "402"]
    assert rows["qwen3.5:0.8b"][1:] == ["20", "0.40", "90"]
    assert rows["qwen3.5:2b"][1:] == [MISSING, MISSING, MISSING]
    assert list(rows) == ["qwen3.5:27b", "qwen3.5:9b", "qwen3.5:4b", "qwen3.5:2b", "qwen3.5:0.8b"]


def test_probe_results_accepts_list_of_models(tmp_path):
    run_dir = tmp_path / "runs" / RUN_ID
    run_dir.mkdir(parents=True)
    listed = [{"model": m, **v} for m, v in PROBE["models"].items()]
    (run_dir / "probe_results.json").write_text(json.dumps({"models": listed}), encoding="utf-8")

    agg.aggregate_run(run_dir, tmp_path / "paper", probe_only=True)

    assert probe_rows(tmp_path / "paper")[1]["qwen3.5:4b"][1:] == ["20", "0.85", "250"]


def test_probe_only_without_probe_results_fails_clearly(tmp_path):
    run_dir = tmp_path / "runs" / RUN_ID
    run_dir.mkdir(parents=True)
    with pytest.raises(agg.AggregateError, match="probe_results.json"):
        agg.aggregate_run(run_dir, tmp_path / "paper", probe_only=True)


def test_full_aggregate_includes_probe_table_when_probe_results_exist(run_dir, tmp_path):
    (run_dir / "probe_results.json").write_text(json.dumps(PROBE), encoding="utf-8")

    result = agg.aggregate_run(run_dir, tmp_path / "paper")

    assert {p.name for p in result.tables} == {f"{stem}.tex" for stem in (*TABLE_STEMS, "probe")}


# ---------------------------------------------------------------- 圖

FIGURE_STEMS = {
    "em_vs_size_hotpotqa", "em_vs_size_2wiki", "em_vs_size_musique", "latency_vs_size", "hops_distribution",
}
CONDITION_LEGEND = {"Recall+LLM", "RAG+LLM", "Recall+Laya", "RAG+Laya"}


def test_figures_are_written_as_non_empty_pdfs(run_dir, tmp_path):
    result = agg.aggregate_run(run_dir, tmp_path / "paper")

    assert {p.stem for p in result.figures} == FIGURE_STEMS
    for path in result.figures:
        assert path.parent == tmp_path / "paper" / "figures" and path.suffix == ".pdf"
        data = path.read_bytes()
        assert data.startswith(b"%PDF") and len(data) > 1000


@pytest.mark.parametrize("with_records", [True, False])
def test_every_figure_has_condition_legend_and_labelled_axes_with_units(run_dir, with_records):
    records, _ = agg.load_records(run_dir / "records") if with_records else ([], 0)
    figures = agg.build_figures(agg.summarize(records), records)

    assert set(figures) == FIGURE_STEMS
    for stem, figure in figures.items():
        legends = [ax.get_legend() for ax in figure.axes if ax.get_legend()] + list(figure.legends)
        assert len(legends) == 1, stem
        assert {t.get_text() for t in legends[0].get_texts()} == CONDITION_LEGEND, stem
        # 軸標籤附單位；小倍數圖共用 x 軸時只有最下方一格標 x
        x_labels = [ax.get_xlabel() for ax in figure.axes if ax.get_xlabel()]
        y_labels = [ax.get_ylabel() for ax in figure.axes]
        assert x_labels, stem
        for label in x_labels + y_labels:
            assert label and "(" in label and ")" in label, (stem, label)


def test_every_legend_lies_inside_the_figure_canvas(run_dir):
    """R8 review B-1：圖例不得超出圖檔頁寬（曾因 latency_vs_size 的四欄圖例被左右裁切）。"""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    records, _ = agg.load_records(run_dir / "records")
    figures = agg.build_figures(agg.summarize(records), records)

    for stem, figure in figures.items():
        canvas_agg = FigureCanvasAgg(figure)   # constrained layout 在繪製時才定位圖例
        canvas_agg.draw()
        renderer = canvas_agg.get_renderer()
        canvas = figure.bbox
        for legend in [ax.get_legend() for ax in figure.axes if ax.get_legend()] + list(figure.legends):
            box = legend.get_window_extent(renderer)
            assert canvas.x0 <= box.x0 and box.x1 <= canvas.x1, (stem, box.x0, box.x1, canvas.x1)
            assert canvas.y0 <= box.y0 and box.y1 <= canvas.y1, (stem, box.y0, box.y1, canvas.y1)


def test_hops_distribution_is_share_of_questions_per_condition(run_dir):
    records, _ = agg.load_records(run_dir / "records")

    shares = agg.hops_distribution(records)

    # rag_laya 4 題 hops = 2,1,1,3；recall_llm 2 題 hops = 1,2
    assert shares["rag_laya"] == {1: 50.0, 2: 25.0, 3: 25.0, 4: 0.0}
    assert shares["recall_llm"] == {1: 50.0, 2: 50.0, 3: 0.0, 4: 0.0}
    assert shares["rag_llm"] == {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}


# ---------------------------------------------------------------- records 讀取邊界


def test_condition_and_model_spellings_are_normalised(tmp_path):
    variant = [dict(r, condition="rag+laya", model="qwen3.5-4b") for r in LAYA_GROUP]
    run_dir = write_records(tmp_path / "runs" / RUN_ID, [variant])

    agg.aggregate_run(run_dir, tmp_path / "paper")

    _, rows = read_summary(run_dir / "summary.csv")
    assert row_for(rows, "hotpotqa", "qwen3.5:4b", "rag_laya")["n"] == "4"


def test_unknown_combo_values_fail_loudly(tmp_path):
    run_dir = write_records(tmp_path / "runs" / RUN_ID, [[dict(LLM_GROUP[0], condition="oracle")]])

    with pytest.raises(agg.AggregateError, match="oracle"):
        agg.aggregate_run(run_dir, tmp_path / "paper")


def test_truncated_last_line_is_skipped_and_counted(run_dir, tmp_path):
    (path,) = [p for p in (run_dir / "records").iterdir() if p.name.startswith("musique")]
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"run_id": "fixture", "dataset": "musi')

    result = agg.aggregate_run(run_dir, tmp_path / "paper")

    assert result.skipped_lines == 1 and result.record_count == 6


def append_raw(run_dir, text):
    (path,) = [p for p in (run_dir / "records").iterdir() if p.name.startswith("musique")]
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)
    return path


def test_corrupt_complete_line_fails_with_file_and_line_and_writes_nothing(run_dir, tmp_path):
    # 損壞行之後還有完整的行 → 不是 append 中斷，必須失敗，而不是從分母靜默消失
    path = append_raw(run_dir, '{"run_id": "fixture", "dataset": "musi\n' + json.dumps(LLM_GROUP[0]) + "\n")

    with pytest.raises(agg.AggregateError, match=rf"{path.name}:3"):
        agg.aggregate_run(run_dir, tmp_path / "paper")

    assert not (run_dir / "summary.csv").exists()
    assert not (tmp_path / "paper").exists()


def test_corrupt_last_line_terminated_by_newline_fails(run_dir, tmp_path):
    # 以換行結尾代表該行已寫完；寫完卻無法解析是資料損壞，不是 append 中斷
    path = append_raw(run_dir, '{"run_id": "fixture", "dataset": "musi\n')

    with pytest.raises(agg.AggregateError, match=rf"{path.name}:3"):
        agg.aggregate_run(run_dir, tmp_path / "paper")


def test_complete_last_line_without_newline_is_counted(run_dir, tmp_path):
    append_raw(run_dir, json.dumps(dict(LLM_GROUP[0], question_id="m3")))

    result = agg.aggregate_run(run_dir, tmp_path / "paper")

    assert result.skipped_lines == 0 and result.record_count == 7


def test_full_aggregate_removes_stale_probe_table_from_previous_run(run_dir, tmp_path):
    paper = tmp_path / "paper"
    other = tmp_path / "runs" / "other"
    other.mkdir(parents=True)
    (other / "probe_results.json").write_text(json.dumps(PROBE), encoding="utf-8")
    agg.aggregate_run(other, paper, probe_only=True)
    assert (paper / "tables" / "probe.tex").exists()

    result = agg.aggregate_run(run_dir, paper)  # 本 run 沒有 probe_results.json

    assert not (paper / "tables" / "probe.tex").exists()
    assert result.removed == [paper / "tables" / "probe.tex"]
    assert "probe.tex" not in {p.name for p in result.tables}
