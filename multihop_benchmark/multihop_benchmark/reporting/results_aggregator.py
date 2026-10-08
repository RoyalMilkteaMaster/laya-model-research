"""匯總 records → summary.csv、論文 LaTeX 表格與 matplotlib 圖（架構資料流 5）。

輸入：Data Root `runs/<run_id>/records/*.jsonl`（直接以 JSONL 讀取）與選用的 `probe_results.json`。
輸出：`runs/<run_id>/summary.csv`、`paper/tables/*.tex`、`paper/figures/*.pdf`。
- 表格（bare tabular，供 paper/main.tex 的 \tableorplaceholder 使用）：`main_results`（EM／F1）、
  `efficiency_hotpotqa`／`efficiency_2wiki`／`efficiency_musique`（每集一張，單欄放得下）、
  `probe`（probe_results.json 存在時；`probe_only=True` 只產生這張）。
- 圖：`em_vs_size_<dataset>`、`latency_vs_size`（每集一格，p50）、`hops_distribution`（跨集與模型合併）。

summary.csv 約定
- 60 列固定順序：dataset（hotpotqa → 2wiki → musique）→ model（27b → 0.8b，由大到小）→
  condition（recall_llm 無輔助 → rag_llm 只 RAG → recall_laya 只 Laya → rag_laya RAG＋Laya）。
  condition 名稱為 `<證據來源>_<控制者>`；records 中寫成 `rag+laya`、`rag-laya` 也視為同一條件。
- 同一組同一 question_id 出現多行（續跑重做未完成題）時，以檔案中最後一行為準。
- n 為該組題數（所有 status）；缺組時 n=0，其餘欄位空白。
- em、f1 為百分比（0–100）；延遲百分位採線性內插；vram_peak_mb 為該組最大值；
  token 為每題 `steps[].llm` 加總後再取平均；laya_calls_mean 為每題含 `laya` 的步數平均；
  laya_latency_ms_mean 為單次 Laya 呼叫延遲平均（無呼叫時空白）；兩個 rate 為 0–1 比例。
- 數值以 VALUE_FORMATS 的固定小數位寫入；論文表格直接沿用同一字串，
  因此表內每個數字都能在 summary.csv 對應列逐字找到。表格中缺組或不適用以「—」標示。
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402

DATASETS = ("hotpotqa", "2wiki", "musique")
MODELS = ("qwen3.5:27b", "qwen3.5:9b", "qwen3.5:4b", "qwen3.5:2b", "qwen3.5:0.8b")
CONDITIONS = ("recall_llm", "rag_llm", "recall_laya", "rag_laya")

SUMMARY_COLUMNS = (
    "dataset", "model", "condition", "n", "em", "f1", "latency_p50_ms", "latency_p95_ms",
    "latency_mean_ms", "vram_peak_mb", "prompt_tokens_mean", "output_tokens_mean", "hops_mean",
    "laya_calls_mean", "laya_latency_ms_mean", "invalid_tool_rate", "timeout_rate",
)
VALUE_FORMATS = {
    "em": ".1f", "f1": ".1f",
    "latency_p50_ms": ".0f", "latency_p95_ms": ".0f", "latency_mean_ms": ".0f",
    "vram_peak_mb": ".0f", "prompt_tokens_mean": ".0f", "output_tokens_mean": ".0f",
    "hops_mean": ".2f", "laya_calls_mean": ".2f", "laya_latency_ms_mean": ".1f",
    "invalid_tool_rate": ".3f", "timeout_rate": ".3f",
}


class AggregateError(RuntimeError):
    """records 或 probe_results.json 無法匯總。"""


@dataclass
class AggregateResult:
    summary_path: Path | None = None
    tables: list[Path] = field(default_factory=list)
    figures: list[Path] = field(default_factory=list)
    record_count: int = 0
    skipped_lines: int = 0
    removed: list[Path] = field(default_factory=list)


# ---------------------------------------------------------------- records


def _canonical_condition(value: object) -> str | None:
    name = re.sub(r"[+\-/ ]", "_", str(value).strip().lower())
    return name if name in CONDITIONS else None


def _canonical_model(value: object) -> str | None:
    match = re.fullmatch(r"(qwen3\.5)[:_\-](\d+(?:\.\d+)?b)", str(value).strip().lower())
    name = f"{match[1]}:{match[2]}" if match else None
    return name if name in MODELS else None


def load_records(records_dir: Path) -> tuple[list[dict], int]:
    """讀取全部 records；回傳（去重後的 records, 略過的未完成尾行數）。

    只容忍 append 中斷：檔案最後一行沒有換行結尾且無法解析時略過並計數；
    其他無法解析的行（含以換行結尾的最後一行）視為資料損壞，帶檔名與行號拋出 AggregateError。
    """
    latest: dict[tuple[str, str, str, str], dict] = {}
    skipped = 0
    unknown: set[str] = set()
    for path in sorted(Path(records_dir).glob("*.jsonl")):
        lines = path.read_text(encoding="utf-8").split("\n")  # 以換行結尾時最後一段為空字串
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                if number == len(lines):  # 沒有換行結尾的最後一行：run 寫到一半
                    skipped += 1
                    continue
                raise AggregateError(f"records 損壞：{path.name}:{number}（{error}）") from error
            dataset = str(item.get("dataset", "")).strip().lower()
            model = _canonical_model(item.get("model"))
            condition = _canonical_condition(item.get("condition"))
            if dataset not in DATASETS or model is None or condition is None:
                unknown.add(f"{path.name}: {item.get('dataset')}/{item.get('model')}/{item.get('condition')}")
                continue
            item.update(dataset=dataset, model=model, condition=condition)
            latest[(dataset, model, condition, str(item.get("question_id")))] = item
    if unknown:
        raise AggregateError("records 含無法辨識的 dataset/model/condition：" + "; ".join(sorted(unknown)))
    return list(latest.values()), skipped


# ---------------------------------------------------------------- summary


def _number(value: object) -> float | None:
    if value is None or isinstance(value, str):
        return None
    return float(value)


def _llm_tokens(item: Mapping, key: str) -> float:
    return sum(float((s.get("llm") or {}).get(key) or 0) for s in item.get("steps") or [])


def _laya_latencies(item: Mapping) -> list[float]:
    return [float(s["laya"].get("latency_ms") or 0) for s in item.get("steps") or [] if s.get("laya")]


def _group_metrics(items: list[Mapping]) -> dict[str, float | None]:
    n = len(items)
    latency = pd.Series([v for v in (_number(i.get("latency_ms")) for i in items) if v is not None], dtype=float)
    vram = [v for v in (_number(i.get("vram_peak_mb")) for i in items) if v is not None]
    laya = [ms for i in items for ms in _laya_latencies(i)]
    return {
        "em": 100 * sum(float(i.get("em") or 0) for i in items) / n,
        "f1": 100 * sum(float(i.get("f1") or 0) for i in items) / n,
        "latency_p50_ms": latency.quantile(0.50) if len(latency) else None,
        "latency_p95_ms": latency.quantile(0.95) if len(latency) else None,
        "latency_mean_ms": latency.mean() if len(latency) else None,
        "vram_peak_mb": max(vram) if vram else None,
        "prompt_tokens_mean": sum(_llm_tokens(i, "prompt_tokens") for i in items) / n,
        "output_tokens_mean": sum(_llm_tokens(i, "output_tokens") for i in items) / n,
        "hops_mean": sum(float(i.get("hops") or 0) for i in items) / n,
        "laya_calls_mean": sum(len(_laya_latencies(i)) for i in items) / n,
        "laya_latency_ms_mean": sum(laya) / len(laya) if laya else None,
        "invalid_tool_rate": sum(i.get("status") == "invalid_tool" for i in items) / n,
        "timeout_rate": sum(i.get("status") == "timeout" for i in items) / n,
    }


def summarize(records: Iterable[Mapping]) -> list[dict[str, str]]:
    """每組一列、固定 60 列順序；值為已格式化字串（缺值為空字串）。"""
    groups: dict[tuple[str, str, str], list[Mapping]] = {}
    for item in records:
        groups.setdefault((item["dataset"], item["model"], item["condition"]), []).append(item)
    rows = []
    for dataset in DATASETS:
        for model in MODELS:
            for condition in CONDITIONS:
                items = groups.get((dataset, model, condition), [])
                row = {"dataset": dataset, "model": model, "condition": condition, "n": str(len(items))}
                metrics = _group_metrics(items) if items else {}
                for column, spec in VALUE_FORMATS.items():
                    value = metrics.get(column)
                    row[column] = "" if value is None else format(value, spec)
                rows.append(row)
    return rows


def write_summary_csv(rows: list[dict[str, str]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


# ---------------------------------------------------------------- LaTeX 表格
# 契約（paper/main.tex 的 \tableorplaceholder）：只輸出 bare tabular（booktabs 規則線），
# 不含 table float、\caption、\label、\centering。每個資料列以行尾註解標示對應的 summary 組別。

DATASET_LABELS = {"hotpotqa": "HotpotQA", "2wiki": "2WikiMultihopQA", "musique": "MuSiQue"}
DATASET_SHORT_LABELS = {"hotpotqa": "HotpotQA", "2wiki": "2Wiki", "musique": "MuSiQue"}
CONDITION_LABELS = {"recall_llm": "Recall+LLM", "rag_llm": "RAG+LLM", "recall_laya": "Recall+Laya", "rag_laya": "RAG+Laya"}
MODEL_SIZES_B = {"qwen3.5:27b": 27.0, "qwen3.5:9b": 9.0, "qwen3.5:4b": 4.0, "qwen3.5:2b": 2.0, "qwen3.5:0.8b": 0.8}
MISSING_CELL = "---"  # 排版為「—」：缺組或不適用
# 單欄（\columnwidth = 252pt）放得下的欄位；latency_mean_ms 只留在 summary.csv
EFFICIENCY_COLUMNS = (
    "latency_p50_ms", "latency_p95_ms", "vram_peak_mb",
    "prompt_tokens_mean", "output_tokens_mean", "hops_mean", "laya_latency_ms_mean",
)
COLUMN_GAP = "4pt"  # 欄距（預設 2\tabcolsep = 12pt 會超出單欄寬）


def _colspec(aligns: str) -> str:
    return "@{}" + f"@{{\\hspace{{{COLUMN_GAP}}}}}".join(aligns) + "@{}"


def _model_label(model: str) -> str:
    return model.split(":", 1)[1].upper()


def _cell(row: Mapping[str, str], column: str) -> str:
    return row[column] or MISSING_CELL


def _tex_row(cells: Iterable[str], trace: str) -> str:
    return " & ".join(cells) + f" \\\\ % {trace}"


def _index(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    return {(r["dataset"], r["model"], r["condition"]): r for r in rows}


def render_main_table(rows: list[dict[str, str]]) -> str:
    """EM／F1（%）：列 = 模型 × 條件，欄 = 資料集 × {EM, F1}。"""
    index = _index(rows)
    lines = [
        f"\\begin{{tabular}}{{{_colspec('ll' + 'r' * 2 * len(DATASETS))}}}",
        "\\toprule",
        " & & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{DATASET_SHORT_LABELS[d]}}}" for d in DATASETS) + " \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(l){7-8}",
        "Model & Condition" + " & EM & F1" * len(DATASETS) + " \\\\",
    ]
    for model in MODELS:
        lines.append("\\midrule")
        for position, condition in enumerate(CONDITIONS):
            cells = [_model_label(model) if position == 0 else "", CONDITION_LABELS[condition]]
            for dataset in DATASETS:
                row = index[(dataset, model, condition)]
                cells += [_cell(row, "em"), _cell(row, "f1")]
            lines.append(_tex_row(cells, f"model={model} condition={condition}"))
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def render_efficiency_table(rows: list[dict[str, str]], dataset: str) -> str:
    """單一資料集：p50／p95 延遲、峰值 VRAM、平均 token、跳數、單次 Laya 延遲；模型為分段標題列。"""
    index = _index(rows)
    width = 1 + len(EFFICIENCY_COLUMNS)
    lines = [
        f"\\begin{{tabular}}{{{_colspec('l' + 'r' * len(EFFICIENCY_COLUMNS))}}}",
        "\\toprule",
        " & \\multicolumn{2}{c}{Latency (ms)} & VRAM & \\multicolumn{2}{c}{Tokens} & & Laya \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){5-6}",
        "Condition & p50 & p95 & (MB) & In & Out & Hops & (ms) \\\\",
    ]
    for model in MODELS:
        lines += ["\\midrule", f"\\multicolumn{{{width}}}{{@{{}}l}}{{\\textit{{Qwen3.5-{_model_label(model)}}}}} \\\\"]
        for condition in CONDITIONS:
            row = index[(dataset, model, condition)]
            cells = [CONDITION_LABELS[condition], *(_cell(row, column) for column in EFFICIENCY_COLUMNS)]
            lines.append(_tex_row(cells, f"dataset={dataset} model={model} condition={condition}"))
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def write_tables(rows: list[dict[str, str]], tables_dir: Path) -> list[Path]:
    """main_results.tex 與每個資料集一張 efficiency_<dataset>.tex。"""
    tables_dir.mkdir(parents=True, exist_ok=True)
    contents = {"main_results": render_main_table(rows)}
    contents.update({f"efficiency_{d}": render_efficiency_table(rows, d) for d in DATASETS})
    written = []
    for stem, text in contents.items():
        path = tables_dir / f"{stem}.tex"
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written


# ---------------------------------------------------------------- 圖（matplotlib PDF，Agg 後端）
# 四條件使用固定色序（參考色盤前四槽，已過色盲驗證）並加上 marker／線型作次要編碼，黑白列印仍可辨識。

CONDITION_STYLES = {
    "recall_llm": {"color": "#2a78d6", "marker": "o", "linestyle": "-"},
    "rag_llm": {"color": "#eb6834", "marker": "s", "linestyle": "--"},
    "recall_laya": {"color": "#1baf7a", "marker": "^", "linestyle": "-."},
    "rag_laya": {"color": "#eda100", "marker": "D", "linestyle": ":"},
}
FIGURE_RC = {
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e3e2de", "grid.linewidth": 0.5, "pdf.fonttype": 42,
}
COLUMN_WIDTH_IN = 3.5
DEFAULT_HOPS = (1, 2, 3, 4)
SIZE_LABEL = "Model size (billion parameters, log scale)"


def _float(value: str) -> float:
    return float(value) if value else float("nan")


def _size_axis(ax) -> None:
    sizes = [MODEL_SIZES_B[m] for m in MODELS]
    ax.set_xscale("log")
    ax.set_xticks(sizes, [_model_label(m) for m in MODELS])
    ax.minorticks_off()
    ax.set_xlim(min(sizes) / 1.35, max(sizes) * 1.35)


def _plot_by_size(ax, index, dataset: str, column: str) -> None:
    for condition in CONDITIONS:
        values = [_float(index[(dataset, model, condition)][column]) for model in MODELS]
        ax.plot([MODEL_SIZES_B[m] for m in MODELS], values, label=CONDITION_LABELS[condition],
                linewidth=1.5, markersize=4, **CONDITION_STYLES[condition])


def hops_distribution(records: Iterable[Mapping]) -> dict[str, dict[int, float]]:
    """每個條件（跨資料集與模型合併）各跳數的題數占比（%）。"""
    counts: dict[str, dict[int, int]] = {c: {} for c in CONDITIONS}
    for item in records:
        if item.get("hops") is not None:
            per_condition = counts[item["condition"]]
            per_condition[int(item["hops"])] = per_condition.get(int(item["hops"]), 0) + 1
    hops = sorted(set(DEFAULT_HOPS).union(*(c.keys() for c in counts.values())))
    shares = {}
    for condition, per_condition in counts.items():
        total = sum(per_condition.values())
        shares[condition] = {h: 100 * per_condition.get(h, 0) / total if total else 0.0 for h in hops}
    return shares


def build_figures(rows: list[dict[str, str]], records: list[Mapping]) -> dict[str, Figure]:
    """回傳 {檔名 stem: Figure}；缺組的點以斷線呈現，不會崩潰。"""
    index = _index(rows)
    figures: dict[str, Figure] = {}
    with matplotlib.rc_context(FIGURE_RC):
        for dataset in DATASETS:
            figure = Figure(figsize=(COLUMN_WIDTH_IN, 2.5), layout="constrained")
            ax = figure.add_subplot()
            _plot_by_size(ax, index, dataset, "em")
            _size_axis(ax)
            ax.set_ylim(0, 100)
            ax.set(title=DATASET_LABELS[dataset], xlabel=SIZE_LABEL, ylabel="Exact match (%)")
            ax.legend(title="Condition", ncols=2, frameon=False)
            figures[f"em_vs_size_{dataset}"] = figure

        figure = Figure(figsize=(COLUMN_WIDTH_IN, 5.0), layout="constrained")
        axes = figure.subplots(len(DATASETS), 1, sharex=True)
        for ax, dataset in zip(axes, DATASETS):
            _plot_by_size(ax, index, dataset, "latency_p50_ms")
            _size_axis(ax)
            if any(line.get_ydata().size and (line.get_ydata() > 0).any() for line in ax.get_lines()):
                ax.set_yscale("log")
            ax.set(title=DATASET_LABELS[dataset], ylabel="p50 latency (ms)")
        axes[-1].set_xlabel(SIZE_LABEL)
        # 兩欄：四欄圖例寬於欄寬，會被 PDF 頁框左右裁切（R8 review B-1）
        figure.legend(*axes[0].get_legend_handles_labels(), title="Condition", loc="outside upper center",
                      ncols=2, frameon=False)
        figures["latency_vs_size"] = figure

        shares = hops_distribution(records)
        hops = list(next(iter(shares.values())))
        width = 0.8 / len(CONDITIONS)
        figure = Figure(figsize=(COLUMN_WIDTH_IN, 2.5), layout="constrained")
        ax = figure.add_subplot()
        for offset, condition in enumerate(CONDITIONS):
            positions = [h + (offset - (len(CONDITIONS) - 1) / 2) * width for h in hops]
            ax.bar(positions, [shares[condition][h] for h in hops], width=width, label=CONDITION_LABELS[condition],
                   color=CONDITION_STYLES[condition]["color"], edgecolor="white", linewidth=0.8)
        ax.set_xticks(hops)
        ax.set_ylim(0, 100)
        ax.grid(axis="x", visible=False)
        ax.set(xlabel="Hops taken (count)", ylabel="Share of questions (%)")
        ax.legend(title="Condition", ncols=2, frameon=False)
        figures["hops_distribution"] = figure
    return figures


def write_figures(rows: list[dict[str, str]], records: list[Mapping], figures_dir: Path) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for stem, figure in build_figures(rows, records).items():
        path = figures_dir / f"{stem}.pdf"
        figure.savefig(path, format="pdf", metadata={"CreationDate": None})
        written.append(path)
    return written


# ---------------------------------------------------------------- 探針表
# probe_results.json（Ticket 09）：每模型 valid_tool_call_rate（0–1）、n、平均延遲。
# 接受 {"models": {tag: {...}}}、{"models": [{"model": tag, ...}]}、{tag: {...}} 或 [{"model": tag, ...}]。

PROBE_LATENCY_KEYS = ("mean_latency_ms", "avg_latency_ms", "latency_ms_mean", "latency_mean_ms")


def load_probe_results(path: Path) -> dict[str, Mapping]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise AggregateError(f"找不到 probe_results.json：{path}") from error
    except json.JSONDecodeError as error:
        raise AggregateError(f"probe_results.json 不是合法 JSON：{path}（{error}）") from error
    entries = data.get("models", data) if isinstance(data, dict) else data
    if isinstance(entries, dict):
        entries = [{"model": key, **value} for key, value in entries.items() if isinstance(value, dict)]
    probes = {}
    for entry in entries or []:
        model = _canonical_model(entry.get("model")) if isinstance(entry, dict) else None
        if model and "valid_tool_call_rate" in entry:
            probes[model] = entry
    if not probes:
        raise AggregateError(f"probe_results.json 沒有任何含 valid_tool_call_rate 的模型：{path}")
    return probes


def render_probe_table(probes: Mapping[str, Mapping]) -> str:
    """每模型一列：探針題數、合法工具呼叫率（0–1）、平均延遲（ms）。"""
    lines = [
        "\\begin{tabular}{@{}lrrr@{}}",
        "\\toprule",
        "Model & $n$ & Valid rate & Latency (ms) \\\\",
        "\\midrule",
    ]
    for model in MODELS:
        probe = probes.get(model, {})
        latency = next((probe[k] for k in PROBE_LATENCY_KEYS if probe.get(k) is not None), None)
        cells = [
            _model_label(model),
            MISSING_CELL if probe.get("n") is None else str(int(probe["n"])),
            MISSING_CELL if probe.get("valid_tool_call_rate") is None else f"{float(probe['valid_tool_call_rate']):.2f}",
            MISSING_CELL if latency is None else f"{float(latency):.0f}",
        ]
        lines.append(_tex_row(cells, f"model={model}"))
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def write_probe_table(probe_path: Path, tables_dir: Path) -> Path:
    probes = load_probe_results(probe_path)
    tables_dir.mkdir(parents=True, exist_ok=True)
    path = tables_dir / "probe.tex"
    path.write_text(render_probe_table(probes), encoding="utf-8")
    return path


# ---------------------------------------------------------------- 入口


def aggregate_run(run_dir: Path, paper_dir: Path, *, probe_only: bool = False) -> AggregateResult:
    """records → summary.csv、主表、效率表、圖；probe_results.json 存在時加上探針表，
    不存在時移除輸出目錄中既有的 probe.tex（避免殘留其他 run 的探針表）。

    probe_only=True 時只把 probe_results.json 轉成 `tables/probe.tex`（檔案不存在則報錯）。
    """
    run_dir, paper_dir = Path(run_dir), Path(paper_dir)
    probe_path = run_dir / "probe_results.json"
    if probe_only:
        return AggregateResult(tables=[write_probe_table(probe_path, paper_dir / "tables")])

    records_dir = run_dir / "records"
    if not records_dir.is_dir():
        raise AggregateError(f"找不到 records 目錄：{records_dir}")
    records, skipped = load_records(records_dir)
    rows = summarize(records)
    result = AggregateResult(record_count=len(records), skipped_lines=skipped)
    result.summary_path = write_summary_csv(rows, run_dir / "summary.csv")
    result.tables = write_tables(rows, paper_dir / "tables")
    stale_probe = paper_dir / "tables" / "probe.tex"
    if probe_path.is_file():
        result.tables.append(write_probe_table(probe_path, paper_dir / "tables"))
    elif stale_probe.exists():  # 前一個 run 留下的探針表：不讓論文混用不同 run 的數字
        stale_probe.unlink()
        result.removed.append(stale_probe)
    result.figures = write_figures(rows, records, paper_dir / "figures")
    return result
