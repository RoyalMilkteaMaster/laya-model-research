"""產生自含式 `reports/dashboard.html`：資料內嵌 JSON、表格與內嵌 SVG 曲線、每 60 秒自動重新整理。

不引用任何本機檔案、CDN 或外部資源，瀏覽器以 file:// 直接開啟即可顯示（不需 JavaScript）。
與 PROGRESS.md 共用 progress_report_writer 的快照，數字一致。
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

from multihop_benchmark.reporting.progress_report_writer import (
    ALL_HAVE_RESULTS,
    APPROXIMATE_ORDER,
    NO_DATA,
    ProgressSnapshot,
    collect_progress,
    combo_key,
    current_combo_text,
    eta_text,
    format_latency,
    format_ratio,
    format_timestamp,
    invalid_lines_text,
    overall_text,
    record_time,
    score,
    write_report_file,
)

REFRESH_SECONDS = 60
# 條件的固定配色（dataviz 參考色盤前四個 slot；淺色／深色各一組），顏色跟著條件走，不隨排序改變
CONDITION_COLORS = {
    "recall_llm": ("#2a78d6", "#3987e5"),
    "rag_llm": ("#eb6834", "#d95926"),
    "recall_laya": ("#1baf7a", "#199e70"),
    "rag_laya": ("#eda100", "#c98500"),
}
OTHER_COLOR = ("#898781", "#898781")


def build_dashboard_data(snapshot: ProgressSnapshot) -> dict[str, Any]:
    latest_ids = {id(record) for combo in snapshot.combos for record in combo.records}
    cumulative: dict[str, dict[str, list[float]]] = {}
    latencies: dict[str, dict[str, list[float]]] = {}
    totals: dict[tuple[str, str], list[float]] = {}
    for record in snapshot.ordered_records:  # 由舊到新，只取每題最後一筆
        if id(record) not in latest_ids:
            continue
        _, model, condition = combo_key(record)
        hits = totals.setdefault((model, condition), [0.0, 0.0])
        hits[0] += score(record, "em")
        hits[1] += 1
        cumulative.setdefault(model, {}).setdefault(condition, []).append(hits[0] / hits[1])
        if isinstance(record.get("latency_ms"), (int, float)):
            latencies.setdefault(model, {}).setdefault(condition, []).append(float(record["latency_ms"]))

    return {
        "run_id": snapshot.run_id,
        "generated_at": snapshot.generated_at.isoformat(timespec="seconds"),
        "has_data": snapshot.has_data,
        "current_combo": snapshot.current_combo.combo if snapshot.current_combo else None,
        "order_is_approximate": snapshot.order_is_approximate,
        "invalid_lines": [list(item) for item in snapshot.invalid_lines],
        "overall": {
            "done": snapshot.done,
            "attempted": snapshot.attempted,
            "non_done": snapshot.non_done,
            "non_done_by_status": snapshot.non_done_by_status,
            "target_total": snapshot.target_total,
            "all_have_results": snapshot.all_have_results,
            "mean_done_latency_ms": snapshot.mean_done_latency_ms,
            "eta_seconds": snapshot.eta_seconds,
            "finish_at": snapshot.finish_at.isoformat(timespec="seconds") if snapshot.finish_at else None,
        },
        "combos": [
            {
                "combo": combo.combo, "dataset": combo.dataset, "model": combo.model, "condition": combo.condition,
                "done": combo.done, "attempted": combo.attempted, "target": combo.target, "em": combo.em, "f1": combo.f1,
                "latency_p50_ms": combo.latency_p50_ms, "non_done": combo.non_done, "state": combo.state,
            }
            for combo in snapshot.combos
        ],
        "cumulative_em": cumulative,
        "latency_ms": latencies,
        "recent_errors": [
            {
                "time": record_time(record), "combo": "__".join(combo_key(record)),
                "question_id": record.get("question_id"), "status": record.get("status"), "error": record.get("error"),
            }
            for record in snapshot.recent_errors
        ],
        "recent_questions": [
            {
                "combo": "__".join(combo_key(record)), "question_id": record.get("question_id"),
                "status": record.get("status"), "question": record.get("question"),
                "prediction": record.get("prediction"), "gold": record.get("gold"),
                "correct": score(record, "em") >= 1, "f1": score(record, "f1"), "hops": record.get("hops"),
            }
            for record in snapshot.recent_questions
        ],
    }


def render_dashboard_html(snapshot: ProgressSnapshot) -> str:
    data = build_dashboard_data(snapshot)
    embedded = json.dumps(data, ensure_ascii=False, indent=1).replace("<", "\\u003c")
    if snapshot.has_data:
        body = "\n".join(
            [
                _overview(snapshot),
                _section("各模型×條件累積 EM（隨完成題數）", _cumulative_em_chart(data["cumulative_em"], snapshot)),
                _section("延遲分布（每題端到端，秒）", _latency_chart(data["latency_ms"], snapshot)),
                _section(f"各組即時指標（{len(snapshot.combos)} 組）", _combo_table(snapshot)),
                _section(
                    "最近 10 筆錯誤（status ≠ done，新到舊）",
                    _invalid_warning(snapshot) + _order_note(snapshot) + _error_table(data["recent_errors"]),
                ),
                _section("最近 3 題摘要（新到舊）", _order_note(snapshot) + _recent_table(data["recent_questions"])),
            ]
        )
    else:
        body = f'<p class="no-data"><strong>{NO_DATA}</strong>：records 目錄不存在或沒有任何 record（{escape(str(snapshot.records_dir))}）。</p>'
    return f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="{REFRESH_SECONDS}">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>實驗進度：{escape(snapshot.run_id)}</title>
<style>{_STYLE}</style>
</head>
<body>
<main class="viz-root">
<h1>實驗進度：{escape(snapshot.run_id)}</h1>
<p class="meta">產生時間：{escape(format_timestamp(snapshot.generated_at))}　·　每 {REFRESH_SECONDS} 秒自動重新整理</p>
{body}
</main>
<script id="dashboard-data" type="application/json">{embedded}</script>
</body>
</html>
"""


def write_dashboard(
    data_root: Path, run_id: str, *, now: datetime | None = None, snapshot: ProgressSnapshot | None = None
) -> Path:
    snapshot = snapshot or collect_progress(data_root, run_id, now=now)
    return write_report_file(Path(data_root) / "reports" / "dashboard.html", render_dashboard_html(snapshot))


# ---- 區塊 ----
def _section(title: str, content: str) -> str:
    return f"<section><h2>{escape(title)}</h2>\n{content}\n</section>"


def _overview(snapshot: ProgressSnapshot) -> str:
    if snapshot.all_have_results:
        finish = f"不適用（{ALL_HAVE_RESULTS}）"
    else:
        finish = snapshot.finish_at.isoformat(timespec="seconds") if snapshot.finish_at else "—"
    tiles = [
        ("目前組別", current_combo_text(snapshot)),
        ("整體完成率", overall_text(snapshot)),
        ("已完成題平均延遲", format_latency(snapshot.mean_done_latency_ms)),
        ("ETA", eta_text(snapshot, "無法估計")),
        ("預計完成", finish),
    ]
    items = "".join(f'<div class="tile"><div class="label">{escape(k)}</div><div class="value">{escape(v)}</div></div>' for k, v in tiles)
    return f'<section class="tiles">{items}</section>'


def _combo_table(snapshot: ProgressSnapshot) -> str:
    rows = "".join(
        f'<tr class="combo-row state-{"idle" if combo.attempted == 0 else "active"}">'
        f"<td>{escape(combo.dataset)}</td><td>{escape(combo.model)}</td><td>{_swatch(combo.condition)}{escape(combo.condition)}</td>"
        f'<td class="num">{combo.done} / {combo.target}</td><td class="num">{combo.attempted}</td>'
        f'<td class="num">{format_ratio(combo.em)}</td>'
        f'<td class="num">{format_ratio(combo.f1)}</td><td class="num">{format_latency(combo.latency_p50_ms)}</td>'
        f'<td class="num">{combo.non_done}</td><td>{escape(combo.state)}</td></tr>'
        for combo in snapshot.combos
    )
    head = "".join(f"<th>{h}</th>" for h in ("dataset", "model", "condition", "完成題數（done）", "已嘗試", "EM", "F1", "延遲 p50", "非 done", "狀態"))
    note = '<p class="muted">完成題數＝status done；EM／F1 以每題最新一筆計、非 done 計 0；延遲 p50 含所有狀態。</p>'
    return f'{note}<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'


def _invalid_warning(snapshot: ProgressSnapshot) -> str:
    if not snapshot.invalid_lines:
        return ""
    return f'<p class="warning">⚠ {escape(invalid_lines_text(snapshot.invalid_lines))}</p>'


def _order_note(snapshot: ProgressSnapshot) -> str:
    return f'<p class="muted"><em>{escape(APPROXIMATE_ORDER)}</em></p>' if snapshot.order_is_approximate else ""


def _error_table(errors: list[dict[str, Any]]) -> str:
    if not errors:
        return "<p>無</p>"
    rows = "".join(
        "<tr>" + "".join(f"<td>{_text(error[key])}</td>" for key in ("time", "combo", "question_id", "status", "error")) + "</tr>"
        for error in errors
    )
    head = "".join(f"<th>{h}</th>" for h in ("時間", "combo", "question_id", "status", "error"))
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'


def _recent_table(recent: list[dict[str, Any]]) -> str:
    rows = "".join(
        f"<tr><td>{_text(item['combo'])}<br><span class=\"muted\">{_text(item['question_id'])}（{_text(item['status'])}）</span></td>"
        f"<td>{_text(item['question'] or '（record 未含題目文字）')}</td><td>{_text(item['prediction'])}</td>"
        f"<td>{_text(item['gold'])}</td><td>{'✔ 對' if item['correct'] else '✘ 錯'}（F1 {item['f1']:.3f}）</td>"
        f'<td class="num">{_text(item["hops"])}</td></tr>'
        for item in recent
    )
    head = "".join(f"<th>{h}</th>" for h in ("combo / 題號", "問題", "預測", "gold", "對錯", "跳數"))
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'


# ---- 圖 ----
def _legend(conditions: list[str]) -> str:
    return '<div class="legend">' + "".join(
        f'<span class="legend-item">{_swatch(c)}{escape(c)}</span>' for c in conditions
    ) + "</div>"


def _cumulative_em_chart(cumulative: dict[str, dict[str, list[float]]], snapshot: ProgressSnapshot) -> str:
    models = list(dict.fromkeys(combo.model for combo in snapshot.combos))
    conditions = _conditions(snapshot)
    width, height, left, right, top, bottom = 300, 190, 34, 10, 22, 30
    panels = []
    for model in models:
        series = cumulative.get(model, {})
        max_n = max((len(values) for values in series.values()), default=0)
        plot_w, plot_h = width - left - right, height - top - bottom
        x = lambda i: left + (plot_w * (i / (max_n - 1) if max_n > 1 else 0.5))  # noqa: E731
        y = lambda v: top + plot_h * (1 - v)  # noqa: E731
        parts = [f'<text class="panel-title" x="{left}" y="14">{escape(model)}</text>']
        for tick in (0, 0.25, 0.5, 0.75, 1):
            parts.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y(tick):.1f}" y2="{y(tick):.1f}"/>')
            parts.append(f'<text class="tick" x="{left - 4}" y="{y(tick) + 3:.1f}" text-anchor="end">{tick:g}</text>')
        parts.append(f'<line class="axis" x1="{left}" x2="{width - right}" y1="{y(0):.1f}" y2="{y(0):.1f}"/>')
        if max_n:
            parts.append(f'<text class="tick" x="{left}" y="{height - 16}" text-anchor="start">1</text>')
            parts.append(f'<text class="tick" x="{width - right}" y="{height - 16}" text-anchor="end">{max_n}</text>')
            parts.append(f'<text class="tick" x="{left + plot_w / 2:.1f}" y="{height - 4}" text-anchor="middle">完成題數</text>')
            for condition in conditions:
                values = series.get(condition)
                if not values:
                    continue
                points = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values))
                last_x, last_y = x(len(values) - 1), y(values[-1])
                tip = f"{model} · {condition}：{len(values)} 題，累積 EM {values[-1]:.3f}"
                parts.append(
                    f'<g class="series {_color_class(condition)}"><title>{escape(tip)}</title>'
                    f'<polyline points="{points}"/><circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="4"/></g>'
                )
        else:
            parts.append(f'<text class="empty" x="{left + plot_w / 2:.1f}" y="{top + plot_h / 2:.1f}" text-anchor="middle">{NO_DATA}</text>')
        panels.append(f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(model)} 累積 EM">{"".join(parts)}</svg>')
    return f'{_legend(conditions)}<div id="chart-cumulative-em" class="panels">{"".join(panels)}</div>'


def _latency_chart(latencies: dict[str, dict[str, list[float]]], snapshot: ProgressSnapshot) -> str:
    models = list(dict.fromkeys(combo.model for combo in snapshot.combos))
    conditions = _conditions(snapshot)
    rows = [
        (model, condition, sorted(v / 1000 for v in latencies[model][condition]))
        for model in models for condition in conditions if latencies.get(model, {}).get(condition)
    ]
    if not rows:
        return f'<p id="chart-latency" class="empty">{NO_DATA}</p>'
    x_max = max(values[-1] for _, _, values in rows) or 1.0
    width, row_h, left, right, top = 1100, 22, 170, 20, 10
    height = top + row_h * len(rows) + 28
    plot_w = width - left - right
    x = lambda v: left + plot_w * v / x_max  # noqa: E731
    parts = []
    for step in range(5):
        tick = x_max * step / 4
        parts.append(f'<line class="grid" x1="{x(tick):.1f}" x2="{x(tick):.1f}" y1="{top}" y2="{height - 26}"/>')
        parts.append(f'<text class="tick" x="{x(tick):.1f}" y="{height - 12}" text-anchor="middle">{tick:.1f}</text>')
    parts.append(f'<text class="tick" x="{left + plot_w / 2:.1f}" y="{height - 1}" text-anchor="middle">秒（鬚：p5–p95，盒：p25–p75，線：p50）</text>')
    for index, (model, condition, values) in enumerate(rows):
        mid = top + row_h * index + row_h / 2
        p5, p25, p50, p75, p95 = (_quantile(values, q) for q in (0.05, 0.25, 0.5, 0.75, 0.95))
        tip = (
            f"{model} · {condition}：n={len(values)}，p50 {p50:.2f} s，p25–p75 {p25:.2f}–{p75:.2f} s，"
            f"p5–p95 {p5:.2f}–{p95:.2f} s，最大 {values[-1]:.2f} s"
        )
        parts.append(
            f'<g class="box {_color_class(condition)}"><title>{escape(tip)}</title>'
            f'<rect class="hit" x="{left}" y="{mid - row_h / 2:.1f}" width="{plot_w}" height="{row_h}"/>'
            f'<text class="row-label" x="{left - 6}" y="{mid + 4:.1f}" text-anchor="end">{escape(model)} · {escape(condition)}</text>'
            f'<line class="whisker" x1="{x(p5):.1f}" x2="{x(p95):.1f}" y1="{mid:.1f}" y2="{mid:.1f}"/>'
            f'<rect x="{x(p25):.1f}" y="{mid - 6:.1f}" width="{max(x(p75) - x(p25), 2):.1f}" height="12" rx="2"/>'
            f'<line class="median" x1="{x(p50):.1f}" x2="{x(p50):.1f}" y1="{mid - 8:.1f}" y2="{mid + 8:.1f}"/></g>'
        )
    return f'{_legend(conditions)}<svg id="chart-latency" viewBox="0 0 {width} {height}" role="img" aria-label="延遲分布">{"".join(parts)}</svg>'


def _quantile(sorted_values: list[float], q: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    return statistics.quantiles(sorted_values, n=100, method="inclusive")[round(q * 100) - 1]


# ---- 小工具 ----
def _conditions(snapshot: ProgressSnapshot) -> list[str]:
    return list(dict.fromkeys(combo.condition for combo in snapshot.combos))  # 網格順序，網格外條件排在後面


def _color_class(condition: str) -> str:
    return f"c-{condition}" if condition in CONDITION_COLORS else "c-other"


def _swatch(condition: str) -> str:
    return f'<span class="swatch {_color_class(condition)}"></span>'


def _text(value: Any) -> str:
    return escape("—" if value is None or value == "" else str(value))


def _color_rules(mode: int) -> str:
    return "".join(
        f".viz-root .c-{name}{{--c:{colors[mode]}}}" for name, colors in [*CONDITION_COLORS.items(), ("other", OTHER_COLOR)]
    )


_STYLE = (
    """
.viz-root{color-scheme:light;--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink-2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);
font:14px/1.5 system-ui,-apple-system,"Segoe UI","Microsoft JhengHei",sans-serif;color:var(--ink);
background:var(--page);max-width:1280px;margin:0 auto;padding:16px 20px 40px}
body{margin:0;background:#f9f9f7}
h1{font-size:20px;margin:4px 0}h2{font-size:16px;margin:0 0 8px}
.meta,.muted{color:var(--ink-2)}
section{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px 14px;margin:12px 0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;background:none;border:0;padding:0}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:10px 12px}
.tile .label{color:var(--ink-2);font-size:12px}.tile .value{font-size:16px;font-weight:600;font-variant-numeric:tabular-nums;word-break:break-all}
.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:8px}
svg{width:100%;height:auto;display:block}
svg text{fill:var(--ink-2);font-size:10px}svg .panel-title{fill:var(--ink);font-size:12px;font-weight:600}
svg .empty{fill:var(--muted);font-size:12px}
.grid{stroke:var(--grid);stroke-width:1}.axis{stroke:var(--axis);stroke-width:1}
.series polyline{fill:none;stroke:var(--c);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.series circle{fill:var(--c);stroke:var(--surface);stroke-width:2}
.series:hover polyline{stroke-width:3}
.box rect{fill:var(--c);stroke:var(--surface);stroke-width:1}.box .hit{fill:transparent;stroke:none}
.box:hover .hit{fill:var(--grid)}
.box .whisker{stroke:var(--c);stroke-width:2}.box .median{stroke:var(--ink);stroke-width:2}
.box .row-label{fill:var(--ink-2);font-size:11px}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin:0 0 8px;color:var(--ink-2)}
.swatch{display:inline-block;width:10px;height:10px;border-radius:2px;background:var(--c);margin-right:6px;vertical-align:baseline}
.table-wrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:4px 8px;border-bottom:1px solid var(--grid);vertical-align:top}
th{color:var(--ink-2);font-weight:600}td.num{text-align:right}
tr.state-idle td{color:var(--muted)}
.no-data{font-size:16px}
.warning{border-left:4px solid #fab219;padding:4px 10px;margin:0 0 8px;color:var(--ink)}
"""
    + _color_rules(0)
    + """
@media (prefers-color-scheme:dark){.viz-root{color-scheme:dark;--surface:#1a1a19;--page:#0d0d0d;--ink:#ffffff;
--ink-2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10)}body{background:#0d0d0d}"""
    + _color_rules(1)
    + "}\n"
)
