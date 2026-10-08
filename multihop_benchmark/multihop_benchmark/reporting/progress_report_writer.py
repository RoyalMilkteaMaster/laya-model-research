"""讀取 run 的 records，產生進度快照並寫出 `reports/PROGRESS.md`。

report 只讀 Data Root 的 `runs/<run_id>/records/*.jsonl`、`runs/<run_id>/manifest.json`（可不存在）與
`datasets/<dataset>/sample_200.jsonl`（僅為「最近 3 題」補題目文字，找不到就略過），只寫 `reports/`。
records 依 Spec schema 直接以 JSONL 讀取（不依賴 record_store），檔名不解析，組別取自 record 欄位。

- 完成語意與續跑一致：完成題數、完成率、組別狀態、目前組別、剩餘題數、ETA 與「已完成題平均延遲」
  一律只計 `status == "done"`；EM／F1 以每題最新一筆計（非 done 計 0，Spec：無效工具呼叫視為答錯），
  延遲 p50 含所有狀態（與 results_aggregator 一致），另列「已嘗試」與「非 done」計數。
- 終態：每組已嘗試題數都達目標且沒有進行中組別時，視為「所有題目已有結果」；完成率仍以 done 計，
  但 ETA 顯示 0 並說明續跑只會重做非 done 題，不再輸出預計完成時間。
- 新舊順序：依選填欄位 `finished_at`（UTC ISO 8601）跨檔排序；缺此欄位的 record 排在有欄位者之前，
  彼此以檔案 mtime 再行號推定（近似），此時「最近錯誤」「最近 3 題」標示順序為近似。
- 同一組同一題有多筆（續跑重做非 done 題）時以最新一筆為準；錯誤清單保留全部歷史。
- 檔尾沒有換行的半行視為 run 正在 append，靜默略過；其他無法解析的行（與 record_store.completed_ids
  一致）記在 `invalid_lines`，報告於「最近錯誤」上方顯示警告。
"""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

DATASETS = ("hotpotqa", "2wiki", "musique")
MODELS = ("qwen3.5:27b", "qwen3.5:9b", "qwen3.5:4b", "qwen3.5:2b", "qwen3.5:0.8b")
CONDITIONS = ("recall_llm", "rag_llm", "recall_laya", "rag_laya")
QUESTIONS_PER_COMBO = 200
RECENT_ERRORS = 10
RECENT_QUESTIONS = 3
MAX_LISTED_INVALID_LINES = 20
NO_DATA = "尚無資料"
APPROXIMATE_ORDER = "順序為近似（record 無 finished_at）"
ALL_HAVE_RESULTS = "所有題目已有結果"
TERMINAL_ETA = f"0（{ALL_HAVE_RESULTS}；續跑只會重做非 done 題）"
NOT_STARTED, RUNNING, FINISHED, PARTIAL = "未開始", "進行中", "已完成", "未完成"


@dataclass
class ComboStats:
    dataset: str
    model: str
    condition: str
    target: int
    records: list[dict[str, Any]] = field(default_factory=list)  # 每題最新一筆，由舊到新
    state: str = NOT_STARTED

    @property
    def combo(self) -> str:
        return f"{self.dataset}__{self.model}__{self.condition}"

    @property
    def attempted(self) -> int:
        return len(self.records)

    @property
    def done(self) -> int:
        return sum(1 for record in self.records if record.get("status") == "done")

    @property
    def non_done(self) -> int:
        return self.attempted - self.done

    @property
    def em(self) -> float | None:
        return _mean([score(record, "em") for record in self.records])

    @property
    def f1(self) -> float | None:
        return _mean([score(record, "f1") for record in self.records])

    @property
    def latencies_ms(self) -> list[float]:
        return [latency for record in self.records if (latency := _latency(record)) is not None]

    @property
    def done_latencies_ms(self) -> list[float]:
        return [
            latency for record in self.records
            if record.get("status") == "done" and (latency := _latency(record)) is not None
        ]

    @property
    def latency_p50_ms(self) -> float | None:
        latencies = self.latencies_ms
        return statistics.median(latencies) if latencies else None


@dataclass
class ProgressSnapshot:
    run_id: str
    generated_at: datetime
    records_dir: Path
    combos: list[ComboStats]
    current_combo: ComboStats | None
    recent_errors: list[dict[str, Any]]
    recent_questions: list[dict[str, Any]]
    ordered_records: list[dict[str, Any]]  # 全部有效 record，由舊到新（含重做前的舊紀錄）
    order_is_approximate: bool  # 有 record 缺 finished_at，新舊順序由 mtime 推定
    invalid_lines: list[tuple[str, int]]  # 無法解析的行（檔名、行號），不含檔尾寫入中的半行

    @property
    def has_data(self) -> bool:
        return bool(self.ordered_records)

    @property
    def done(self) -> int:
        return sum(min(combo.done, combo.target) for combo in self.combos)

    @property
    def attempted(self) -> int:
        return sum(combo.attempted for combo in self.combos)

    @property
    def non_done(self) -> int:
        return sum(combo.non_done for combo in self.combos)

    @property
    def target_total(self) -> int:
        return sum(combo.target for combo in self.combos)

    @property
    def non_done_by_status(self) -> dict[str, int]:
        """非 done 題依 status 計數（每題最新一筆），多到少。"""
        counts: dict[str, int] = {}
        for combo in self.combos:
            for record in combo.records:
                if record.get("status") != "done":
                    counts[str(record.get("status"))] = counts.get(str(record.get("status")), 0) + 1
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

    @property
    def all_have_results(self) -> bool:
        """終態：每組都已嘗試滿目標題數且沒有進行中組別（續跑只會重做非 done 題）。"""
        return (
            self.has_data and self.current_combo is None
            and all(combo.attempted >= combo.target for combo in self.combos)
        )

    @property
    def mean_done_latency_ms(self) -> float | None:
        return _mean([latency for combo in self.combos for latency in combo.done_latencies_ms])

    @property
    def eta_seconds(self) -> float | None:
        if self.all_have_results:
            return 0.0
        mean_latency = self.mean_done_latency_ms
        if mean_latency is None:
            return None
        return (self.target_total - self.done) * mean_latency / 1000

    @property
    def finish_at(self) -> datetime | None:
        eta = self.eta_seconds
        return None if eta is None or self.all_have_results else self.generated_at + timedelta(seconds=round(eta))


def score(record: dict[str, Any], key: str) -> float:
    if record.get("status") != "done":
        return 0.0
    value = record.get(key)
    return float(value) if isinstance(value, (int, float)) else 0.0


def combo_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (str(record.get("dataset")), str(record.get("model")), str(record.get("condition")))


def record_time(record: dict[str, Any]) -> str:
    return str(record["finished_at"]) if _finished_at(record) is not None else "—"


def collect_progress(data_root: Path, run_id: str, *, now: datetime | None = None) -> ProgressSnapshot:
    generated_at = (now or datetime.now()).astimezone()
    run_dir = Path(data_root) / "runs" / run_id
    records_dir = run_dir / "records"
    manifest = _read_manifest(run_dir / "manifest.json")
    target = _manifest_int(manifest, "questions_per_combo", "limit") or QUESTIONS_PER_COMBO

    grid: dict[tuple[str, str, str], ComboStats] = {}
    for dataset in _manifest_names(manifest, "datasets", DATASETS):
        for model in _manifest_names(manifest, "models", MODELS):
            for condition in _manifest_names(manifest, "conditions", CONDITIONS):
                grid[(dataset, model, condition)] = ComboStats(dataset, model, condition, target)

    ordered, invalid_lines = _read_records(records_dir)
    latest: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for record in ordered:
        per_question = latest.setdefault(combo_key(record), {})
        per_question.pop(str(record.get("question_id")), None)  # 重做的題移到最新位置
        per_question[str(record.get("question_id"))] = record
    for key, per_question in latest.items():
        combo = grid.setdefault(key, ComboStats(*key, target))  # 網格外的組別附加在表尾
        combo.records = list(per_question.values())

    current = grid[combo_key(ordered[-1])] if ordered else None
    if current is not None and current.done >= current.target:
        current = None
    for combo in grid.values():
        if combo.done >= combo.target:
            combo.state = FINISHED
        elif combo is current:
            combo.state = RUNNING
        elif combo.attempted:
            combo.state = PARTIAL

    errors = [record for record in reversed(ordered) if record.get("status") != "done"][:RECENT_ERRORS]
    recent = list(reversed(ordered))[:RECENT_QUESTIONS]
    _attach_question_text(Path(data_root), recent)
    return ProgressSnapshot(
        run_id=run_id,
        generated_at=generated_at,
        records_dir=records_dir,
        combos=list(grid.values()),
        current_combo=current,
        recent_errors=errors,
        recent_questions=recent,
        ordered_records=ordered,
        order_is_approximate=any(_finished_at(record) is None for record in ordered),
        invalid_lines=invalid_lines,
    )


def render_progress_markdown(snapshot: ProgressSnapshot) -> str:
    lines = [
        f"# 實驗進度報告：{snapshot.run_id}",
        "",
        f"- 產生時間：{format_timestamp(snapshot.generated_at)}",
        f"- records：`{snapshot.records_dir}`",
        "",
    ]
    if not snapshot.has_data:
        lines += [f"**{NO_DATA}**：records 目錄不存在或沒有任何 record。", ""]
        return "\n".join(lines)

    lines += ["## 目前進行中的組別", "", current_combo_text(snapshot)]

    lines += [
        "",
        "## 整體完成率與 ETA",
        "",
        f"- {overall_text(snapshot)}",
        f"- 已完成題平均延遲：{format_latency(snapshot.mean_done_latency_ms)}",
        f"- ETA：{eta_text(snapshot, '無法估計（尚無 done 題的延遲資料）')}",
    ]
    if snapshot.finish_at is not None:
        lines.append(f"- 預計完成：{snapshot.finish_at.isoformat(timespec='seconds')}")

    lines += [
        "",
        f"## 各組即時指標（{len(snapshot.combos)} 組）",
        "",
        "完成題數＝status done；EM／F1 以每題最新一筆計、非 done 計 0；延遲 p50 含所有狀態。",
        "",
        "| dataset | model | condition | 完成題數 | 已嘗試 | EM | F1 | 延遲 p50 | 非 done | 狀態 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for combo in snapshot.combos:
        lines.append(
            f"| {combo.dataset} | {combo.model} | {combo.condition} | {combo.done} | {combo.attempted} "
            f"| {format_ratio(combo.em)} | {format_ratio(combo.f1)} | {format_latency(combo.latency_p50_ms)} "
            f"| {combo.non_done} | {combo.state} |"
        )

    lines += ["", f"## 最近 {RECENT_ERRORS} 筆錯誤（status ≠ done，新到舊）", ""]
    if snapshot.invalid_lines:
        lines += [f"> ⚠ {invalid_lines_text(snapshot.invalid_lines)}", ""]
    if snapshot.order_is_approximate:
        lines += [f"_{APPROXIMATE_ORDER}_", ""]
    if snapshot.recent_errors:
        lines += ["| 時間 | combo | question_id | status | error |", "|---|---|---|---|---|"]
        for record in snapshot.recent_errors:
            lines.append(
                f"| {record_time(record)} | {'__'.join(combo_key(record))} | {cell(record.get('question_id'))} "
                f"| {cell(record.get('status'))} | {cell(record.get('error'), 160)} |"
            )
    else:
        lines.append("無")

    lines += ["", f"## 最近 {RECENT_QUESTIONS} 題摘要（新到舊）", ""]
    if snapshot.order_is_approximate:
        lines += [f"_{APPROXIMATE_ORDER}_", ""]
    for index, record in enumerate(snapshot.recent_questions, start=1):
        correct = "對" if score(record, "em") >= 1 else "錯"
        lines += [
            f"{index}. `{'__'.join(combo_key(record))}` / `{record.get('question_id')}`（{record.get('status')}）",
            f"   - 問題：{cell(record.get('question') or '（record 未含題目文字）', 300)}",
            f"   - 預測：{cell(record.get('prediction'), 200)}",
            f"   - gold：{cell(record.get('gold'), 200)}",
            f"   - 對錯：{correct}（EM {score(record, 'em'):.0f}，F1 {score(record, 'f1'):.3f}）；跳數：{record.get('hops', '—')}",
        ]
    lines.append("")
    return "\n".join(lines)


def write_progress_report(
    data_root: Path, run_id: str, *, now: datetime | None = None, snapshot: ProgressSnapshot | None = None
) -> Path:
    snapshot = snapshot or collect_progress(data_root, run_id, now=now)
    return write_report_file(Path(data_root) / "reports" / "PROGRESS.md", render_progress_markdown(snapshot))


def write_report_file(path: Path, content: str) -> Path:
    """先寫暫存檔再取代，Windows 端開檔時不會讀到半份報告。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)
    return path


# ---- 共用文字（PROGRESS.md、dashboard、CLI） ----
def current_combo_text(snapshot: ProgressSnapshot) -> str:
    current = snapshot.current_combo
    if current is None:
        last = "__".join(combo_key(snapshot.ordered_records[-1])) if snapshot.ordered_records else "—"
        return f"無（最新一筆 record 所屬組別已完成：{last}）"
    return f"{current.combo}（done {current.done} / {current.target}，已嘗試 {current.attempted}）"


def overall_text(snapshot: ProgressSnapshot) -> str:
    done = f"已完成（status=done）{snapshot.done} / {snapshot.target_total} 題（{percent(snapshot.done, snapshot.target_total)}）；"
    if not snapshot.all_have_results:
        return f"{done}已嘗試 {snapshot.attempted} 題，其中非 done {snapshot.non_done} 題"
    breakdown = "、".join(f"{status} {count}" for status, count in snapshot.non_done_by_status.items())
    return (
        f"{done}已嘗試 {snapshot.attempted} / {snapshot.target_total} 題"
        f"（{ALL_HAVE_RESULTS}；非 done {snapshot.non_done} 題{'：' + breakdown if breakdown else ''}）"
    )


def eta_text(snapshot: ProgressSnapshot, unknown: str) -> str:
    if snapshot.all_have_results:
        return TERMINAL_ETA
    eta = snapshot.eta_seconds
    return format_duration(eta) if eta is not None else unknown


def invalid_lines_text(invalid_lines: list[tuple[str, int]]) -> str:
    listed = ", ".join(f"{name}:{number}" for name, number in invalid_lines[:MAX_LISTED_INVALID_LINES])
    more = f" 等（另 {len(invalid_lines) - MAX_LISTED_INVALID_LINES} 行）" if len(invalid_lines) > MAX_LISTED_INVALID_LINES else ""
    return f"有 {len(invalid_lines)} 行 record 無法解析，已略過：{listed}{more}"


# ---- 格式化 ----
def format_timestamp(moment: datetime) -> str:
    utc = moment.astimezone(timezone.utc)
    return f"{moment.isoformat(timespec='seconds')}（UTC {utc.isoformat(timespec='seconds')}）"


def format_ratio(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def format_latency(value_ms: float | None) -> str:
    return "—" if value_ms is None else f"{value_ms / 1000:.2f} s"


def format_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes // 1440} 天 {minutes % 1440 // 60} 小時 {minutes % 60} 分"


def percent(part: int, whole: int) -> str:
    if not whole:
        return "—"
    value = (Decimal(part) * 100 / Decimal(whole)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"{value}%"


def cell(value: Any, limit: int = 120) -> str:
    text = "—" if value is None or value == "" else str(value)
    text = " ".join(text.split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ---- 讀取 ----
def _read_records(records_dir: Path) -> tuple[list[dict[str, Any]], list[tuple[str, int]]]:
    """回傳（由舊到新的 record、無法解析的行）。"""
    if not records_dir.is_dir():
        return [], []
    keyed: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    invalid: list[tuple[str, int]] = []
    for path in sorted(records_dir.glob("*.jsonl")):
        mtime = path.stat().st_mtime
        items, bad = _read_jsonl(path)
        invalid += [(path.name, number) for number in bad]
        for number, record in items:
            finished = _finished_at(record)
            order = (1, finished, 0.0, "", 0) if finished is not None else (0, 0.0, mtime, path.name, number)
            keyed.append((order, record))
    keyed.sort(key=lambda item: item[0])  # 同 finished_at 時維持檔名、行號順序（穩定排序）
    return [record for _, record in keyed], invalid


def _read_jsonl(path: Path) -> tuple[list[tuple[int, dict[str, Any]]], list[int]]:
    """以 bytes 逐行讀（截斷的 UTF-8 只影響該行）；回傳（(行號, dict)、無法解析的行號）。"""
    items: list[tuple[int, dict[str, Any]]] = []
    bad: list[int] = []
    with path.open("rb") as handle:
        for number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                item = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                item = None
            if isinstance(item, dict):
                items.append((number, item))
            elif raw.endswith(b"\n"):
                bad.append(number)
            # 否則是檔尾沒有換行的半行：run 正在 append，下次重寫會讀到
    return items, bad


def _finished_at(record: dict[str, Any]) -> float | None:
    value = record.get("finished_at")
    if not isinstance(value, str):
        return None
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)  # 契約為 UTC
    return moment.timestamp()


def _latency(record: dict[str, Any]) -> float | None:
    value = record.get("latency_ms")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return manifest if isinstance(manifest, dict) else {}


def _manifest_int(manifest: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = manifest.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _manifest_names(manifest: dict[str, Any], key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = manifest.get(key)
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return tuple(value)
    return default


def _attach_question_text(data_root: Path, records: list[dict[str, Any]]) -> None:
    """record 未含題目文字時，從 prepare-data 的 sample_200.jsonl 補上（找不到就略過）。"""
    missing = {str(record.get("dataset")) for record in records if not record.get("question")}
    questions: dict[tuple[str, str], str] = {}
    for dataset in missing:
        path = data_root / "datasets" / dataset / "sample_200.jsonl"
        if not path.is_file():
            continue
        for _, item in _read_jsonl(path)[0]:
            questions[(dataset, str(item.get("id")))] = str(item.get("question", ""))
    for index, record in enumerate(records):
        text = questions.get((str(record.get("dataset")), str(record.get("question_id"))))
        if not record.get("question") and text:
            records[index] = {**record, "question": text}


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
