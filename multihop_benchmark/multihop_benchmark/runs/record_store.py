"""唯一寫 records 的地方：Data Root `runs/<run_id>/records/<combo>.jsonl`，append-only。

combo = `<dataset>__<model>__<condition>`，原樣作為檔名（不另做轉換）。每題完成即
append 一行並 flush；續跑以 completed_ids 取 `status == "done"` 的 question_id 跳過。
強制終止可能留下寫到一半的末行：讀取時略過並記 WARNING，下一次 append 先補換行，
不改寫既有內容。
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from multihop_benchmark.benchmark_logger import get_logger
from multihop_benchmark.settings import load_settings

VALID_STATUSES = ("done", "invalid_tool", "timeout", "oom", "error")
LAYA_NEXT_ACTIONS = ("A", "B", "C")  # answer / decompose_and_search / rephrase_and_search
LAYA_REMAINING_HOPS = (0, 1, 2)  # 2 代表 ≥ 2
_MISSING = object()


class RecordValidationError(ValueError):
    """record 不符合 Spec record schema；訊息含違規欄位路徑。"""


# ---- schema 驗證 -------------------------------------------------------------


def _fail(field: str, reason: str) -> None:
    raise RecordValidationError(f"record 欄位 {field} {reason}")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_str(obj: Mapping, key: str, path: str, *, non_empty: bool = False) -> None:
    value = obj.get(key, _MISSING)
    if value is _MISSING:
        _fail(path, "缺少")
    if not isinstance(value, str) or (non_empty and not value):
        _fail(path, f"必須是{'非空' if non_empty else ''}字串，目前為 {value!r}")


def _require_number(obj: Mapping, key: str, path: str, *, low: float = 0.0, high: float | None = None) -> None:
    value = obj.get(key, _MISSING)
    if value is _MISSING:
        _fail(path, "缺少")
    if not _is_number(value) or value < low or (high is not None and value > high):
        bound = f"[{low}, {high}]" if high is not None else f">= {low}"
        _fail(path, f"必須是 {bound} 的數值，目前為 {value!r}")


def _require_count(obj: Mapping, key: str, path: str) -> None:
    value = obj.get(key, _MISSING)
    if value is _MISSING:
        _fail(path, "缺少")
    if not _is_int(value) or value < 0:
        _fail(path, f"必須是 >= 0 的整數，目前為 {value!r}")


def _require_choice(obj: Mapping, key: str, path: str, choices: tuple) -> None:
    value = obj.get(key, _MISSING)
    if value is _MISSING:
        _fail(path, "缺少")
    # 以型別一併比對，避免 True == 1、1.0 == 1 混入整數 score。
    if not any(type(value) is type(choice) and value == choice for choice in choices):
        _fail(path, f"必須是 {'|'.join(map(str, choices))} 之一，目前為 {value!r}")


def _reject_unknown(obj: Mapping, allowed: set[str], path: str) -> None:
    for key in obj:
        if key not in allowed:
            _fail(f"{path}{key}", "不在 Spec record schema 內")


_RECORD_KEYS = {
    "run_id", "dataset", "model", "condition", "question_id", "gold", "gold_aliases", "prediction",
    "em", "f1", "status", "hops", "steps", "latency_ms", "vram_peak_mb", "error", "finished_at",
}
_STEP_KEYS = {"hop", "sub_question", "evidence_added", "laya", "llm"}
_LAYA_KEYS = {"sufficient_p", "next_action", "remaining", "latency_ms"}
_LLM_KEYS = {"prompt_tokens", "output_tokens", "latency_ms"}


def _validate_step(step: Any, path: str) -> None:
    if not isinstance(step, Mapping):
        _fail(path, f"必須是物件，目前為 {step!r}")
    _reject_unknown(step, _STEP_KEYS, f"{path}.")
    _require_count(step, "hop", f"{path}.hop")
    _require_str(step, "sub_question", f"{path}.sub_question")
    if not isinstance(step.get("evidence_added", _MISSING), list):
        _fail(f"{path}.evidence_added", "必須是陣列")

    laya = step.get("laya")
    if laya is not None:
        if not isinstance(laya, Mapping):
            _fail(f"{path}.laya", f"必須是物件或省略，目前為 {laya!r}")
        _reject_unknown(laya, _LAYA_KEYS, f"{path}.laya.")
        _require_number(laya, "sufficient_p", f"{path}.laya.sufficient_p", high=1.0)
        _require_choice(laya, "next_action", f"{path}.laya.next_action", LAYA_NEXT_ACTIONS)
        _require_choice(laya, "remaining", f"{path}.laya.remaining", LAYA_REMAINING_HOPS)
        _require_number(laya, "latency_ms", f"{path}.laya.latency_ms")

    llm = step.get("llm", _MISSING)
    if llm is _MISSING:
        _fail(f"{path}.llm", "缺少")
    if not isinstance(llm, Mapping):
        _fail(f"{path}.llm", f"必須是物件，目前為 {llm!r}")
    _reject_unknown(llm, _LLM_KEYS, f"{path}.llm.")
    _require_count(llm, "prompt_tokens", f"{path}.llm.prompt_tokens")
    _require_count(llm, "output_tokens", f"{path}.llm.output_tokens")
    _require_number(llm, "latency_ms", f"{path}.llm.latency_ms")


def validate_record(record: Any) -> None:
    """依 Spec record schema 檢查一筆 record；不合規則拋 RecordValidationError（含欄位路徑）。"""
    if not isinstance(record, Mapping):
        _fail("record", f"必須是物件，目前為 {type(record).__name__}")
    _reject_unknown(record, _RECORD_KEYS, "")
    for key in ("run_id", "dataset", "model", "condition", "question_id"):
        _require_str(record, key, key, non_empty=True)
    _require_str(record, "gold", "gold")
    aliases = record.get("gold_aliases", _MISSING)
    if not isinstance(aliases, list) or not all(isinstance(a, str) for a in aliases):
        _fail("gold_aliases", f"必須是字串陣列，目前為 {aliases!r}")
    _require_str(record, "prediction", "prediction")
    if record.get("em", _MISSING) not in (0, 1) or not _is_int(record["em"]):
        _fail("em", f"必須是 0 或 1，目前為 {record.get('em')!r}")
    _require_number(record, "f1", "f1", high=1.0)
    if record.get("status") not in VALID_STATUSES:
        _fail("status", f"必須是 {'|'.join(VALID_STATUSES)} 之一，目前為 {record.get('status')!r}")
    _require_count(record, "hops", "hops")

    steps = record.get("steps", _MISSING)
    if not isinstance(steps, list):
        _fail("steps", f"必須是陣列，目前為 {steps!r}")
    for index, step in enumerate(steps):
        _validate_step(step, f"steps[{index}]")

    _require_number(record, "latency_ms", "latency_ms")
    if record.get("vram_peak_mb", _MISSING) is not None:
        _require_number(record, "vram_peak_mb", "vram_peak_mb")
    if record.get("error") is not None and not isinstance(record["error"], str):
        _fail("error", f"必須是字串或省略，目前為 {record['error']!r}")
    if "finished_at" in record:  # 選填：benchmark_runner 寫入的完成時間（UTC ISO 8601，含時區）
        stamp = record["finished_at"]
        try:
            parsed = datetime.fromisoformat(stamp) if isinstance(stamp, str) else None
        except ValueError:
            parsed = None
        if parsed is None or parsed.utcoffset() is None:
            _fail("finished_at", f"必須是含時區的 ISO 8601 字串或省略，目前為 {stamp!r}")


# ---- 讀寫 --------------------------------------------------------------------


def _safe_part(value: str, name: str) -> str:
    if not isinstance(value, str) or value in ("", ".", "..") or any(ch in value for ch in "/\\\0"):
        raise ValueError(f"{name} 不可作為檔名：{value!r}")
    return value


def records_path(run_id: str, combo: str, *, data_root: Path | None = None) -> Path:
    root = Path(data_root) if data_root is not None else load_settings().data_root
    return root / "runs" / _safe_part(run_id, "run_id") / "records" / f"{_safe_part(combo, 'combo')}.jsonl"


def append(run_id: str, combo: str, record: Mapping[str, Any], *, data_root: Path | None = None) -> Path:
    """驗證後把 record 以一行 JSON append 到 combo 的 records 檔，flush 並 fsync。"""
    path = records_path(run_id, combo, data_root=data_root)
    validate_record(record)
    if record["run_id"] != run_id:
        _fail("run_id", f"為 {record['run_id']!r}，與寫入目標 run_id {run_id!r} 不符")

    line = json.dumps(record, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as fh:
        if fh.tell() > 0:
            with path.open("rb") as tail:
                tail.seek(-1, os.SEEK_END)
                if tail.read(1) != b"\n":  # 前次中斷留下半行：另起新行，不改寫舊內容
                    line = "\n" + line
        fh.write(line.encode("utf-8"))
        fh.flush()
        os.fsync(fh.fileno())
    return path


def completed_ids(run_id: str, combo: str, *, data_root: Path | None = None) -> set[str]:
    """回傳 records 檔中 `status == "done"` 的 question_id；檔案不存在時為空集合。"""
    path = records_path(run_id, combo, data_root=data_root)
    if not path.is_file():
        return set()

    done: set[str] = set()
    # 以 bytes 逐行讀：中斷可能截在多位元組 UTF-8 字元中間，解碼錯誤只影響該行。
    with path.open("rb") as fh:
        for number, raw in enumerate(fh, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                row = None
            if not isinstance(row, dict):
                get_logger("record_store", log_dir=path.parents[3] / "logs").warning(
                    f"略過無法解析的 record 行 {path.name}:{number}",
                    event="record_line_invalid",
                    run_id=run_id,
                    combo=combo,
                )
                continue
            if row.get("status") == "done" and isinstance(row.get("question_id"), str):
                done.add(row["question_id"])
    return done
