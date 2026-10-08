"""Laya 決策層評估：held-out `laya_decisions/test.jsonl` 上的分組指標、PASS／FAIL 判定與報告。

指標沿用 `laya_zh_tw/04_test_laya.py`，依問題型別分組（sufficient=noul、next_action=choice、remaining_hops=score）：

- `accuracy`：argmax 與 target argmax 相同的比例。
- `soft_accuracy`、`brier`（Σ(p−target)²）：只算 noul 與 choice（同 04）。
- `ece`：`laya.common.ece_score(max(p), correct)`，15 bins，每組各算。
- `mae`（門檻用，同 04）：`|期望分數 Σ i·p_i − gold|`；`mae_argmax`：`|argmax − gold|`（= decision client 回傳的
  `remaining`）；`within_one`：期望分數與 gold 差 ≤ 1 的比例。
- `p50_latency_ms`：一次 `predict` 的中位延遲。同一 state 的相鄰列（型別不重複）合併成一次 predict 回答，
  與 `LayaDecisionClient.decide` 一次回答三題相同，因此這是「一次決策」的延遲。

門檻（Spec）：sufficient 與 next_action accuracy ≥ 0.75 且比 zero-shot 基底高 ≥ 0.20；remaining_hops MAE ≤ 0.5。
比較容許 1e-9 浮點誤差（0.75 − 0.55 在浮點下是 0.19999…）。
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from multihop_benchmark.laya_decision.laya_decision_client import load_agent
from multihop_benchmark.laya_decision.weak_label_generator import DECISION_QUESTIONS

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

QUESTION_BY_TYPE = {q["type"]: name for name, q in DECISION_QUESTIONS.items()}
THRESHOLDS = {"accuracy_min": 0.75, "gain_over_zero_shot_min": 0.20, "remaining_mae_max": 0.5}
EPSILON = 1e-9
REPORT_NAME = "laya_eval_report"
TUNABLE_PARAMETERS = [
    ("--epochs", "epochs", "訓練輪數；欠擬合時增加"),
    ("--lr-encoder", "lr_encoder", "encoder 學習率"),
    ("--lr-head", "lr_head", "decision head 學習率"),
    ("--micro-batch", "micro_batch", "每次前向的決策筆數"),
    ("--grad-accum", "grad_accum", "梯度累積次數（effective batch = micro batch × grad accum）"),
    ("--group-size", "group_size", "RLCD 每筆的探索樣本數（≥ 2）"),
    ("--sigma-start", "sigma_start", "Gaussian exploration 起始幅度"),
    ("--sigma-end", "sigma_end", "Gaussian exploration 結束幅度"),
    ("--ce-weight", "ce_weight", "soft cross-entropy guidance 權重"),
    ("--seed", "seed", "隨機種子"),
]


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if limit is not None and len(rows) >= limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _external_question(row: Mapping[str, Any]) -> dict[str, Any]:
    return {"type": row["type"], "instructions": row["instructions"], "criteria": row.get("criteria")}


def _probabilities(row: Mapping[str, Any], answer: Mapping[str, Any]) -> np.ndarray:
    """同 04 的 `probabilities_from_result`：依 target 的選項順序排列。"""
    if row["type"] == "noul":
        p_true = float(answer["noul"])
        return np.array([1.0 - p_true, p_true], dtype=float)
    probs = answer["probabilities"]
    if row["type"] == "choice":
        return np.array([float(probs[k]) for k in row["criteria"]], dtype=float)
    return np.array([float(probs[str(i)]) for i in range(len(row["criteria"]))], dtype=float)


def _groups(rows: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    """相鄰且 state 相同、型別不重複的列合成一組（一次 predict）。"""
    groups: list[list[Mapping[str, Any]]] = []
    for row in rows:
        last = groups[-1] if groups else None
        if last and last[0]["state"] == row["state"] and row["type"] not in {r["type"] for r in last}:
            last.append(row)
        else:
            groups.append([row])
    return groups


def _ece(confidence: list[float], correct: list[float]) -> float:
    from laya.common import ece_score

    return float(ece_score(np.array(confidence), np.array(correct)))


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def evaluate_agent(agent: Any, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """回傳 `{decisions, predict_calls, p50_latency_ms, overall{n, accuracy, ece}, by_question{name: 指標}}`。"""
    stats: dict[str, dict[str, list[float]]] = {}
    latencies = []
    for group in _groups(rows):
        questions = {QUESTION_BY_TYPE[row["type"]]: _external_question(row) for row in group}
        started = time.perf_counter()
        answers = agent.predict(state=group[0]["state"], questions=questions)["answers"]
        latencies.append((time.perf_counter() - started) * 1000.0)
        for row in group:
            name = QUESTION_BY_TYPE[row["type"]]
            answer = answers[name]
            s = stats.setdefault(name, {k: [] for k in ("correct", "confidence", "soft", "brier", "mae",
                                                        "mae_argmax", "within_one")})
            p = _probabilities(row, answer)
            target = np.array(row["target"], dtype=float)
            pred, gold = int(np.argmax(p)), int(np.argmax(target))
            s["correct"].append(float(pred == gold))
            s["confidence"].append(float(np.max(p)))
            if row["type"] != "score":
                s["soft"].append(float(np.sum(p * target)))
                s["brier"].append(float(np.sum((p - target) ** 2)))
            else:
                expected = float(np.sum(np.arange(len(p)) * p))
                s["mae"].append(abs(expected - gold))
                s["mae_argmax"].append(float(abs(pred - gold)))
                s["within_one"].append(float(abs(expected - gold) <= 1.0))

    by_question = {}
    for name in DECISION_QUESTIONS:
        if name not in stats:
            continue
        s = stats[name]
        metrics = {"n": len(s["correct"]), "accuracy": _mean(s["correct"]), "ece": _ece(s["confidence"], s["correct"])}
        if s["soft"]:
            metrics |= {"soft_accuracy": _mean(s["soft"]), "brier": _mean(s["brier"])}
        if s["mae"]:
            metrics |= {"mae": _mean(s["mae"]), "mae_argmax": _mean(s["mae_argmax"]),
                        "within_one": _mean(s["within_one"])}
        by_question[name] = metrics
    correct = [c for s in stats.values() for c in s["correct"]]
    confidence = [c for s in stats.values() for c in s["confidence"]]
    return {
        "decisions": len(rows),
        "predict_calls": len(latencies),
        "p50_latency_ms": float(statistics.median(latencies)) if latencies else None,
        "overall": {"n": len(correct), "accuracy": _mean(correct), "ece": _ece(confidence, correct) if correct else None},
        "by_question": by_question,
    }


def evaluate(
    model_dir: str | Path, test_path: str | Path, *, device: str = "cuda", limit: int | None = None,
) -> dict[str, Any]:
    """載入 `model_dir`（本地目錄或 HF id）並在 `test_path` 前 `limit` 列上評估；回傳同 `evaluate_agent`。"""
    agent = load_agent(str(model_dir), device)
    try:
        return evaluate_agent(agent, read_jsonl(test_path, limit))
    finally:
        del agent
        _release_cuda()


def _release_cuda() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _metric(result: Mapping[str, Any], name: str, key: str) -> float | None:
    return result.get("by_question", {}).get(name, {}).get(key)


def judge(finetuned: Mapping[str, Any], zero_shot: Mapping[str, Any],
          thresholds: Mapping[str, float] = THRESHOLDS) -> dict[str, Any]:
    """回傳 `{passed, checks[{name, value, threshold, passed}]}`；缺指標的檢查視為不通過。"""
    checks = []

    def check(name: str, value: float | None, threshold: float, at_least: bool) -> None:
        ok = value is not None and (value >= threshold - EPSILON if at_least else value <= threshold + EPSILON)
        checks.append({"name": name, "value": value, "threshold": (">= " if at_least else "<= ") + f"{threshold:g}",
                       "passed": ok})

    for name in ("sufficient", "next_action"):
        accuracy, base = _metric(finetuned, name, "accuracy"), _metric(zero_shot, name, "accuracy")
        check(f"{name} accuracy", accuracy, thresholds["accuracy_min"], True)
        gain = None if accuracy is None or base is None else accuracy - base
        check(f"{name} gain over zero-shot", gain, thresholds["gain_over_zero_shot_min"], True)
    check("remaining_hops MAE", _metric(finetuned, "remaining_hops", "mae"), thresholds["remaining_mae_max"], False)
    return {"passed": all(c["passed"] for c in checks), "checks": checks}


def _fmt(value: Any, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_report(
    out_dir: str | Path, *, finetuned: Mapping[str, Any], zero_shot: Mapping[str, Any], verdict: Mapping[str, Any],
    train_summary: Mapping[str, Any] | None, model_dir: str | Path, model_hash: str | None,
    test_path: str | Path, limit: int | None,
) -> tuple[Path, Path]:
    """寫 `laya_eval_report.md` 與 `.json`；FAIL 時列出可調參數與目前值。回傳兩個路徑。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    status = "PASS" if verdict["passed"] else "FAIL"
    hyper = (train_summary or {}).get("hyperparameters", {})
    tunable = [{"flag": flag, "current": hyper.get(key), "note": note} for flag, key, note in TUNABLE_PARAMETERS]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status,
        "verdict": dict(verdict),
        "thresholds": THRESHOLDS,
        "model_dir": str(model_dir),
        "model_hash": model_hash,
        "test_path": str(test_path),
        "test_sha256": _sha256(Path(test_path)),
        "limit": limit,
        "train_summary": train_summary,
        "finetuned": finetuned,
        "zero_shot": zero_shot,
        "tunable_parameters": tunable if not verdict["passed"] else [],
    }
    json_path = out_dir / f"{REPORT_NAME}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    elapsed = (train_summary or {}).get("elapsed_s")
    lines = [
        "# Laya 多跳決策層評估報告",
        "",
        f"## 判定：{status}",
        "",
        f"- 產生時間：{payload['generated_at']}",
        f"- 模型目錄：`{model_dir}`",
        f"- 模型 hash（`laya_decision_client.model_hash`）：`{model_hash or '—'}`",
        f"- 測試集：`{test_path}`（sha256 `{payload['test_sha256'] or '—'}`），決策 {finetuned['decisions']} 筆、"
        f"predict {finetuned['predict_calls']} 次",
    ]
    if limit is not None:
        lines.append(f"- **冒煙模式**：`--limit {limit}`，只用各檔前 {limit} 列，非正式結果")
    if train_summary:
        lines += [
            f"- 訓練耗時：{elapsed / 60:.1f} min（{train_summary.get('gpu', '—')}），"
            f"train {train_summary.get('train_items', '—')} 筆、calibration {train_summary.get('calibration_items', '—')} 筆",
            f"- 溫度 [choice, score, noul]：{[round(t, 3) for t in train_summary.get('temperature') or []] or '—'}",
            f"- 超參數：`{json.dumps(hyper, ensure_ascii=False)}`",
        ]
    else:
        lines.append("- 訓練耗時：— （找不到 train_summary.json）")
    lines += ["", "## 門檻", "", "| 檢查 | 數值 | 門檻 | 結果 |", "|---|---|---|---|"]
    lines += [f"| {c['name']} | {_fmt(c['value'])} | {c['threshold']} | {'PASS' if c['passed'] else 'FAIL'} |"
              for c in verdict["checks"]]
    lines += ["", "## 指標（微調 vs zero-shot 基底 `convaiinnovations/laya-multilingual`）", "",
              "| 問題 | 模型 | n | accuracy | soft accuracy | Brier | ECE | MAE | MAE (argmax) | within-one |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for name in DECISION_QUESTIONS:
        for label, result in (("微調", finetuned), ("zero-shot", zero_shot)):
            m = result["by_question"].get(name)
            if m:
                lines.append(f"| {name} | {label} | {m['n']} | " + " | ".join(
                    _fmt(m.get(k)) for k in ("accuracy", "soft_accuracy", "brier", "ece", "mae", "mae_argmax",
                                             "within_one")) + " |")
    lines += [
        "",
        f"- 整體 accuracy：微調 {_fmt(finetuned['overall']['accuracy'])}、zero-shot {_fmt(zero_shot['overall']['accuracy'])}；"
        f"整體 ECE：微調 {_fmt(finetuned['overall']['ece'])}、zero-shot {_fmt(zero_shot['overall']['ece'])}",
        f"- p50 延遲（一次 predict 回答同一 state 的三題）：微調 {_fmt(finetuned['p50_latency_ms'], 1)} ms、"
        f"zero-shot {_fmt(zero_shot['p50_latency_ms'], 1)} ms",
        "- MAE 用期望分數（同 `04_test_laya.py`，門檻依此）；MAE (argmax) 為 decision client 回傳的 `remaining` 誤差。"
        "Soft accuracy 與 Brier 只算 noul／choice。",
    ]
    if not verdict["passed"]:
        lines += ["", "## 未達門檻：可調參數（`train-laya` 旗標）", "", "| 旗標 | 目前值 | 說明 |", "|---|---|---|"]
        lines += [f"| `{t['flag']}` | {'—' if t['current'] is None else f'{t["current"]:g}'} | {t['note']} |"
                  for t in tunable]
    md_path = out_dir / f"{REPORT_NAME}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path
