"""`DecisionModel` 的 Laya 實作與模型目錄雜湊。

- `LayaDecisionClient(model_dir, device="cuda").decide(state)`：以 `decision_state_builder.build_state`
  組 state 文字（與弱標籤訓練資料同一函式、同一預設 tokenizer），一次 `predict` 回答
  `weak_label_generator.DECISION_QUESTIONS` 的三題，轉成
  `{sufficient_p: P(true) ∈ [0,1], next_action: A|B|C, remaining: score 分布 argmax ∈ {0,1,2}, latency_ms}`。
  `latency_ms` 只量 `predict`（不含 state 組裝）。
- `model_hash(model_dir)`：目錄內所有檔案（略過 `.` 開頭的暫存檔）依相對路徑排序，對「相對路徑 + 檔案 sha256」
  逐行再取 sha256；與目錄位置無關。供 benchmark_runner 的 manifest 使用。
- torch 2.14 的 ModernBERT 需 `TORCH_DISABLE_NATIVE_JIT=1`，且必須在 torch 第一次匯入前設定；本模組在匯入時
  `setdefault`，若呼叫端已先匯入 torch（例如先載入 BGE-M3），須由呼叫端在程序啟動時設定。
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from multihop_benchmark.agent.protocols import Decision, DecisionState
from multihop_benchmark.laya_decision.decision_state_builder import Tokenizer, build_state
from multihop_benchmark.laya_decision.weak_label_generator import ACTIONS, DECISION_QUESTIONS

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

REMAINING_LEVELS = ("0", "1", "2")


def load_agent(model_dir: str, device: str) -> Any:
    import laya

    return laya.load(model_dir, device=device)


def decision_from_answers(answers: Mapping[str, Any]) -> tuple[float, str, int]:
    """把 `predict()["answers"]` 轉成 (sufficient_p, next_action, remaining)。

    next_action 不在 A/B/C，或 remaining_hops 機率的鍵不是恰好 "0"／"1"／"2"、值不是有限數字時丟 ValueError
    （訊息含欄位名），確保回傳值符合 `Decision` 契約。"""
    sufficient_p = min(1.0, max(0.0, float(answers["sufficient"]["noul"])))
    next_action = answers["next_action"]["choice"]
    if next_action not in ACTIONS:
        raise ValueError(f"next_action 不在 {ACTIONS}：{next_action!r}")
    probs = answers["remaining_hops"]["probabilities"]
    if not isinstance(probs, Mapping) or set(probs) != set(REMAINING_LEVELS):
        raise ValueError(f"remaining_hops 機率的鍵必須恰為 {REMAINING_LEVELS}：{probs!r}")
    values = []
    for level in REMAINING_LEVELS:
        try:
            value = float(probs[level])
        except (TypeError, ValueError):
            value = math.nan
        if not math.isfinite(value):
            raise ValueError(f"remaining_hops 機率 {level!r} 不是有限數字：{probs[level]!r}")
        values.append(value)
    return sufficient_p, next_action, max(range(len(values)), key=values.__getitem__)


class LayaDecisionClient:
    """實作 `agent.protocols.DecisionModel`；`agent`／`tokenizer` 供測試注入。"""

    def __init__(
        self, model_dir: str | Path, device: str = "cuda", *, agent: Any = None, tokenizer: Tokenizer | None = None,
    ) -> None:
        self.model_dir = str(model_dir)
        self.agent = agent if agent is not None else load_agent(self.model_dir, device)
        self.tokenizer = tokenizer

    def decide(self, state: DecisionState) -> Decision:
        text = build_state(state["question"], state["sub_questions"], state["evidence"], tokenizer=self.tokenizer)
        started = time.perf_counter()
        result = self.agent.predict(state=text, questions=DECISION_QUESTIONS)
        latency_ms = (time.perf_counter() - started) * 1000.0
        sufficient_p, next_action, remaining = decision_from_answers(result["answers"])
        return {"sufficient_p": sufficient_p, "next_action": next_action, "remaining": remaining,
                "latency_ms": float(latency_ms)}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def model_hash(model_dir: str | Path) -> str:
    root = Path(model_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"模型目錄不存在：{root}")
    files = sorted(
        (path.relative_to(root).as_posix(), path) for path in root.rglob("*")
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(root).parts)
    )
    digest = hashlib.sha256()
    for relative, path in files:
        digest.update(f"{relative}\t{_file_sha256(path)}\n".encode("utf-8"))
    return digest.hexdigest()
