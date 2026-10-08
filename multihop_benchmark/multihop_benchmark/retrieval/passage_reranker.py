"""`bge-reranker-v2-m3` cross-encoder 重排（Spec「RAG 管線」）。

`rerank(query, passages, top_n)` 對每段 `(query, 標題＋全文)` 打分，依分數由高到低（同分保留輸入順序）
截取 `top_n`；回傳段落是輸入的複本，保留原欄位並加上 `rerank_score`。
測試接縫：`model` 具 `predict(pairs) -> scores`（sentence-transformers `CrossEncoder` 的介面），可注入假物件。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from multihop_benchmark.retrieval.dense_index import DEFAULT_DEVICE, passage_chunk

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
PREDICT_BATCH_SIZE = 32


class CrossEncoderModel(Protocol):
    def predict(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]: ...


def load_cross_encoder(device: str = DEFAULT_DEVICE) -> CrossEncoderModel:
    """載入 bge-reranker-v2-m3（HF 快取於 `HF_HOME`）；分數經 sigmoid 落在 0–1。"""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(RERANKER_MODEL, device=device)

    class _Batched:
        def predict(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]:
            return model.predict(list(pairs), batch_size=PREDICT_BATCH_SIZE, show_progress_bar=False)

    return _Batched()


def rerank(
    query: str, passages: Sequence[Mapping[str, Any]], top_n: int, *, model: CrossEncoderModel
) -> list[dict[str, Any]]:
    if top_n <= 0:
        raise ValueError(f"top_n 必須是正整數，收到 {top_n}")
    if not passages:
        return []
    scores = model.predict([(query, passage_chunk(passage)) for passage in passages])
    ranked = sorted(
        ({**passage, "rerank_score": float(score)} for passage, score in zip(passages, scores)),
        key=lambda passage: passage["rerank_score"],
        reverse=True,
    )
    return ranked[:top_n]


class PassageReranker:
    """持有已載入的 cross-encoder；未注入 `model` 時依 `device` 載入 bge-reranker-v2-m3。"""

    def __init__(self, device: str = DEFAULT_DEVICE, *, model: CrossEncoderModel | None = None) -> None:
        self.model = model or load_cross_encoder(device)

    def rerank(self, query: str, passages: Sequence[Mapping[str, Any]], top_n: int) -> list[dict[str, Any]]:
        return rerank(query, passages, top_n, model=self.model)
