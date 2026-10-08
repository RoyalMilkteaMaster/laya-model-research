"""檢索堆疊：BGE-M3 dense 索引、bge-reranker-v2-m3 重排與 Retriever。"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

DENSE_K = 20
TOP_N = 3
REPORT_SAMPLE_SIZE = 20
REPORT_SEED = 42


class _Searchable(Protocol):
    def search(self, query: str, k: int) -> list[dict[str, Any]]: ...


class _Reranking(Protocol):
    def rerank(self, query: str, passages: Sequence[Mapping[str, Any]], top_n: int) -> list[dict[str, Any]]: ...


class Retriever:
    """Spec 的 `Retriever` 行為：`retrieve(query) -> list[passage]`。

    dense 取 top-`dense_k`（20）→ cross-encoder 重排 → 回傳前 `top_n`（3）段，依重排分數由高到低。
    每段是 `{pid, title, text, score, rerank_score}` dict（`score` 為 dense cosine），符合 Ticket 08
    `agent.protocols.Retriever` 的簽章；依 `pid` 去重與 evidence 上限由迴圈負責。
    """

    def __init__(self, index: _Searchable, reranker: _Reranking, *, dense_k: int = DENSE_K, top_n: int = TOP_N) -> None:
        self.index = index
        self.reranker = reranker
        self.dense_k = dense_k
        self.top_n = top_n

    @classmethod
    def load(cls, dataset: str, *, device: str = "cuda", indexes_dir: Path | None = None) -> Retriever:
        """載入 `indexes/<dataset>/` 與 BGE-M3、bge-reranker-v2-m3（皆放在 `device`）。"""
        from multihop_benchmark.retrieval.dense_index import DenseIndex
        from multihop_benchmark.retrieval.passage_reranker import PassageReranker

        return cls(DenseIndex.load(dataset, indexes_dir=indexes_dir, device=device), PassageReranker(device))

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        return self.reranker.rerank(query, self.index.search(query, self.dense_k), self.top_n)


def _rank(passages: Sequence[Mapping[str, Any]], pid: str) -> int | None:
    return next((rank for rank, passage in enumerate(passages, start=1) if passage["pid"] == pid), None)


def supporting_title_report(
    questions: Sequence[Mapping[str, Any]],
    index: _Searchable,
    reranker: _Reranking,
    *,
    sample_size: int = REPORT_SAMPLE_SIZE,
    seed: int = REPORT_SEED,
    dense_k: int = DENSE_K,
    top_n: int = TOP_N,
) -> dict[str, Any]:
    """以固定 seed 抽題，用其任一支持段落的標題查詢，量測該段落在 dense top-k 與重排後 top-n 的比例。

    `dense_rank`／`rerank_rank` 為 1 起算名次（重排對整個 dense top-k 排序），不在候選內為 `None`。
    """
    rng = random.Random(seed)
    candidates = [q for q in questions if any(p["is_supporting"] for p in q["paragraphs"])]
    sampled = rng.sample(candidates, min(sample_size, len(candidates)))
    items = []
    for question in sampled:
        target = rng.choice([p for p in question["paragraphs"] if p["is_supporting"]])
        dense = index.search(target["title"], dense_k)
        reranked = reranker.rerank(target["title"], dense, len(dense)) if dense else []
        items.append({
            "question_id": question["id"], "query": target["title"], "pid": target["pid"],
            "dense_rank": _rank(dense, target["pid"]), "rerank_rank": _rank(reranked, target["pid"]),
        })
    n = len(items)
    return {
        "n": n, "seed": seed, "dense_k": dense_k, "top_n": top_n,
        "dense_hit_rate": sum(item["dense_rank"] is not None for item in items) / n if n else None,
        "rerank_top_n_rate": sum((item["rerank_rank"] or top_n + 1) <= top_n for item in items) / n if n else None,
        "items": items,
    }
