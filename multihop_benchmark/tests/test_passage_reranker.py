import pytest

from multihop_benchmark.retrieval.passage_reranker import PassageReranker, rerank

PASSAGES = [
    {"pid": "a", "score": 0.9, "title": "A", "text": "cat"},
    {"pid": "b", "score": 0.8, "title": "B", "text": "dog dog dog"},
    {"pid": "c", "score": 0.7, "title": "C", "text": "dog"},
    {"pid": "d", "score": 0.6, "title": "D", "text": "dog dog"},
]


class OverlapCrossEncoder:
    """假 cross-encoder：分數 = 查詢詞在段落中出現次數。"""

    def __init__(self):
        self.calls = 0

    def predict(self, pairs):
        self.calls += 1
        return [float(passage.split().count(query)) for query, passage in pairs]


def test_rerank_orders_by_cross_encoder_score_and_truncates_to_top_n():
    ranked = rerank("dog", PASSAGES, top_n=2, model=OverlapCrossEncoder())

    assert [p["pid"] for p in ranked] == ["b", "d"]
    assert [p["rerank_score"] for p in ranked] == [3.0, 2.0]
    # 原欄位保留（含 dense 分數），輸入不被修改
    assert ranked[0] == {"pid": "b", "score": 0.8, "title": "B", "text": "dog dog dog", "rerank_score": 3.0}
    assert "rerank_score" not in PASSAGES[1]


def test_top_n_larger_than_candidates_returns_all_sorted():
    ranked = rerank("dog", PASSAGES, top_n=10, model=OverlapCrossEncoder())

    assert [p["pid"] for p in ranked] == ["b", "d", "c", "a"]


def test_empty_candidates_skip_the_model():
    model = OverlapCrossEncoder()

    assert rerank("dog", [], top_n=3, model=model) == []
    assert model.calls == 0


def test_passage_reranker_uses_injected_model():
    reranker = PassageReranker(model=OverlapCrossEncoder())

    assert [p["pid"] for p in reranker.rerank("cat", PASSAGES, top_n=1)] == ["a"]


def test_top_n_must_be_positive():
    with pytest.raises(ValueError):
        rerank("dog", PASSAGES, top_n=0, model=OverlapCrossEncoder())
