from multihop_benchmark.retrieval import Retriever, supporting_title_report


class ListIndex:
    """假 dense 索引：依查詢回傳預設的 pid 清單（已排序），並記錄收到的 k。"""

    def __init__(self, results):
        self.results = results
        self.ks = []

    def search(self, query, k):
        self.ks.append(k)
        pids = self.results.get(query, [])[:k]
        return [{"pid": pid, "score": 1.0 - i / 100, "title": pid.upper(), "text": f"text {pid}"} for i, pid in enumerate(pids)]


class ScoreReranker:
    """假重排器：依預設 pid 分數排序（未列出者 0 分）。"""

    def __init__(self, scores):
        self.scores = scores

    def rerank(self, query, passages, top_n):
        ranked = sorted(passages, key=lambda p: self.scores.get(p["pid"], 0.0), reverse=True)
        return [{**p, "rerank_score": self.scores.get(p["pid"], 0.0)} for p in ranked[:top_n]]


def test_retrieve_takes_dense_top_20_then_keeps_reranked_top_3():
    candidates = [f"p{i}" for i in range(25)]
    index = ListIndex({"q": candidates})
    retriever = Retriever(index, ScoreReranker({"p19": 0.9, "p3": 0.8, "p20": 0.99, "p7": 0.7, "p0": 0.1}))

    passages = retriever.retrieve("q")

    assert index.ks == [20]
    # p20 在 dense 第 21 名之外，不可能被重排撈回
    assert [p["pid"] for p in passages] == ["p19", "p3", "p7"]
    assert all({"pid", "title", "text"} <= set(p) for p in passages)


def _paragraph(pid, title, supporting):
    return {"pid": pid, "title": title, "text": "...", "is_supporting": supporting, "supporting_sentences": None}


QUESTIONS = [
    {"id": "q1", "paragraphs": [_paragraph("n1", "N1", False), _paragraph("s1", "T1", True)]},
    {"id": "q2", "paragraphs": [_paragraph("s2", "T2", True), _paragraph("n2", "N2", False)]},
    {"id": "q3", "paragraphs": [_paragraph("n3", "N3", False), _paragraph("s3", "T3", True)]},
]
INDEX_RESULTS = {
    "T1": ["s1", "x1", "x2", "x3"],  # dense 第 1、重排第 1
    "T2": ["x1", "x2", "x3", "x4", "s2"],  # dense 第 5、重排後第 5（在 top-3 外）
    "T3": ["x1", "x2"],  # dense 沒找到
}
RERANK_SCORES = {"s1": 0.9, "x1": 0.8, "x2": 0.7, "x3": 0.6, "x4": 0.5, "s2": 0.1}


def test_report_measures_supporting_title_hits_in_dense_top_k_and_reranked_top_n():
    report = supporting_title_report(QUESTIONS, ListIndex(INDEX_RESULTS), ScoreReranker(RERANK_SCORES), sample_size=20)

    assert report["n"] == 3
    assert report["dense_hit_rate"] == 2 / 3
    assert report["rerank_top_n_rate"] == 1 / 3
    items = {item["question_id"]: item for item in report["items"]}
    assert items["q1"] == {"question_id": "q1", "query": "T1", "pid": "s1", "dense_rank": 1, "rerank_rank": 1}
    assert (items["q2"]["dense_rank"], items["q2"]["rerank_rank"]) == (5, 5)
    assert (items["q3"]["dense_rank"], items["q3"]["rerank_rank"]) == (None, None)


def test_report_sampling_is_fixed_by_seed_and_limited_to_sample_size():
    def run(seed):
        return supporting_title_report(
            QUESTIONS, ListIndex(INDEX_RESULTS), ScoreReranker(RERANK_SCORES), sample_size=2, seed=seed
        )

    first, again = run(7), run(7)

    assert first == again
    assert first["n"] == 2 and first["seed"] == 7
    assert {item["question_id"] for item in first["items"]} <= {"q1", "q2", "q3"}
    assert all(item["pid"].startswith("s") for item in first["items"])
