import json
import math
from pathlib import Path

import numpy as np
import pytest

from multihop_benchmark.retrieval.dense_index import DenseIndex, build_index

CORPUS = [
    json.loads(line)
    for line in (Path(__file__).parent / "fixtures" / "retrieval" / "corpus_5.jsonl").read_text("utf-8").splitlines()
]
VOCAB = ("river", "city", "painter", "music")


class KeywordEmbedder:
    """假嵌入：每維是一個關鍵字在文字中出現的次數（未正規化，fixture 標題不含關鍵字）。"""

    def encode(self, texts):
        return np.array([[text.lower().split().count(word) for word in VOCAB] for text in texts], dtype=np.float64)


def _built(tmp_path, dataset="toy"):
    build_index(dataset, CORPUS, "cpu", indexes_dir=tmp_path, embedder=KeywordEmbedder())
    return tmp_path / dataset


def test_search_ranks_by_cosine_with_flat_inner_product(tmp_path):
    _built(tmp_path)
    index = DenseIndex.load("toy", indexes_dir=tmp_path, embedder=KeywordEmbedder())

    hits = index.search("river", k=3)

    # 查詢 (1,0,0,0)：Danube (1,0,0,0) = 1；Seine (2,1,0,0) = 2/√5；Monet (1,0,1,1) = 1/√3
    assert [hit["pid"] for hit in hits] == ["p-danube", "p-seine", "p-monet"]
    assert [hit["score"] for hit in hits] == pytest.approx([1.0, 2 / math.sqrt(5), 1 / math.sqrt(3)], abs=1e-6)
    assert hits[1] == {"pid": "p-seine", "score": hits[1]["score"], "title": "Seine", "text": "river river city"}


def test_index_files_hold_float32_normalized_rows_aligned_with_passage_ids(tmp_path):
    index_dir = _built(tmp_path)

    embeddings = np.load(index_dir / "embeddings.npy")
    meta = json.loads((index_dir / "passage_ids.json").read_text("utf-8"))

    assert embeddings.dtype == np.float32
    assert embeddings.shape == (5, len(VOCAB))
    assert np.linalg.norm(embeddings, axis=1) == pytest.approx(np.ones(5), abs=1e-6)
    assert [(row["pid"], row["title"]) for row in meta["passages"]] == [(row["pid"], row["title"]) for row in CORPUS]
    # 第 0 列是 Seine (2,1,0,0) 正規化後的結果
    assert embeddings[0] == pytest.approx(np.array([2, 1, 0, 0]) / math.sqrt(5), abs=1e-6)


def test_reloaded_index_returns_identical_results(tmp_path):
    _built(tmp_path)

    first = DenseIndex.load("toy", indexes_dir=tmp_path, embedder=KeywordEmbedder()).search("painter city", k=5)
    second = DenseIndex.load("toy", indexes_dir=tmp_path, embedder=KeywordEmbedder()).search("painter city", k=5)

    assert first == second
    assert first[0]["pid"] == "p-louvre"


def test_k_larger_than_corpus_returns_every_passage_once(tmp_path):
    _built(tmp_path)

    hits = DenseIndex.load("toy", indexes_dir=tmp_path, embedder=KeywordEmbedder()).search("music", k=20)

    assert sorted(hit["pid"] for hit in hits) == sorted(row["pid"] for row in CORPUS)
    assert [hit["score"] for hit in hits] == sorted((hit["score"] for hit in hits), reverse=True)


def test_injected_faiss_index_is_used_for_search(tmp_path):
    _built(tmp_path)

    class FixedFaiss:
        def search(self, queries, k):
            return np.array([[0.9, 0.5]], dtype=np.float32), np.array([[3, 1]])

    index = DenseIndex.load("toy", indexes_dir=tmp_path, embedder=KeywordEmbedder(), faiss_index=FixedFaiss())

    hits = index.search("anything", k=2)

    assert [(hit["pid"], hit["score"]) for hit in hits] == [("p-mozart", pytest.approx(0.9)), ("p-louvre", 0.5)]


def test_load_rejects_row_count_mismatch(tmp_path):
    index_dir = _built(tmp_path)
    np.save(index_dir / "embeddings.npy", np.zeros((4, len(VOCAB)), dtype=np.float32))

    with pytest.raises(ValueError, match="4.*5|5.*4"):
        DenseIndex.load("toy", indexes_dir=tmp_path, embedder=KeywordEmbedder())
