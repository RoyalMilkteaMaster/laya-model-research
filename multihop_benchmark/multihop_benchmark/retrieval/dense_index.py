"""BGE-M3 dense 嵌入 + faiss-cpu FlatIP 精確搜尋（Spec「RAG 管線」）。

每集一個索引，存於 Data Root `indexes/<dataset>/`：
- `embeddings.npy`：float32、L2 正規化，inner product 即 cosine。
- `passage_ids.json`：`{"model", "device", "count", "passages": [{pid, title, text}]}`，`passages` 與 npy 列一一對齊；
  連同全文一起存，`search` 回傳段落時不再依賴 `datasets/`。

測試接縫：`embedder`（具 `encode(texts) -> 2D array`，不必正規化）與 `faiss_index`（具 `search(x, k) -> (D, I)`）
皆可注入假物件；未注入時才載入 BGE-M3（`sentence-transformers`）與建 `faiss.IndexFlatIP`。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDINGS_NAME = "embeddings.npy"
PASSAGES_NAME = "passage_ids.json"
DEFAULT_DEVICE = "cuda"
ENCODE_BATCH_SIZE = 16


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class _SentenceTransformerEmbedder:
    def __init__(self, device: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self._model.encode(
            list(texts), batch_size=ENCODE_BATCH_SIZE, normalize_embeddings=True, convert_to_numpy=True
        )


def load_embedder(device: str = DEFAULT_DEVICE) -> Embedder:
    """載入 BGE-M3（HF 快取於 `HF_HOME`）。"""
    return _SentenceTransformerEmbedder(device)


def passage_chunk(passage: Mapping[str, Any]) -> str:
    """chunk = 標題＋全文（Spec「資料集」）；重排也用同一字串。"""
    return f"{passage['title']}\n{passage['text']}"


def default_indexes_dir() -> Path:
    from multihop_benchmark.settings import load_settings

    return load_settings().data_root / "indexes"


def _normalized(vectors: Any) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.where(norms == 0, 1, norms)


def build_index(
    dataset: str,
    corpus_rows: Iterable[Mapping[str, Any]],
    device: str = DEFAULT_DEVICE,
    *,
    indexes_dir: Path | None = None,
    embedder: Embedder | None = None,
) -> None:
    """嵌入 `corpus_rows`（`{pid, title, text}`）並寫入 `indexes/<dataset>/`；既有檔案整份覆寫。"""
    passages = [{"pid": str(row["pid"]), "title": str(row["title"]), "text": str(row["text"])} for row in corpus_rows]
    if not passages:
        raise ValueError(f"{dataset}：語料為空，無法建索引")
    embedder = embedder or load_embedder(device)
    embeddings = _normalized(embedder.encode([passage_chunk(p) for p in passages]))
    if embeddings.shape[0] != len(passages):
        raise ValueError(f"{dataset}：嵌入 {embeddings.shape[0]} 列，語料 {len(passages)} 段")

    index_dir = (indexes_dir or default_indexes_dir()) / dataset
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / EMBEDDINGS_NAME, embeddings)
    meta = {"model": EMBEDDING_MODEL, "device": device, "count": len(passages), "passages": passages}
    (index_dir / PASSAGES_NAME).write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


class DenseIndex:
    """已載入的單集索引；`search(query, k)` 回傳 `[{pid, score, title, text}]`，分數由高到低。"""

    def __init__(
        self,
        embeddings: np.ndarray,
        passages: Sequence[Mapping[str, Any]],
        embedder: Embedder,
        faiss_index: Any = None,
    ) -> None:
        if len(embeddings) != len(passages):
            raise ValueError(f"embeddings 有 {len(embeddings)} 列，passage_ids 有 {len(passages)} 段，兩者必須對齊")
        self.passages = [dict(p) for p in passages]
        self._embedder = embedder
        if faiss_index is None:
            import faiss

            faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
            faiss_index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
        self._faiss = faiss_index

    @classmethod
    def load(
        cls,
        dataset: str,
        *,
        indexes_dir: Path | None = None,
        device: str = DEFAULT_DEVICE,
        embedder: Embedder | None = None,
        faiss_index: Any = None,
    ) -> DenseIndex:
        index_dir = (indexes_dir or default_indexes_dir()) / dataset
        embeddings = np.load(index_dir / EMBEDDINGS_NAME)
        meta = json.loads((index_dir / PASSAGES_NAME).read_text("utf-8"))
        passages = meta["passages"]
        if len(embeddings) != len(passages):  # 先檢查，避免對壞掉的索引白白載入模型
            raise ValueError(f"{index_dir}：embeddings 有 {len(embeddings)} 列，passage_ids 有 {len(passages)} 段")
        return cls(embeddings, passages, embedder or load_embedder(device), faiss_index)

    def __len__(self) -> int:
        return len(self.passages)

    def search(self, query: str, k: int) -> list[dict[str, Any]]:
        query_vector = _normalized(self._embedder.encode([query]))
        scores, ids = self._faiss.search(query_vector, min(k, len(self.passages)))
        return [
            {"pid": self.passages[i]["pid"], "score": float(score),
             "title": self.passages[i]["title"], "text": self.passages[i]["text"]}
            for score, i in zip(scores[0], ids[0])
            if i >= 0
        ]
