import json
from pathlib import Path

import numpy as np
import pytest

from multihop_benchmark.cli import main
from multihop_benchmark.retrieval import dense_index, passage_reranker

CORPUS_TEXT = (Path(__file__).parent / "fixtures" / "retrieval" / "corpus_5.jsonl").read_text("utf-8")
CORPUS = [json.loads(line) for line in CORPUS_TEXT.splitlines()]
VOCAB = ("river", "city", "painter", "music")


class KeywordEmbedder:
    def encode(self, texts):
        return np.array([[text.lower().split().count(word) for word in VOCAB] for text in texts], dtype=np.float64)


class TitleMatchCrossEncoder:
    """假 cross-encoder：段落首行（標題）等於查詢時 1 分，否則 0 分。"""

    def predict(self, pairs):
        return [1.0 if chunk.split("\n", 1)[0] == query else 0.0 for query, chunk in pairs]


def _question(qid, supporting_pid):
    return {
        "id": qid, "question": "?", "answer": "a", "answer_aliases": [],
        "paragraphs": [
            {**row, "is_supporting": row["pid"] == supporting_pid, "supporting_sentences": None} for row in CORPUS
        ],
    }


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "laya_data"
    monkeypatch.setenv("PROJECT_DATA_ROOT", str(root))
    monkeypatch.setenv("PROJECT_RUNTIME_ROOT", str(tmp_path / "laya_runtime"))
    for dataset in ("hotpotqa", "2wiki"):
        dataset_dir = root / "datasets" / dataset
        dataset_dir.mkdir(parents=True)
        (dataset_dir / "corpus.jsonl").write_text(CORPUS_TEXT, encoding="utf-8")
        questions = [_question("q1", "p-seine"), _question("q2", "p-mozart"), _question("q3", "p-monet")]
        (dataset_dir / "sample_200.jsonl").write_text(
            "".join(json.dumps(q) + "\n" for q in questions), encoding="utf-8"
        )
    return root


@pytest.fixture
def fake_models(monkeypatch):
    devices = []

    def load_embedder(device="cuda"):
        devices.append(("embedder", device))
        return KeywordEmbedder()

    def load_cross_encoder(device="cuda"):
        devices.append(("reranker", device))
        return TitleMatchCrossEncoder()

    monkeypatch.setattr(dense_index, "load_embedder", load_embedder)
    monkeypatch.setattr(passage_reranker, "load_cross_encoder", load_cross_encoder)
    return devices


def test_build_index_writes_one_vector_per_corpus_row_and_report(data_root, fake_models, capsys):
    exit_code = main(["build-index", "--datasets", "hotpotqa", "2wiki", "--device", "cpu", "--report"])

    assert exit_code == 0
    for dataset in ("hotpotqa", "2wiki"):
        embeddings = np.load(data_root / "indexes" / dataset / "embeddings.npy")
        meta = json.loads((data_root / "indexes" / dataset / "passage_ids.json").read_text("utf-8"))
        assert embeddings.shape[0] == len(meta["passages"]) == len(CORPUS)
        assert meta["device"] == "cpu"
    assert "hotpotqa: vectors=5 corpus_rows=5" in capsys.readouterr().out
    assert set(fake_models) == {("embedder", "cpu"), ("reranker", "cpu")}

    report = json.loads((data_root / "indexes" / "index_report.json").read_text("utf-8"))
    for dataset in ("hotpotqa", "2wiki"):
        entry = report["datasets"][dataset]
        assert entry["vectors"] == 5
        assert entry["n"] == 3
        # 語料只有 5 段 < 20：支持段一定在 dense top-20；假重排把標題相符者排第 1
        assert entry["dense_hit_rate"] == 1.0
        assert entry["rerank_top_n_rate"] == 1.0
        assert entry["device"] == "cpu"

    events = [
        json.loads(line)["event"]
        for line in (data_root / "logs" / "prepare_data.jsonl").read_text("utf-8").splitlines()
    ]
    assert events[0] == "build_index_started" and events[-1] == "build_index_finished"
    assert events.count("index_built") == 2


def test_report_run_keeps_entries_of_datasets_not_rebuilt(data_root, fake_models):
    assert main(["build-index", "--datasets", "hotpotqa", "--device", "cpu", "--report"]) == 0
    assert main(["build-index", "--datasets", "2wiki", "--device", "cpu", "--report"]) == 0

    report = json.loads((data_root / "indexes" / "index_report.json").read_text("utf-8"))
    assert set(report["datasets"]) == {"hotpotqa", "2wiki"}


def test_missing_corpus_fails_that_dataset_but_builds_the_others(data_root, fake_models, capsys):
    exit_code = main(["build-index", "--datasets", "musique", "hotpotqa", "--device", "cpu"])

    assert exit_code == 1
    assert not (data_root / "indexes" / "musique").exists()
    assert (data_root / "indexes" / "hotpotqa" / "embeddings.npy").is_file()
    assert not (data_root / "indexes" / "index_report.json").exists()
    assert "musique: 失敗" in capsys.readouterr().out


def _log_records(data_root):
    return [json.loads(line) for line in (data_root / "logs" / "prepare_data.jsonl").read_text("utf-8").splitlines()]


class FailOnce:
    """包住假模型：第一次呼叫 `method` 拋 RuntimeError（模擬 GPU／推論錯誤），之後正常。"""

    def __init__(self, inner, method):
        self.inner, self.method, self.failed = inner, method, False

    def __getattr__(self, name):
        call = getattr(self.inner, name)
        if name != self.method or self.failed:
            return call

        def boom(*args, **kwargs):
            self.failed = True
            raise RuntimeError(f"synthetic {name} failure")

        return boom


@pytest.mark.parametrize(
    ("patch_target", "method", "stage"),
    [("load_cross_encoder", "predict", "report"), ("load_embedder", "encode", "build")],
)
def test_model_error_fails_only_that_dataset_and_logs_are_complete(
    data_root, fake_models, monkeypatch, capsys, patch_target, method, stage
):
    module = passage_reranker if patch_target == "load_cross_encoder" else dense_index
    original = getattr(module, patch_target)
    monkeypatch.setattr(module, patch_target, lambda device="cuda": FailOnce(original(device), method))

    exit_code = main(["build-index", "--datasets", "hotpotqa", "2wiki", "--device", "cpu", "--report"])

    assert exit_code == 1
    assert "hotpotqa: 失敗" in capsys.readouterr().out
    report = json.loads((data_root / "indexes" / "index_report.json").read_text("utf-8"))
    assert set(report["datasets"]) == {"2wiki"}
    records = _log_records(data_root)
    failure = next(r for r in records if r["event"] == "index_build_failed")
    assert (failure["dataset"], failure["stage"], failure["errorCode"]) == ("hotpotqa", stage, "INDEX_BUILD_FAILED")
    assert f"synthetic {method} failure" in failure["exception"]
    assert [r["dataset"] for r in records if r["event"] == "index_built"] == (
        ["hotpotqa", "2wiki"] if stage == "report" else ["2wiki"]
    )
    assert records[-1]["event"] == "build_index_finished"
    assert records[-1]["failed"] == ["hotpotqa"] and records[-1]["level"] == "ERROR"


def test_model_load_failure_returns_nonzero_and_logs_finish(data_root, monkeypatch, capsys):
    def broken_loader(device="cuda"):
        raise OSError("model files missing")

    monkeypatch.setattr(dense_index, "load_embedder", broken_loader)

    exit_code = main(["build-index", "--datasets", "hotpotqa", "--device", "cpu"])

    assert exit_code == 1
    assert "模型載入失敗" in capsys.readouterr().out
    assert not (data_root / "indexes" / "hotpotqa").exists()
    records = _log_records(data_root)
    assert [r["event"] for r in records] == ["build_index_started", "model_load_failed", "build_index_finished"]
    assert records[1]["errorCode"] == "MODEL_LOAD_FAILED" and "model files missing" in records[1]["exception"]
    assert records[-1]["failed"] == ["hotpotqa"] and records[-1]["level"] == "ERROR"


def test_report_write_failure_returns_nonzero_and_logs_finish(data_root, fake_models):
    (data_root / "indexes" / "index_report.json").mkdir(parents=True)  # 寫檔會丟 IsADirectoryError

    exit_code = main(["build-index", "--datasets", "hotpotqa", "--device", "cpu", "--report"])

    assert exit_code == 1
    assert (data_root / "indexes" / "hotpotqa" / "embeddings.npy").is_file()
    records = _log_records(data_root)
    assert [r["event"] for r in records][-2:] == ["index_report_failed", "build_index_finished"]
    assert records[-2]["errorCode"] == "INDEX_REPORT_FAILED"
    assert records[-1]["level"] == "ERROR" and records[-1]["failed"] == [] and records[-1]["report_error"]
