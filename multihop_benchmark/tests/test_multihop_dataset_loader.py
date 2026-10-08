import hashlib
import json
from datetime import datetime
from pathlib import Path

import pytest

from multihop_benchmark.datasets.multihop_dataset_loader import (
    DATASET_SPECS,
    MANIFEST_NAME,
    SCHEMA_VERSION,
    SEED,
    TRAIN_SPLIT_SIZES,
    OUTPUT_NAMES,
    DatasetDownloadError,
    convert_question,
    load_raw_dataset,
    output_problem,
    paragraph_id,
    prepare_dataset,
    sample_ids,
    split_train_ids,
)

RAW = json.loads((Path(__file__).parent / "fixtures" / "prepare_data" / "raw_samples.json").read_text("utf-8"))


def raw(dataset, question_id):
    return next(row for row in RAW[dataset] if row["id"] == question_id)


def assert_unified_schema(record):
    assert list(record) == ["id", "question", "answer", "answer_aliases", "paragraphs"]
    assert isinstance(record["answer_aliases"], list)
    for paragraph in record["paragraphs"]:
        assert list(paragraph) == ["pid", "title", "text", "is_supporting", "supporting_sentences"]
        assert isinstance(paragraph["is_supporting"], bool)


def sentences_by_title(record):
    return {p["title"]: p["supporting_sentences"] for p in record["paragraphs"]}


def test_dataset_specs_cover_three_datasets():
    assert {name: (spec.repo, spec.config) for name, spec in DATASET_SPECS.items()} == {
        "hotpotqa": ("hotpotqa/hotpot_qa", "distractor"),
        "2wiki": ("framolfese/2WikiMultihopQA", "default"),
        "musique": ("dgslibisey/MuSiQue", "default"),
    }


def test_paragraph_id_is_sha256_prefix_of_title_and_text():
    expected = hashlib.sha256("Tim Burton\nTim Burton is an American filmmaker.".encode()).hexdigest()[:16]
    assert paragraph_id("Tim Burton", "Tim Burton is an American filmmaker.") == expected
    assert paragraph_id("Tim Burton", "other text") != expected


def test_hotpotqa_supporting_paragraphs_follow_supporting_fact_titles():
    record = convert_question("hotpotqa", raw("hotpotqa", "hp-1"))

    assert_unified_schema(record)
    assert record["id"] == "hp-1"
    assert record["answer"] == "yes"
    assert record["answer_aliases"] == []
    assert [(p["title"], p["is_supporting"]) for p in record["paragraphs"]] == [
        ("Ed Wood (film)", False),
        ("Scott Derrickson", True),
        ("Ed Wood", True),
    ]
    # HotpotQA 句子自帶前導空白，接起來不得出現雙空白
    assert record["paragraphs"][1]["text"] == (
        "Scott Derrickson (born July 16, 1966) is an American director. He lives in Los Angeles, California."
    )
    assert sentences_by_title(record) == {
        "Ed Wood (film)": [],
        "Scott Derrickson": ["Scott Derrickson (born July 16, 1966) is an American director."],
        "Ed Wood": ["Edward Davis Wood Jr. was an American filmmaker."],
    }


def test_supporting_sentences_follow_original_sentence_order_without_edge_whitespace():
    record = convert_question("hotpotqa", raw("hotpotqa", "hp-2"))

    # supporting_facts 列為 sent_id [1, 0]，輸出依原始句序；HotpotQA 句首空白去除
    assert sentences_by_title(record) == {
        "Tim Burton": ["Tim Burton is an American filmmaker.", "He was born in Burbank, California."],
        "Ed Wood (film)": ["It was directed by Tim Burton."],
        "Woodson, Arkansas": [],
    }


def test_out_of_range_sent_id_is_skipped_and_logged_as_warning():
    logger = RecordingLogger()

    record = convert_question("hotpotqa", raw("hotpotqa", "hp-0"), logger=logger)

    assert sentences_by_title(record)["Marilyn Manson (band)"] == ["Marilyn Manson is an American rock band."]
    warnings = [(level, event, fields) for level, event, fields in logger.events]
    assert warnings == [
        (
            "WARNING",
            "supporting_sentence_out_of_range",
            {"dataset": "hotpotqa", "question_id": "hp-0", "title": "Marilyn Manson (band)", "sent_id": 5, "sentence_count": 1},
        )
    ]


def test_2wiki_joins_sentences_with_space_and_dedups_repeated_paragraphs():
    record = convert_question("2wiki", raw("2wiki", "2w-1"))

    assert_unified_schema(record)
    assert record["answer_aliases"] == []
    assert [(p["title"], p["is_supporting"]) for p in record["paragraphs"]] == [
        ("Maheen Khan", False),
        ("Polish-Russian War (film)", True),
        ("Xawery Żuławski", True),
    ]
    assert record["paragraphs"][1]["text"] == (
        "Polish-Russian War (Wojna polsko-ruska) is a 2009 Polish film directed by Xawery Żuławski."
    )
    assert len({p["pid"] for p in record["paragraphs"]}) == len(record["paragraphs"])
    assert sentences_by_title(record) == {
        "Maheen Khan": [],
        "Polish-Russian War (film)": ["(Wojna polsko-ruska) is a 2009 Polish film directed by Xawery Żuławski."],
        "Xawery Żuławski": ["He is the son of Andrzej Żuławski and Małgorzata Braunek."],
    }


def test_musique_uses_is_supporting_and_keeps_aliases():
    with_aliases = convert_question("musique", raw("musique", "2hop__1_2"))
    without_aliases = convert_question("musique", raw("musique", "3hop1__3_4_5"))

    assert_unified_schema(with_aliases)
    assert with_aliases["answer"] == "Blue Note"
    assert with_aliases["answer_aliases"] == ["Blue Note Records", "BN"]
    assert [p["is_supporting"] for p in with_aliases["paragraphs"]] == [False, True, True]
    assert with_aliases["paragraphs"][1] == {
        "pid": paragraph_id("Grant Green", "Grant Green was an American jazz guitarist."),
        "title": "Grant Green",
        "text": "Grant Green was an American jazz guitarist.",
        "is_supporting": True,
        "supporting_sentences": None,
    }
    assert all(p["supporting_sentences"] is None for p in with_aliases["paragraphs"] + without_aliases["paragraphs"])
    assert without_aliases["answer_aliases"] == []
    assert sum(p["is_supporting"] for p in without_aliases["paragraphs"]) == 3


def test_same_paragraph_gets_same_pid_across_questions():
    first = convert_question("hotpotqa", raw("hotpotqa", "hp-1"))
    second = convert_question("hotpotqa", raw("hotpotqa", "hp-2"))

    pid_of = lambda record, title: next(p["pid"] for p in record["paragraphs"] if p["title"] == title)
    assert pid_of(first, "Ed Wood (film)") == pid_of(second, "Ed Wood (film)")


def test_unknown_dataset_is_rejected():
    with pytest.raises(KeyError):
        convert_question("squad", raw("hotpotqa", "hp-1"))


IDS = [f"q{i:04d}" for i in range(1500)]


def test_seed_and_split_sizes_follow_spec():
    assert SEED == 42
    assert TRAIN_SPLIT_SIZES == {"train_800": 800, "calibration_100": 100, "test_100": 100}


def test_sample_ids_is_deterministic_and_independent_of_input_order():
    first = sample_ids(IDS, 200)

    assert len(first) == len(set(first)) == 200
    assert set(first) <= set(IDS)
    assert sample_ids(list(reversed(IDS)), 200) == first
    assert sample_ids(IDS, 200, seed=7) != first


def test_sample_ids_rejects_too_small_pool():
    with pytest.raises(ValueError):
        sample_ids(IDS[:10], 11)


def test_split_train_ids_gives_disjoint_splits_of_requested_sizes():
    splits = split_train_ids(list(reversed(IDS)))

    assert list(splits) == ["train_800", "calibration_100", "test_100"]
    assert {name: len(ids) for name, ids in splits.items()} == TRAIN_SPLIT_SIZES
    all_ids = [question_id for ids in splits.values() for question_id in ids]
    assert len(set(all_ids)) == 1000
    assert split_train_ids(IDS) == splits


class FakeBuilder:
    """模擬 datasets 的 DatasetBuilder：cache_dir 內有 dataset_info.json 即視為已快取。"""

    def __init__(self, cache_dir, failures):
        self.cache_dir = str(cache_dir)
        self.failures = failures
        self.prepared = 0

    def download_and_prepare(self):
        if self.failures:
            self.failures.pop(0)
            raise ConnectionError("network down")
        self.prepared += 1
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        (Path(self.cache_dir) / "dataset_info.json").write_text("{}", encoding="utf-8")

    def as_dataset(self):
        return {"train": ["t"], "validation": ["v"]}


class RecordingLogger:
    def __init__(self):
        self.events = []

    def _record(self, level, message, event=None, **fields):
        self.events.append((level, event, fields))

    def info(self, message, **fields):
        self._record("INFO", message, **fields)

    def warning(self, message, **fields):
        self._record("WARNING", message, **fields)

    def error(self, message, **fields):
        self._record("ERROR", message, **fields)


def make_factory(builder, calls):
    def factory(repo, config):
        calls.append((repo, config))
        return builder

    return factory


def test_load_raw_dataset_retries_with_backoff_then_succeeds(tmp_path):
    builder = FakeBuilder(tmp_path / "cache", failures=[1, 1])
    calls, sleeps, logger = [], [], RecordingLogger()

    splits = load_raw_dataset(
        "musique", logger=logger, builder_factory=make_factory(builder, calls), sleep=sleeps.append
    )

    assert splits == {"train": ["t"], "validation": ["v"]}
    assert calls == [("dgslibisey/MuSiQue", "default")] * 3
    assert len(sleeps) == 2 and sleeps[0] < sleeps[1]
    events = [event for _, event, _ in logger.events]
    assert events.count("dataset_download_retry") == 2
    assert "dataset_download_started" in events and "dataset_loaded" in events


def test_load_raw_dataset_logs_cache_hit_when_already_prepared(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "dataset_info.json").write_text("{}", encoding="utf-8")
    logger = RecordingLogger()

    load_raw_dataset("hotpotqa", logger=logger, builder_factory=make_factory(FakeBuilder(cache_dir, []), []))

    hits = [fields for _, event, fields in logger.events if event == "dataset_cache_hit"]
    assert hits and hits[0]["cache_dir"] == str(cache_dir)
    assert "dataset_download_started" not in [event for _, event, _ in logger.events]


def test_load_raw_dataset_raises_after_last_attempt(tmp_path):
    builder = FakeBuilder(tmp_path / "cache", failures=[1, 1, 1])
    logger = RecordingLogger()

    with pytest.raises(DatasetDownloadError):
        load_raw_dataset("2wiki", logger=logger, builder_factory=make_factory(builder, []), max_attempts=3, sleep=lambda s: None)

    failed = [(level, fields) for level, event, fields in logger.events if event == "dataset_download_failed"]
    assert failed and failed[0][0] == "ERROR" and failed[0][1]["errorCode"] == "DATASET_DOWNLOAD_FAILED"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


def hotpot_splits():
    from datasets import Dataset

    rows = RAW["hotpotqa"]
    renamed = lambda suffix: [{**row, "id": row["id"] + suffix} for row in rows]
    return {"train": Dataset.from_list(renamed("-t1") + renamed("-t2")), "validation": Dataset.from_list(rows)}


SMALL = {"sample_size": 2, "train_sizes": {"train_800": 3, "calibration_100": 1, "test_100": 1}}


def test_prepare_dataset_writes_five_files_that_match_conversion_and_sampling(tmp_path):
    splits = hotpot_splits()

    counts = prepare_dataset("hotpotqa", splits, tmp_path, **SMALL)

    assert OUTPUT_NAMES == ("sample_200", "train_800", "calibration_100", "test_100", "corpus")
    assert output_problem(tmp_path, **SMALL) is None
    sample = read_jsonl(tmp_path / "sample_200.jsonl")
    assert [q["id"] for q in sample] == sample_ids(["hp-0", "hp-1", "hp-2"], 2)
    assert sample == [convert_question("hotpotqa", raw("hotpotqa", q["id"])) for q in sample]
    train_ids = {name: [q["id"] for q in read_jsonl(tmp_path / f"{name}.jsonl")] for name in SMALL["train_sizes"]}
    assert train_ids == split_train_ids(list(splits["train"]["id"]), SMALL["train_sizes"])
    corpus = read_jsonl(tmp_path / "corpus.jsonl")
    assert len({row["pid"] for row in corpus}) == len(corpus)
    assert {p["pid"] for q in sample for p in q["paragraphs"] if p["is_supporting"]} <= {r["pid"] for r in corpus}
    assert counts == {"sample_200": 2, "train_800": 3, "calibration_100": 1, "test_100": 1, "corpus": len(corpus)}


def test_prepare_dataset_output_is_byte_identical_across_runs(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    prepare_dataset("hotpotqa", hotpot_splits(), first, **SMALL)
    reordered = {name: split.shuffle(seed=1) for name, split in hotpot_splits().items()}
    prepare_dataset("hotpotqa", reordered, second, **SMALL)

    for name in OUTPUT_NAMES:
        assert (first / f"{name}.jsonl").read_bytes() == (second / f"{name}.jsonl").read_bytes()
    assert "\\u" not in (first / "sample_200.jsonl").read_text("utf-8")


def test_prepare_dataset_passes_logger_to_conversion(tmp_path):
    logger = RecordingLogger()

    prepare_dataset("hotpotqa", hotpot_splits(), tmp_path, sample_size=3, train_sizes={"train_800": 1}, logger=logger)

    assert [fields["question_id"] for _, event, fields in logger.events if event == "supporting_sentence_out_of_range"][:1] == ["hp-0"]


def _rewrite(path, transform):
    path.write_text(transform(path.read_text("utf-8")), encoding="utf-8")


def _drop_supporting_sentences(text):
    rows = [json.loads(line) for line in text.splitlines()]
    for row in rows:
        for paragraph in row["paragraphs"]:
            del paragraph["supporting_sentences"]
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


CORRUPTIONS = {
    "missing_file": ("corpus", lambda path: path.unlink()),
    "empty_file": ("sample_200", lambda path: path.write_text("", encoding="utf-8")),
    "truncated_json": ("train_800", lambda path: _rewrite(path, lambda text: text[: len(text) - 40])),
    "line_count": ("test_100", lambda path: _rewrite(path, lambda text: "")),
    "old_schema": ("sample_200", lambda path: _rewrite(path, _drop_supporting_sentences)),
    "corpus_not_matching_sample": ("corpus", lambda path: _rewrite(path, lambda text: "".join(text.splitlines(True)[1:]))),
    "overlapping_train_splits": (
        "test_100",
        lambda path: path.write_text(
            (path.parent / "train_800.jsonl").read_text("utf-8").splitlines(True)[0], encoding="utf-8"
        ),
    ),
}


def test_output_problem_is_none_for_freshly_prepared_outputs(tmp_path):
    prepare_dataset("hotpotqa", hotpot_splits(), tmp_path, **SMALL)

    assert output_problem(tmp_path, **SMALL) is None


@pytest.mark.parametrize("corruption", list(CORRUPTIONS))
def test_output_problem_names_the_damaged_file(tmp_path, corruption):
    prepare_dataset("hotpotqa", hotpot_splits(), tmp_path, **SMALL)
    name, damage = CORRUPTIONS[corruption]
    damage(tmp_path / f"{name}.jsonl")

    problem = output_problem(tmp_path, **SMALL)

    assert problem is not None and f"{name}.jsonl" in problem


def sha256_of(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_manifest(output_dir):
    return json.loads((output_dir / MANIFEST_NAME).read_text("utf-8"))


def resign_manifest(output_dir):
    """模擬偽造：把 manifest 的 sha256 改成與目前檔案相符，用來測第二層結構檢查。"""
    manifest = read_manifest(output_dir)
    for file_name, entry in manifest["files"].items():
        if (output_dir / file_name).is_file():
            entry["sha256"] = sha256_of(output_dir / file_name)
    (output_dir / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")


def test_prepare_dataset_writes_manifest_describing_outputs(tmp_path):
    counts = prepare_dataset("hotpotqa", hotpot_splits(), tmp_path, **SMALL)

    manifest = read_manifest(tmp_path)
    assert MANIFEST_NAME == "prepare_manifest.json" and SCHEMA_VERSION == "2"
    assert manifest["schema_version"] == "2"
    assert manifest["generator"]["module"] == "multihop_benchmark.datasets.multihop_dataset_loader"
    assert manifest["generator"]["version"]
    assert manifest["seed"] == SEED and manifest["dataset"] == "hotpotqa"
    assert manifest["created_at"].endswith("Z")
    datetime.fromisoformat(manifest["created_at"].replace("Z", "+00:00"))
    assert manifest["source"]["repo"] == "hotpotqa/hotpot_qa" and manifest["source"]["config"] == "distractor"
    assert manifest["source"]["splits"] == {"validation": {"rows": 3}, "train": {"rows": 6}}
    assert manifest["files"] == {
        f"{name}.jsonl": {"sha256": sha256_of(tmp_path / f"{name}.jsonl"), "line_count": counts[name]}
        for name in OUTPUT_NAMES
    }


def _edit_first_answer(path):
    lines = path.read_text("utf-8").splitlines(True)
    row = json.loads(lines[0])
    row["answer"] = "__REVIEWER_CORRUPTED_GOLD__"
    lines[0] = json.dumps(row, ensure_ascii=False) + "\n"
    path.write_text("".join(lines), encoding="utf-8")


def test_output_problem_reports_edited_answer(tmp_path):
    prepare_dataset("hotpotqa", hotpot_splits(), tmp_path, **SMALL)
    _edit_first_answer(tmp_path / "sample_200.jsonl")

    problem = output_problem(tmp_path, **SMALL)

    assert problem is not None and "sample_200.jsonl" in problem


def test_output_problem_reports_missing_manifest(tmp_path):
    prepare_dataset("hotpotqa", hotpot_splits(), tmp_path, **SMALL)
    (tmp_path / MANIFEST_NAME).unlink()

    problem = output_problem(tmp_path, **SMALL)

    assert problem is not None and MANIFEST_NAME in problem


def test_output_problem_reports_old_and_new_files_mixed_after_interrupted_force(tmp_path):
    current, other = tmp_path / "current", tmp_path / "other"
    prepare_dataset("hotpotqa", hotpot_splits(), current, **SMALL)
    prepare_dataset("hotpotqa", hotpot_splits(), other, seed=7, **SMALL)
    (current / "train_800.jsonl").write_bytes((other / "train_800.jsonl").read_bytes())

    problem = output_problem(current, **SMALL)

    assert problem is not None and "train_800.jsonl" in problem


def test_output_problem_reports_schema_version_mismatch(tmp_path):
    prepare_dataset("hotpotqa", hotpot_splits(), tmp_path, **SMALL)
    manifest = read_manifest(tmp_path)
    manifest["schema_version"] = "1"
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")

    problem = output_problem(tmp_path, **SMALL)

    assert problem is not None and "schema_version" in problem


def test_failed_rebuild_leaves_no_manifest(tmp_path, monkeypatch):
    from multihop_benchmark.datasets import multihop_dataset_loader as loader

    prepare_dataset("hotpotqa", hotpot_splits(), tmp_path, **SMALL)
    original_write = loader.write_jsonl

    def failing_write(path, rows):
        if path.name == "corpus.jsonl":
            raise OSError("disk full")
        return original_write(path, rows)

    monkeypatch.setattr(loader, "write_jsonl", failing_write)
    with pytest.raises(OSError):
        prepare_dataset("hotpotqa", hotpot_splits(), tmp_path, seed=7, **SMALL)

    assert not (tmp_path / MANIFEST_NAME).exists()
    assert MANIFEST_NAME in output_problem(tmp_path, **SMALL)


@pytest.mark.parametrize("corruption", list(CORRUPTIONS))
def test_structure_check_still_catches_damage_when_manifest_is_resigned(tmp_path, corruption):
    prepare_dataset("hotpotqa", hotpot_splits(), tmp_path, **SMALL)
    name, damage = CORRUPTIONS[corruption]
    damage(tmp_path / f"{name}.jsonl")
    resign_manifest(tmp_path)

    problem = output_problem(tmp_path, **SMALL)

    assert problem is not None and f"{name}.jsonl" in problem
