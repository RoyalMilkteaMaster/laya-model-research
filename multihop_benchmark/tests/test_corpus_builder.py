import json
from pathlib import Path

from multihop_benchmark.datasets.corpus_builder import build_corpus
from multihop_benchmark.datasets.multihop_dataset_loader import convert_question

RAW = json.loads((Path(__file__).parent / "fixtures" / "prepare_data" / "raw_samples.json").read_text("utf-8"))


def test_corpus_dedups_by_pid_and_covers_every_supporting_paragraph():
    questions = [convert_question("musique", row) for row in RAW["musique"]]

    corpus = build_corpus(questions)

    pids = [row["pid"] for row in corpus]
    assert len(pids) == len(set(pids))
    # 兩題共用的 "Green" 干擾段落只出現一次：3 + 4 段去掉 1 段重複
    assert len(corpus) == 6
    assert [row["title"] for row in corpus].count("Green") == 1
    supporting = {p["pid"] for q in questions for p in q["paragraphs"] if p["is_supporting"]}
    assert supporting <= set(pids)


def test_corpus_rows_hold_only_pid_title_text_in_first_seen_order():
    questions = [convert_question("hotpotqa", row) for row in RAW["hotpotqa"]]

    corpus = build_corpus(questions)

    assert all(list(row) == ["pid", "title", "text"] for row in corpus)
    first_seen = []
    for question in questions:
        for paragraph in question["paragraphs"]:
            if paragraph["pid"] not in first_seen:
                first_seen.append(paragraph["pid"])
    assert [row["pid"] for row in corpus] == first_seen
    assert len({(row["title"], row["text"]) for row in corpus}) == len(corpus)
