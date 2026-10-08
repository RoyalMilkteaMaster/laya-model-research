"""weak_label_generator：四種狀態的弱標籤規則、兩種 evidence 寫法、MuSiQue null 退回、輸出確定性。"""

import hashlib
import json
import random

import pytest

from multihop_benchmark.laya_decision import weak_label_generator as wlg

STATE_KINDS = ("question_only", "partial_gold", "full_gold", "distractor_only")


class WhitespaceTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


TOK = WhitespaceTokenizer()


def paragraph(pid, title, text, supporting, sentences):
    return {"pid": pid, "title": title, "text": text, "is_supporting": supporting, "supporting_sentences": sentences}


def hotpot_question(qid="h1"):
    # 黃金段 GoldB 的支持句是第 3 句：整段寫法（前 2 句）看不到它，只有支持句寫法看得到。
    return {
        "id": qid, "question": f"Which film did the director of {qid} make?", "answer": "X", "answer_aliases": [],
        "paragraphs": [
            paragraph(f"{qid}-ga", "GoldA", "GoldA was directed by Ann. Ann is a director. Ann likes tea.", True,
                      ["GoldA was directed by Ann."]),
            paragraph(f"{qid}-gb", "GoldB", "Ann was born in Oslo. She moved to Rome. Ann directed Film X.", True,
                      ["Ann directed Film X."]),
            paragraph(f"{qid}-d1", "Noise1", "Noise1 is a town. It has a river.", False, []),
            paragraph(f"{qid}-d2", "Noise2", "Noise2 is a song. It was a hit.", False, []),
            paragraph(f"{qid}-d3", "Noise3", "Noise3 is a band. They broke up.", False, []),
        ],
    }


def musique_question(qid="m1", gold=4):
    paragraphs = [paragraph(f"{qid}-g{i}", f"MGold{i}", f"MGold{i} fact one. MGold{i} fact two. Extra.", True, None)
                  for i in range(gold)]
    paragraphs += [paragraph(f"{qid}-d{i}", f"MNoise{i}", f"MNoise{i} is noise. More noise.", False, None)
                   for i in range(4)]
    return {"id": qid, "question": f"Multi-hop question {qid}?", "answer": "Y", "answer_aliases": [],
            "paragraphs": paragraphs}


def expected_labels(gold_total, gold_present, distractors):
    """Spec 規則的獨立寫法：sufficient = 黃金全到齊；全到齊→A、只有干擾→C、其餘缺黃金→B；remaining = min(缺, 2)。"""
    if gold_present == gold_total:
        action = "A"
    elif gold_present == 0 and distractors > 0:
        action = "C"
    else:
        action = "B"
    return gold_present == gold_total, action, min(gold_total - gold_present, 2)


def check_record(record, sufficient, action, remaining):
    assert set(record) == {"state", "type", "instructions", "criteria", "target", "label"}
    if record["type"] == "noul":
        assert record["label"] == ("true" if sufficient else "false")
        assert record["target"] == ([0.0, 1.0] if sufficient else [1.0, 0.0])
        assert set(record["criteria"]) == {"false", "true"}
    elif record["type"] == "choice":
        assert record["label"] == action
        assert record["target"] == [1.0 if k == action else 0.0 for k in "ABC"]
        assert list(record["criteria"]) == ["A", "B", "C"]
    else:
        assert record["type"] == "score"
        assert record["label"] == remaining
        assert record["target"] == [1.0 if i == remaining else 0.0 for i in range(3)]
        assert len(record["criteria"]) == 3


@pytest.mark.parametrize("kind", STATE_KINDS)
def test_each_state_kind_has_expected_composition_and_labels(kind):
    row = hotpot_question()
    gold_titles = {"GoldA", "GoldB"}
    for seed in range(20):
        decisions = wlg.decisions_for_state(row, kind, random.Random(seed), tokenizer=TOK)
        assert decisions, kind
        for d in decisions:
            titles = [line.split("] ", 1)[1].split(":", 1)[0] for line in d.record["state"].splitlines()
                      if line.startswith("[")]
            present = len(gold_titles & set(titles))
            distractors = len([t for t in titles if t not in gold_titles])
            assert d.kind == kind and d.gold_total == 2 and d.gold_present == present
            if kind == "question_only":
                assert titles == [] and d.writing == "none"
            elif kind == "partial_gold":
                assert present == 1 and 0 <= distractors <= 2
            elif kind == "full_gold":
                assert present == 2 and 0 <= distractors <= 2
            else:
                assert present == 0 and distractors >= 1
            check_record(d.record, *expected_labels(2, present, distractors))


def test_rules_for_four_gold_question_cap_remaining_at_two():
    row = musique_question(gold=4)
    seen = set()
    for seed in range(40):
        for d in wlg.decisions_for_state(row, "partial_gold", random.Random(seed), tokenizer=TOK):
            assert 1 <= d.gold_present <= 3
            check_record(d.record, *expected_labels(4, d.gold_present, 0 if d.gold_present else 1))
            if d.record["type"] == "score":
                seen.add(d.record["label"])
    assert seen == {1, 2}


def test_evidence_states_come_in_both_writings_with_same_evidence_and_labels():
    row = hotpot_question()
    decisions = wlg.decisions_for_state(row, "full_gold", random.Random(0), tokenizer=TOK)
    by_writing = {}
    for d in decisions:
        by_writing.setdefault(d.writing, []).append(d.record)
    assert set(by_writing) == {"full_paragraph", "supporting_only"}
    assert [r["label"] for r in by_writing["full_paragraph"]] == [r["label"] for r in by_writing["supporting_only"]]
    full_state = by_writing["full_paragraph"][0]["state"]
    support_state = by_writing["supporting_only"][0]["state"]
    # 整段寫法只有前 2 句；只有支持句寫法用支持句原文。
    assert "GoldB: Ann was born in Oslo. She moved to Rome." in full_state
    assert "Ann directed Film X." not in full_state
    assert "GoldB: Ann directed Film X." in support_state
    assert "Oslo" not in support_state


def test_musique_null_supporting_sentences_fall_back_to_whole_paragraph():
    row = musique_question()
    decisions = wlg.decisions_for_state(row, "full_gold", random.Random(3), tokenizer=TOK)
    full = [d for d in decisions if d.writing == "full_paragraph"]
    support = [d for d in decisions if d.writing == "supporting_only"]
    assert [d.record for d in full] == [d.record for d in support]
    assert all(d.fallback for d in support) and not any(d.fallback for d in full)


def test_question_decisions_are_deterministic_for_seed_and_cover_all_kinds_across_questions():
    rows = [hotpot_question(f"h{i}") for i in range(30)]
    first = [d.record for r in rows for d in wlg.question_decisions(r, dataset="hotpotqa", seed=42, tokenizer=TOK)]
    again = [d.record for r in rows for d in wlg.question_decisions(r, dataset="hotpotqa", seed=42, tokenizer=TOK)]
    other = [d.record for r in rows for d in wlg.question_decisions(r, dataset="hotpotqa", seed=7, tokenizer=TOK)]
    assert first == again
    assert first != other
    kinds = {d.kind for r in rows for d in wlg.question_decisions(r, dataset="hotpotqa", seed=42, tokenizer=TOK)}
    assert kinds == set(STATE_KINDS)


def write_split_files(data_root, datasets):
    for name, rows in datasets.items():
        out = data_root / "datasets" / name
        out.mkdir(parents=True)
        for split, offset in (("train_800", 0), ("calibration_100", 100), ("test_100", 200)):
            with (out / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows(offset):
                    handle.write(json.dumps(row) + "\n")


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / "laya_data"
    write_split_files(root, {
        "hotpotqa": lambda o: [hotpot_question(f"h{o + i}") for i in range(12)],
        "2wiki": lambda o: [hotpot_question(f"w{o + i}") for i in range(4)],
        "musique": lambda o: [musique_question(f"m{o + i}", gold=2 + i % 3) for i in range(8)],
    })
    return root


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_make_laya_data_writes_three_files_with_stats_and_is_byte_identical_on_rerun(data_root):
    summary = wlg.make_laya_data(data_root, seed=42, tokenizer=TOK)
    out = data_root / "laya_decisions"
    digests = {name: sha256(out / f"{name}.jsonl") for name in ("train", "calibration", "test")}

    for split in ("train", "calibration", "test"):
        rows = [json.loads(line) for line in (out / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(rows) == summary["splits"][split]["records"] > 0
        assert max(len(TOK.encode(r["state"])) for r in rows) == summary["splits"][split]["max_state_tokens"]
    train_stats = summary["splits"]["train"]["counts"]
    assert {k for k, _, _ in train_stats} == set(STATE_KINDS)
    assert {t for _, t, _ in train_stats} == {"noul", "choice", "score"}
    assert {w for _, _, w in train_stats} == {"none", "full_paragraph", "supporting_only"}
    assert summary["splits"]["train"]["fallback_records"] > 0  # MuSiQue null 退回整段

    wlg.make_laya_data(data_root, seed=42, tokenizer=TOK)
    assert {name: sha256(out / f"{name}.jsonl") for name in digests} == digests


def test_make_laya_data_fails_when_a_split_file_is_missing(data_root):
    (data_root / "datasets" / "2wiki" / "test_100.jsonl").unlink()
    with pytest.raises(FileNotFoundError):
        wlg.make_laya_data(data_root, seed=42, tokenizer=TOK)


def test_audit_confirms_generated_labels_and_flags_a_tampered_one(data_root):
    wlg.make_laya_data(data_root, seed=42, tokenizer=TOK)
    path = data_root / "laya_decisions" / "train.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    questions = wlg.load_split_questions(data_root, "train")

    result = wlg.audit_labels(records, questions, sample_size=100, seed=0)
    assert result["checked"] == 100 and result["mismatches"] == []

    choice = next(i for i, r in enumerate(records) if r["type"] == "choice" and r["label"] == "A")
    records[choice] = {**records[choice], "label": "B", "target": [0.0, 1.0, 0.0]}
    result = wlg.audit_labels(records, questions, sample_size=len(records), seed=0)
    assert [m["index"] for m in result["mismatches"]] == [choice]


def test_audit_resolves_duplicate_title_distractors_with_same_opening():
    # MuSiQue 實例：同一題有兩段標題與開頭相同的干擾段，兩者都不是黃金，不影響標籤。
    row = musique_question("dup", gold=2)
    twin = paragraph("dup-twin", "MNoise0", "MNoise0 is noise. More noise. A third sentence differs.", False, None)
    row["paragraphs"].append(twin)
    records = [d.record for seed in range(30)
               for d in wlg.decisions_for_state(row, "distractor_only", random.Random(seed), tokenizer=TOK)]
    assert any("MNoise0:" in r["state"] for r in records)

    result = wlg.audit_labels(records, {"musique": [row]}, sample_size=len(records), seed=0)

    assert result["mismatches"] == []


def test_audit_separates_gold_and_distractor_sharing_title_and_long_opening():
    # MuSiQue 實例：黃金段與干擾段同標題、開頭 30 字以上相同，後文不同。
    row = musique_question("same", gold=2)
    opening = "MGold0 fact one is a long shared opening sentence"
    row["paragraphs"][0]["text"] = f"{opening} for the gold paragraph. Gold second."
    row["paragraphs"].append(paragraph("same-twin", "MGold0", f"{opening} about something else. Other.", False, None))
    records = [d.record for seed in range(30) for kind in ("full_gold", "partial_gold", "distractor_only")
               for d in wlg.decisions_for_state(row, kind, random.Random(seed), tokenizer=TOK)]

    result = wlg.audit_labels(records, {"musique": [row]}, sample_size=len(records), seed=0)

    assert result["mismatches"] == []
