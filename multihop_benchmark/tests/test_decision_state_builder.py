"""decision_state_builder：唯一 state 組裝點（每段前 2 句、token 上限、確定性）。"""

import pytest

from multihop_benchmark.laya_decision.decision_state_builder import MAX_STATE_TOKENS, build_state


class WhitespaceTokenizer:
    """假 tokenizer：以空白切詞，一詞一 token（免下載 Laya tokenizer）。"""

    def encode(self, text, add_special_tokens=False):
        return text.split()


TOK = WhitespaceTokenizer()


def count(text):
    return len(TOK.encode(text))


def test_question_only_state_contains_question_and_marks_empty_parts():
    state = build_state("Who directed Jaws?", [], [], tokenizer=TOK)

    assert "Who directed Jaws?" in state
    assert state.count("(none yet)") == 2


def test_evidence_keeps_only_first_two_sentences_of_each_passage():
    evidence = [{"pid": "p1", "title": "Jaws", "text": "Jaws is a film. It was released in 1975. Spielberg directed it."}]

    state = build_state("Who directed Jaws?", ["Who directed Jaws?"], evidence, tokenizer=TOK)

    assert "Jaws: Jaws is a film. It was released in 1975." in state
    assert "Spielberg" not in state


def test_accepts_plain_string_evidence_and_sub_questions_in_order():
    state = build_state("Q?", ["first sub?", "second sub?"], ["Fact one. Fact two. Fact three."], tokenizer=TOK)

    assert state.index("first sub?") < state.index("second sub?")
    assert "Fact one. Fact two." in state
    assert "Fact three." not in state


def test_long_state_is_truncated_to_token_limit_keeping_every_passage():
    long_text = " ".join(f"word{i}" for i in range(400)) + "."
    evidence = [{"pid": f"p{i}", "title": f"Title{i}", "text": long_text} for i in range(4)]

    state = build_state("Which title is right?", ["sub one?"], evidence, tokenizer=TOK)

    assert count(state) <= MAX_STATE_TOKENS
    assert "Which title is right?" in state
    for i in range(4):
        assert f"[{i + 1}] Title{i}:" in state


def test_custom_limit_is_respected_and_question_is_kept_first():
    evidence = ["alpha " * 50 + ".", "beta " * 50 + "."]

    state = build_state("Short question?", [], evidence, tokenizer=TOK, max_tokens=40)

    assert count(state) <= 40
    assert state.startswith("Question: Short question?")


def test_question_longer_than_limit_is_cut_to_limit():
    question = " ".join(f"q{i}" for i in range(100))

    state = build_state(question, [], [], tokenizer=TOK, max_tokens=30)

    assert count(state) <= 30
    assert state.startswith("Question: q0 q1")


def test_same_input_gives_identical_output():
    long_text = " ".join(f"w{i}" for i in range(500))
    args = ("Q?", ["s1?"], [{"pid": "a", "title": "A", "text": long_text}, "B text. More."])

    assert build_state(*args, tokenizer=TOK) == build_state(*args, tokenizer=TOK)
    assert build_state(*args, tokenizer=TOK, max_tokens=50) == build_state(*args, tokenizer=TOK, max_tokens=50)


def test_rejects_non_positive_limit():
    with pytest.raises(ValueError):
        build_state("Q?", [], [], tokenizer=TOK, max_tokens=0)
