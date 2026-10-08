"""laya_decision_client：以假 laya agent 驗證 state 組裝、三題一次 predict、欄位轉換與值域；model_hash 可由目錄計算。"""

from __future__ import annotations

import pytest

from multihop_benchmark.agent.protocols import DecisionModel
from multihop_benchmark.laya_decision import decision_state_builder, laya_decision_client
from multihop_benchmark.laya_decision.laya_decision_client import LayaDecisionClient, model_hash
from multihop_benchmark.laya_decision.weak_label_generator import DECISION_QUESTIONS


class WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class FakeAgent:
    """模仿 `laya.Agent.predict` 的輸出形狀（見 laya/agent.py `_decode_answers`）。"""

    def __init__(self, noul=0.83, choice="B", score_probs=(0.1, 0.7, 0.2)):
        self.noul, self.choice, self.score_probs = noul, choice, score_probs
        self.calls = []

    def predict(self, state, questions):
        self.calls.append((state, questions))
        probs = {str(i): p for i, p in enumerate(self.score_probs)}
        return {"answers": {
            "sufficient": {"type": "noul", "noul": self.noul, "confidence": max(self.noul, 1 - self.noul)},
            "next_action": {"type": "choice", "choice": self.choice,
                            "probabilities": {"A": 0.1, "B": 0.6, "C": 0.3}},
            "remaining_hops": {"type": "score", "score": sum(i * p for i, p in enumerate(self.score_probs)),
                               "probabilities": probs},
        }}


STATE = {
    "question": "Which band did the singer of Song X join?",
    "sub_questions": ["Who sang Song X?"],
    "evidence": [{"pid": "p1", "title": "Song X", "text": "Song X was sung by Ann. It charted in 1999. Extra."}],
}


def test_decide_builds_state_with_decision_state_builder_and_asks_the_three_fixed_questions():
    agent = FakeAgent()
    client = LayaDecisionClient("unused", agent=agent, tokenizer=WordTokenizer())

    client.decide(STATE)

    (state_text, questions), = agent.calls
    assert state_text == decision_state_builder.build_state(
        STATE["question"], STATE["sub_questions"], STATE["evidence"], tokenizer=WordTokenizer())
    assert questions == DECISION_QUESTIONS


def test_decide_returns_the_four_fields_with_contract_value_ranges():
    client = LayaDecisionClient("unused", agent=FakeAgent(noul=0.83, choice="B", score_probs=(0.1, 0.7, 0.2)),
                                tokenizer=WordTokenizer())

    decision = client.decide(STATE)

    assert set(decision) == {"sufficient_p", "next_action", "remaining", "latency_ms"}
    assert decision["sufficient_p"] == pytest.approx(0.83)
    assert decision["next_action"] == "B"
    assert decision["remaining"] == 1 and isinstance(decision["remaining"], int)
    assert isinstance(decision["latency_ms"], float) and decision["latency_ms"] >= 0.0
    assert isinstance(client, DecisionModel)


def test_remaining_is_the_argmax_of_the_score_distribution_not_the_rounded_expectation():
    # 期望值 0*0.45 + 1*0.05 + 2*0.5 = 1.05 → 四捨五入為 1，但 argmax 為 2。
    client = LayaDecisionClient("unused", agent=FakeAgent(score_probs=(0.45, 0.05, 0.5)), tokenizer=WordTokenizer())

    assert client.decide(STATE)["remaining"] == 2


def test_sufficient_p_is_clamped_into_unit_interval():
    client = LayaDecisionClient("unused", agent=FakeAgent(noul=1.00004), tokenizer=WordTokenizer())

    assert client.decide(STATE)["sufficient_p"] == 1.0


def test_unknown_choice_label_is_rejected():
    client = LayaDecisionClient("unused", agent=FakeAgent(choice="D"), tokenizer=WordTokenizer())

    with pytest.raises(ValueError, match="next_action"):
        client.decide(STATE)


def test_empty_state_is_accepted():
    agent = FakeAgent()
    client = LayaDecisionClient("unused", agent=agent, tokenizer=WordTokenizer())

    client.decide({"question": "Q?", "sub_questions": [], "evidence": []})

    assert "(none yet)" in agent.calls[0][0]


def test_constructor_loads_model_dir_with_laya_load_on_requested_device(monkeypatch):
    loaded = []
    monkeypatch.setattr(laya_decision_client, "load_agent", lambda path, device: loaded.append((path, device)) or FakeAgent())

    client = LayaDecisionClient("/models/laya_multihop", device="cpu", tokenizer=WordTokenizer())

    assert loaded == [("/models/laya_multihop", "cpu")]
    assert client.decide(STATE)["next_action"] == "B"


def _write_model_dir(root, weights=b"w1"):
    (root / "encoder").mkdir(parents=True)
    (root / "tokenizer").mkdir()
    (root / "model.safetensors").write_bytes(weights)
    (root / "rl_agent_config.json").write_text('{"temperature": [1.2, 1.0, 0.9]}', encoding="utf-8")
    (root / "encoder" / "config.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    return root


def test_model_hash_is_a_stable_sha256_of_the_directory_contents(tmp_path):
    a = _write_model_dir(tmp_path / "a")
    b = _write_model_dir(tmp_path / "b")

    digest = model_hash(a)

    assert len(digest) == 64 and int(digest, 16) >= 0
    assert digest == model_hash(a) == model_hash(b)  # 同內容、不同位置 → 同雜湊


def test_model_hash_changes_when_any_file_or_file_name_changes(tmp_path):
    base = model_hash(_write_model_dir(tmp_path / "a"))
    other_weights = model_hash(_write_model_dir(tmp_path / "b", weights=b"w2"))
    renamed = _write_model_dir(tmp_path / "c")
    (renamed / "encoder" / "config.json").rename(renamed / "encoder" / "config2.json")

    assert len({base, other_weights, model_hash(renamed)}) == 3


def test_model_hash_rejects_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        model_hash(tmp_path / "missing")


@pytest.mark.parametrize(
    "score_probs",
    [
        {"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.9},  # 多一個等級
        {"0": 0.5, "1": 0.5},  # 少一個等級
        {"0": 0.1, "1": 0.2, "two": 0.7},  # 鍵名不符
        {"0": 0.1, "1": float("nan"), "2": 0.2},  # 非有限數
        {"0": 0.1, "1": "high", "2": 0.2},  # 非數字
    ],
)
def test_malformed_remaining_hops_distribution_is_rejected_with_the_field_name(score_probs):
    agent = FakeAgent()
    agent.predict = lambda state, questions: {"answers": {
        "sufficient": {"noul": 0.5}, "next_action": {"choice": "A"},
        "remaining_hops": {"probabilities": score_probs},
    }}
    client = LayaDecisionClient("unused", agent=agent, tokenizer=WordTokenizer())

    with pytest.raises(ValueError, match="remaining_hops"):
        client.decide(STATE)
