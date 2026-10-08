"""真實 GPU：載入 Runtime Root `models/laya_multihop`（`train-laya` 輸出），`laya.load()` + `predict` 一筆並經 client 決策一次。

以 `pytest -m integration` 執行；模型目錄不存在時 skip。
"""

from __future__ import annotations

import os

import pytest

from multihop_benchmark.settings import load_settings

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")


@pytest.mark.integration
def test_trained_model_loads_with_laya_and_decides_one_state():
    import laya

    from multihop_benchmark.laya_decision.laya_decision_client import LayaDecisionClient, model_hash
    from multihop_benchmark.laya_decision.weak_label_generator import DECISION_QUESTIONS

    model_dir = load_settings().runtime_root / "models" / "laya_multihop"
    if not (model_dir / "model.safetensors").is_file():
        pytest.skip(f"尚未訓練：{model_dir}")
    assert (model_dir / "rl_agent_config.json").is_file()

    agent = laya.load(str(model_dir), device="cuda")
    raw = agent.predict(state="Question: Who wrote Hamlet?\nSub-questions asked: (none yet)\nEvidence: (none yet)",
                        questions=DECISION_QUESTIONS)
    assert set(raw["answers"]) == set(DECISION_QUESTIONS)

    client = LayaDecisionClient(model_dir, device="cuda", agent=agent)
    decision = client.decide({
        "question": "Which country is the birthplace of the director of film Alpha?",
        "sub_questions": ["Who directed Alpha?"],
        "evidence": [{"pid": "p1", "title": "Alpha (film)", "text": "Alpha is a 2018 film directed by Albert Hughes."}],
    })
    print(decision, model_hash(model_dir))
    assert 0.0 <= decision["sufficient_p"] <= 1.0
    assert decision["next_action"] in {"A", "B", "C"}
    assert decision["remaining"] in {0, 1, 2}
    assert decision["latency_ms"] > 0
