"""laya_evaluator：以假預測驗證分組指標（依 04_test_laya.py 定義，預期值手算）、PASS／FAIL 判定與報告內容。"""

from __future__ import annotations

import json

import pytest

from multihop_benchmark.laya_decision import laya_evaluator
from multihop_benchmark.laya_decision.weak_label_generator import DECISION_QUESTIONS


def _rows(state, sufficient, action, remaining):
    targets = {
        "sufficient": [0.0, 1.0] if sufficient else [1.0, 0.0],
        "next_action": [1.0 if a == action else 0.0 for a in "ABC"],
        "remaining_hops": [1.0 if i == remaining else 0.0 for i in range(3)],
    }
    labels = {"sufficient": "true" if sufficient else "false", "next_action": action, "remaining_hops": remaining}
    return [{"state": state, **DECISION_QUESTIONS[name], "target": targets[name], "label": labels[name]}
            for name in ("sufficient", "next_action", "remaining_hops")]


ROWS = _rows("S1", True, "A", 0) + _rows("S2", False, "B", 2)

# 每個 state 的假預測：P(true)、choice 機率、score 機率。
PREDICTIONS = {
    "S1": (0.8, {"A": 0.7, "B": 0.2, "C": 0.1}, (0.6, 0.3, 0.1)),
    "S2": (0.4, {"A": 0.5, "B": 0.4, "C": 0.1}, (0.1, 0.2, 0.7)),
}


class FakeAgent:
    def __init__(self):
        self.calls = []

    def predict(self, state, questions):
        self.calls.append((state, sorted(questions)))
        p_true, choice, score = PREDICTIONS[state]
        answers = {}
        for key, q in questions.items():
            if q["type"] == "noul":
                answers[key] = {"type": "noul", "noul": p_true}
            elif q["type"] == "choice":
                answers[key] = {"type": "choice", "choice": max(choice, key=choice.get), "probabilities": choice}
            else:
                answers[key] = {"type": "score", "score": sum(i * p for i, p in enumerate(score)),
                                "probabilities": {str(i): p for i, p in enumerate(score)}}
        return {"answers": answers}


def test_rows_sharing_a_state_are_answered_in_one_predict_call_like_the_decision_client():
    agent = FakeAgent()

    result = laya_evaluator.evaluate_agent(agent, ROWS)

    assert agent.calls == [("S1", ["next_action", "remaining_hops", "sufficient"]),
                           ("S2", ["next_action", "remaining_hops", "sufficient"])]
    assert result["decisions"] == 6 and result["predict_calls"] == 2
    assert result["p50_latency_ms"] >= 0.0


def test_sufficient_metrics_match_hand_computed_values():
    m = laya_evaluator.evaluate_agent(FakeAgent(), ROWS)["by_question"]["sufficient"]

    assert m["n"] == 2
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["soft_accuracy"] == pytest.approx((0.8 + 0.6) / 2)
    assert m["brier"] == pytest.approx((0.08 + 0.32) / 2)
    assert m["ece"] == pytest.approx(0.5 * 0.2 + 0.5 * 0.4)


def test_next_action_metrics_match_hand_computed_values():
    m = laya_evaluator.evaluate_agent(FakeAgent(), ROWS)["by_question"]["next_action"]

    assert m["accuracy"] == pytest.approx(0.5)
    assert m["soft_accuracy"] == pytest.approx((0.7 + 0.4) / 2)
    assert m["brier"] == pytest.approx((0.14 + 0.62) / 2)
    assert m["ece"] == pytest.approx(0.5 * 0.3 + 0.5 * 0.5)


def test_remaining_hops_mae_uses_expected_score_and_also_reports_argmax_mae():
    m = laya_evaluator.evaluate_agent(FakeAgent(), ROWS)["by_question"]["remaining_hops"]

    assert m["accuracy"] == pytest.approx(1.0)
    assert m["mae"] == pytest.approx((0.5 + 0.4) / 2)  # 期望分數 0.5 對 0、1.6 對 2
    assert m["mae_argmax"] == pytest.approx(0.0)
    assert m["within_one"] == pytest.approx(1.0)
    assert m["ece"] == pytest.approx(0.5 * 0.4 + 0.5 * 0.3)


def test_repeated_question_type_for_the_same_state_starts_a_new_predict_call():
    rows = _rows("S1", True, "A", 0)[:1] * 2
    agent = FakeAgent()

    result = laya_evaluator.evaluate_agent(agent, rows)

    assert len(agent.calls) == 2 and result["by_question"]["sufficient"]["n"] == 2


def _metrics(sufficient, next_action, mae):
    return {"by_question": {"sufficient": {"accuracy": sufficient}, "next_action": {"accuracy": next_action},
                            "remaining_hops": {"mae": mae}}}


def test_verdict_passes_when_every_threshold_holds_including_float_boundaries():
    verdict = laya_evaluator.judge(_metrics(0.75, 0.80, 0.5), _metrics(0.55, 0.40, 1.0))

    assert verdict["passed"] is True
    assert [c["passed"] for c in verdict["checks"]] == [True] * 5


@pytest.mark.parametrize(
    ("finetuned", "zero_shot", "failed"),
    [
        (_metrics(0.74, 0.90, 0.3), _metrics(0.10, 0.10, 1.0), "sufficient accuracy"),
        (_metrics(0.90, 0.90, 0.3), _metrics(0.10, 0.75, 1.0), "next_action gain over zero-shot"),
        (_metrics(0.90, 0.90, 0.51), _metrics(0.10, 0.10, 1.0), "remaining_hops MAE"),
    ],
)
def test_verdict_fails_and_names_the_failing_check(finetuned, zero_shot, failed):
    verdict = laya_evaluator.judge(finetuned, zero_shot)

    assert verdict["passed"] is False
    assert [c["name"] for c in verdict["checks"] if not c["passed"]] == [failed]


def _report_inputs(passed):
    finetuned = laya_evaluator.evaluate_agent(FakeAgent(), ROWS)
    zero_shot = laya_evaluator.evaluate_agent(FakeAgent(), ROWS)
    verdict = {"passed": passed, "checks": laya_evaluator.judge(finetuned, zero_shot)["checks"]}
    train_summary = {"elapsed_s": 5400.0, "gpu": "NVIDIA GeForce RTX 4090", "hyperparameters": {"epochs": 2}}
    return dict(finetuned=finetuned, zero_shot=zero_shot, verdict=verdict, train_summary=train_summary,
                model_dir="/rt/models/laya_multihop", model_hash="ab" * 32, test_path="/data/test.jsonl", limit=None)


def test_fail_report_marks_fail_and_lists_tunable_parameters(tmp_path):
    md_path, json_path = laya_evaluator.write_report(tmp_path / "laya_eval", **_report_inputs(passed=False))

    md = md_path.read_text(encoding="utf-8")
    assert md_path.name == "laya_eval_report.md" and json_path.name == "laya_eval_report.json"
    assert "判定：FAIL" in md and "判定：PASS" not in md
    for word in ("--epochs", "--lr-encoder", "--sigma-start", "--group-size", "Brier", "ECE", "p50",
                 "90.0 min", "RTX 4090", "ab" * 32, "zero-shot"):
        assert word in md
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["verdict"]["passed"] is False and data["model_hash"] == "ab" * 32
    assert data["finetuned"]["by_question"]["sufficient"]["n"] == 2
    assert data["tunable_parameters"]


def test_pass_report_has_no_tunable_parameter_section(tmp_path):
    md_path, _ = laya_evaluator.write_report(tmp_path, **_report_inputs(passed=True))

    md = md_path.read_text(encoding="utf-8")
    assert "判定：PASS" in md and "--lr-encoder" not in md


def test_evaluate_reads_test_jsonl_with_limit_and_loads_model_dir(tmp_path, monkeypatch):
    path = tmp_path / "test.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in ROWS), encoding="utf-8")
    loaded = []
    monkeypatch.setattr(laya_evaluator, "load_agent", lambda d, device: loaded.append((d, device)) or FakeAgent())

    result = laya_evaluator.evaluate("/models/x", path, device="cpu", limit=3)

    assert loaded == [("/models/x", "cpu")]
    assert result["decisions"] == 3 and result["predict_calls"] == 1
