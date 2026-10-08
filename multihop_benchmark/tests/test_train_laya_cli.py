"""cli train-laya：以假訓練（GPU 邊界）與假 laya agent 驗證流程、旗標、報告與退出碼。"""

from __future__ import annotations

import json

import pytest

from multihop_benchmark.cli import main
from multihop_benchmark.laya_decision import laya_evaluator, laya_trainer
from multihop_benchmark.laya_decision.weak_label_generator import DECISION_QUESTIONS


def _rows(sufficient, action, remaining):
    state = f"S|{sufficient}|{action}|{remaining}"
    return [
        {"state": state, **DECISION_QUESTIONS["sufficient"], "target": [0.0, 1.0] if sufficient else [1.0, 0.0]},
        {"state": state, **DECISION_QUESTIONS["next_action"], "target": [float(a == action) for a in "ABC"]},
        {"state": state, **DECISION_QUESTIONS["remaining_hops"], "target": [float(i == remaining) for i in range(3)]},
    ]


class OracleAgent:
    """`right=True` 依 state 內嵌的標籤作答；否則固定答 false／C／0。"""

    def __init__(self, right):
        self.right = right

    def predict(self, state, questions):
        _, sufficient, action, remaining = state.split("|")
        if not self.right:
            sufficient, action, remaining = "False", "C", "0"
        score = [0.05, 0.05, 0.05]
        score[int(remaining)] = 0.9
        answers = {}
        for key, q in questions.items():
            if q["type"] == "noul":
                answers[key] = {"noul": 0.9 if sufficient == "True" else 0.1}
            elif q["type"] == "choice":
                answers[key] = {"choice": action, "probabilities": {a: 0.9 if a == action else 0.05 for a in "ABC"}}
            else:
                answers[key] = {"score": sum(i * p for i, p in enumerate(score)),
                                "probabilities": {str(i): p for i, p in enumerate(score)}}
        return {"answers": answers}


@pytest.fixture
def env(tmp_path, monkeypatch):
    data_root, runtime_root = tmp_path / "laya_data", tmp_path / "laya_runtime"
    monkeypatch.setenv("PROJECT_DATA_ROOT", str(data_root))
    monkeypatch.setenv("PROJECT_RUNTIME_ROOT", str(runtime_root))
    rows = _rows(True, "A", 0) + _rows(False, "B", 1) + _rows(False, "C", 2) + _rows(False, "B", 2)
    decisions = data_root / "laya_decisions"
    decisions.mkdir(parents=True)
    for split in ("train", "calibration", "test"):
        (decisions / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    model_dir = runtime_root / "models" / "laya_multihop"
    state = {"train_calls": [], "loads": [], "zero_shot_right": False}

    def fake_train(train_path, calibration_path, output_dir, config, *, device, logger, **kwargs):
        state["train_calls"].append({"train_path": train_path, "calibration_path": calibration_path,
                                     "output_dir": output_dir, "config": config, "device": device})
        _write_model(output_dir)
        return {"elapsed_s": 4321.0, "gpu": "FakeGPU 24GB", "hyperparameters": {"epochs": config.epochs},
                "temperature": [1.1, 1.0, 0.9], "train_items": 12, "calibration_items": 12}

    def fake_load(model, device):
        state["loads"].append((str(model), device))
        return OracleAgent(right=str(model) == str(model_dir) or state["zero_shot_right"])

    monkeypatch.setattr(laya_trainer, "train", fake_train)
    monkeypatch.setattr(laya_evaluator, "load_agent", fake_load)
    return {"data_root": data_root, "model_dir": model_dir, "state": state,
            "report_dir": data_root / "runs" / "laya_eval"}


def _write_model(output_dir):
    (output_dir / "encoder").mkdir(parents=True, exist_ok=True)
    (output_dir / "model.safetensors").write_bytes(b"weights")
    (output_dir / "rl_agent_config.json").write_text('{"temperature": [1.1, 1.0, 0.9]}', encoding="utf-8")


def _events(data_root):
    lines = (data_root / "logs" / "train_laya.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_train_then_evaluate_passes_writes_summary_report_and_logs(env, capsys):
    code = main(["train-laya", "--epochs", "3", "--limit", "9", "--lr-encoder", "3e-5", "--micro-batch", "4",
                 "--device", "cpu"])

    assert code == 0
    call, = env["state"]["train_calls"]
    assert call["output_dir"] == env["model_dir"] and call["device"] == "cpu"
    assert call["train_path"] == env["data_root"] / "laya_decisions" / "train.jsonl"
    assert (call["config"].epochs, call["config"].limit, call["config"].lr_encoder, call["config"].micro_batch) \
        == (3, 9, 3e-5, 4)
    summary = json.loads((env["report_dir"] / "train_summary.json").read_text(encoding="utf-8"))
    assert summary["elapsed_s"] == 4321.0
    md = (env["report_dir"] / "laya_eval_report.md").read_text(encoding="utf-8")
    assert "判定：PASS" in md and "72.0 min" in md and "--limit 9" in md
    report = json.loads((env["report_dir"] / "laya_eval_report.json").read_text(encoding="utf-8"))
    assert report["finetuned"]["decisions"] == 9 and report["model_hash"]
    assert [load[0] for load in env["state"]["loads"]] == [str(env["model_dir"]), laya_trainer.BASE_MODEL]
    out = capsys.readouterr().out
    assert "PASS" in out and "sufficient" in out
    events = [e["event"] for e in _events(env["data_root"])]
    assert events[0] == "train_laya_started" and "train_summary_written" in events and events[-1] == "eval_finished"


def test_thresholds_not_met_writes_fail_report_and_exits_non_zero(env):
    env["state"]["zero_shot_right"] = True  # zero-shot 一樣好 → 增益 < 0.20

    assert main(["train-laya", "--device", "cpu"]) == 1

    md = (env["report_dir"] / "laya_eval_report.md").read_text(encoding="utf-8")
    assert "判定：FAIL" in md and "--epochs" in md
    finished = _events(env["data_root"])[-1]
    assert finished["event"] == "eval_finished" and finished["level"] == "ERROR"
    assert finished["errorCode"] == "LAYA_EVAL_THRESHOLD_FAILED"


def test_eval_only_without_a_model_fails_without_training(env, capsys):
    assert main(["train-laya", "--eval-only", "--device", "cpu"]) == 1

    assert env["state"]["train_calls"] == [] and env["state"]["loads"] == []
    assert "找不到模型" in capsys.readouterr().out


def test_eval_only_reuses_existing_model_and_train_summary(env):
    _write_model(env["model_dir"])
    env["report_dir"].mkdir(parents=True)
    (env["report_dir"] / "train_summary.json").write_text(json.dumps({"elapsed_s": 600.0, "gpu": "G"}), encoding="utf-8")

    assert main(["train-laya", "--eval-only", "--device", "cpu"]) == 0

    assert env["state"]["train_calls"] == []
    assert "10.0 min" in (env["report_dir"] / "laya_eval_report.md").read_text(encoding="utf-8")


def test_skip_train_trains_only_when_the_model_is_missing(env):
    assert main(["train-laya", "--skip-train", "--device", "cpu"]) == 0
    assert main(["train-laya", "--skip-train", "--device", "cpu"]) == 0

    assert len(env["state"]["train_calls"]) == 1


def test_missing_decision_data_fails_with_a_hint(env, capsys):
    (env["data_root"] / "laya_decisions" / "test.jsonl").unlink()

    assert main(["train-laya", "--device", "cpu"]) == 1

    assert "make-laya-data" in capsys.readouterr().out and env["state"]["train_calls"] == []


def test_training_failure_is_logged_and_exits_non_zero_without_evaluating(env, monkeypatch, capsys):
    def exploding_train(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(laya_trainer, "train", exploding_train)

    assert main(["train-laya", "--device", "cpu"]) == 1

    assert env["state"]["loads"] == [] and "CUDA out of memory" in capsys.readouterr().out
    finished = _events(env["data_root"])[-1]
    assert finished["event"] == "train_laya_finished" and finished["errorCode"] == "LAYA_TRAIN_FAILED"
