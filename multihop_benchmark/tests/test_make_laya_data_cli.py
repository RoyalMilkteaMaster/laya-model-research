"""cli make-laya-data：產生三檔、印出統計、token 最大值與抽驗結果。"""

import json

import pytest

from multihop_benchmark.cli import main
from multihop_benchmark.laya_decision import decision_state_builder

from test_weak_label_generator import TOK, write_split_files, hotpot_question, musique_question


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "laya_data"
    monkeypatch.setenv("PROJECT_DATA_ROOT", str(root))
    monkeypatch.setenv("PROJECT_RUNTIME_ROOT", str(tmp_path / "laya_runtime"))
    monkeypatch.setattr(decision_state_builder, "load_laya_tokenizer", lambda: TOK)
    write_split_files(root, {
        "hotpotqa": lambda o: [hotpot_question(f"h{o + i}") for i in range(10)],
        "2wiki": lambda o: [hotpot_question(f"w{o + i}") for i in range(10)],
        "musique": lambda o: [musique_question(f"m{o + i}", gold=2 + i % 3) for i in range(10)],
    })
    return root


def test_make_laya_data_prints_stats_max_tokens_and_audit(data_root, capsys):
    assert main(["make-laya-data", "--seed", "42"]) == 0

    out = capsys.readouterr().out
    for split in ("train", "calibration", "test"):
        assert (data_root / "laya_decisions" / f"{split}.jsonl").is_file()
        assert f"[{split}]" in out
    for word in ("question_only", "partial_gold", "full_gold", "distractor_only", "full_paragraph",
                 "supporting_only", "max_state_tokens=", "sha256=", "fallback", "mismatches=0"):
        assert word in out
    logs = (data_root / "logs" / "prepare_data.jsonl").read_text(encoding="utf-8").splitlines()
    events = [json.loads(line)["event"] for line in logs]
    assert "make_laya_data_started" in events and "make_laya_data_finished" in events


def test_make_laya_data_missing_inputs_exits_nonzero(data_root, capsys):
    (data_root / "datasets" / "musique" / "train_800.jsonl").unlink()

    assert main(["make-laya-data"]) == 1
    assert "prepare-data" in capsys.readouterr().out
    assert not (data_root / "laya_decisions" / "train.jsonl").exists()
