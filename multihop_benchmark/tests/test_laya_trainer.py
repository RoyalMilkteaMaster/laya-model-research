"""laya_trainer：資料組裝（build_item，用假 tokenizer）、溫度校準與設定檢查；實際訓練由 `cli train-laya` 在 GPU 上驗證。"""

from __future__ import annotations

import json

import numpy as np
import pytest

from multihop_benchmark.laya_decision import laya_trainer
from multihop_benchmark.laya_decision.weak_label_generator import DECISION_QUESTIONS


class FakeTokenizer:
    """laya.common.build_sequence 需要的最小介面：`tok(text, **kw)["input_ids"]` 與特殊 token。"""

    mask_token, mask_token_id, cls_token_id, sep_token_id, pad_token_id = "[MASK]", 4, 1, 2, 0

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [10 + len(word) for word in text.split()]
        return {"input_ids": ids[:max_length] if truncation and max_length else ids}


def _row(name, target, state="Question: Who? Evidence: [1] A: b c."):
    return {"state": state, **DECISION_QUESTIONS[name], "target": target, "label": None}


@pytest.mark.parametrize(
    ("name", "target", "qtype", "label"),
    [("sufficient", [0.0, 1.0], 2, 1), ("next_action", [0.0, 0.0, 1.0], 0, 2), ("remaining_hops", [0.0, 1.0, 0.0], 1, 1)],
)
def test_build_item_has_one_marker_per_option_and_keeps_the_target(name, target, qtype, label):
    item = laya_trainer.build_item(_row(name, target), FakeTokenizer(), max_len=1024, head_max_len=256)

    assert item["qtype"] == qtype and item["label"] == label and item["target"] == target
    assert len(item["markers"]) == len(target)
    assert all(item["ids"][m] == FakeTokenizer.mask_token_id for m in item["markers"])
    assert item["ids"][0] == FakeTokenizer.cls_token_id and item["ids"][-1] == FakeTokenizer.sep_token_id


def test_build_item_truncates_the_state_to_max_len():
    long_state = "word " * 5000
    item = laya_trainer.build_item(_row("sufficient", [1.0, 0.0], long_state), FakeTokenizer(), max_len=300, head_max_len=64)

    assert len(item["ids"]) == 300 and len(item["markers"]) == 2


def test_build_item_rejects_target_length_that_does_not_match_the_options():
    assert laya_trainer.build_item(_row("next_action", [1.0, 0.0]), FakeTokenizer(), max_len=1024, head_max_len=256) is None


def test_build_items_reads_jsonl_with_limit_and_counts_skipped(tmp_path):
    path = tmp_path / "train.jsonl"
    rows = [_row("sufficient", [1.0, 0.0]), _row("next_action", [1.0, 0.0]), _row("remaining_hops", [0, 0, 1.0]),
            _row("sufficient", [0.0, 1.0])]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    items, skipped = laya_trainer.build_items(path, FakeTokenizer(), max_len=1024, head_max_len=256, limit=3)

    assert len(items) == 2 and skipped == 1


def test_fit_temperature_recovers_the_scale_of_overconfident_logits():
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(3000):
        z = rng.normal(size=3)
        p = np.exp(z) / np.exp(z).sum()
        target = np.eye(3)[rng.choice(3, p=p)]
        samples.append((2.0 * z, target))  # 真實機率為 softmax(z)，模型輸出放大 2 倍 → 最佳溫度約 2

    assert laya_trainer.fit_temperature(samples) == pytest.approx(2.0, abs=0.2)


def test_fit_temperature_defaults_to_one_with_too_few_samples():
    assert laya_trainer.fit_temperature([(np.array([3.0, 0.0]), np.array([1.0, 0.0]))] * 5) == 1.0


@pytest.mark.parametrize(
    "overrides",
    [{"group_size": 1}, {"sigma_start": 0.0}, {"sigma_end": -0.1}, {"epochs": 0}, {"micro_batch": 0}, {"grad_accum": 0}],
)
def test_invalid_config_is_rejected_before_any_model_is_loaded(overrides, tmp_path):
    config = laya_trainer.TrainConfig(**overrides)

    with pytest.raises(ValueError):
        laya_trainer.train(tmp_path / "t.jsonl", tmp_path / "c.jsonl", tmp_path / "out", config)


def test_default_config_keeps_the_teaching_rlcd_settings_and_contract_sequence_lengths():
    config = laya_trainer.TrainConfig()

    assert (config.max_len, config.head_max_len) == (1024, 256)
    assert (config.group_size, config.sigma_start, config.sigma_end) == (4, 0.4, 0.1)
    assert (config.w_sph, config.w_rps, config.ce_weight) == (0.75, 1.0, 1.0)


def _generation_dir(path, generation):
    (path / "encoder").mkdir(parents=True)
    (path / "model.safetensors").write_text(generation, encoding="utf-8")
    (path / "rl_agent_config.json").write_text("{}", encoding="utf-8")
    return path


def _leftovers(root):
    return sorted(p.name for p in root.iterdir() if p.name != "laya_multihop")


def test_publish_validates_the_partial_before_it_replaces_the_published_model(tmp_path):
    output = _generation_dir(tmp_path / "laya_multihop", "old")
    partial = _generation_dir(tmp_path / "laya_multihop.partial", "new")
    seen = []

    def validate(path):
        seen.append((path, (output / "model.safetensors").read_text(encoding="utf-8")))

    laya_trainer.publish_model(partial, output, validate=validate)

    assert seen == [(partial, "old")]  # 驗證的是暫存目錄，且當下正式目錄仍是舊模型
    assert (output / "model.safetensors").read_text(encoding="utf-8") == "new"
    assert _leftovers(tmp_path) == []


def test_validation_failure_keeps_the_last_known_good_model_and_removes_the_partial(tmp_path):
    output = _generation_dir(tmp_path / "laya_multihop", "old")
    partial = _generation_dir(tmp_path / "laya_multihop.partial", "new-unvalidated")

    def validate(path):
        raise ValueError("simulated laya.load failure")

    with pytest.raises(ValueError, match="simulated"):
        laya_trainer.publish_model(partial, output, validate=validate)

    assert (output / "model.safetensors").read_text(encoding="utf-8") == "old"
    assert _leftovers(tmp_path) == []


def test_validation_failure_without_a_previous_model_publishes_nothing(tmp_path):
    output = tmp_path / "laya_multihop"
    partial = _generation_dir(tmp_path / "laya_multihop.partial", "new-unvalidated")

    with pytest.raises(RuntimeError):
        laya_trainer.publish_model(partial, output, validate=lambda path: (_ for _ in ()).throw(RuntimeError("bad")))

    assert not output.exists() and _leftovers(tmp_path) == []


def test_failed_rename_into_place_restores_the_previous_model(tmp_path, monkeypatch):
    output = _generation_dir(tmp_path / "laya_multihop", "old")
    partial = _generation_dir(tmp_path / "laya_multihop.partial", "new")
    real_rename = type(partial).rename

    def rename(self, target):
        if self.name == partial.name:
            raise OSError("simulated rename failure")
        return real_rename(self, target)

    monkeypatch.setattr(type(partial), "rename", rename)

    with pytest.raises(OSError, match="simulated"):
        laya_trainer.publish_model(partial, output, validate=lambda path: None)

    assert (output / "model.safetensors").read_text(encoding="utf-8") == "old"
    assert not (tmp_path / "laya_multihop.old").exists()
