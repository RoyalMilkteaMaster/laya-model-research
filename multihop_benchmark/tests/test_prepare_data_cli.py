import json

import pytest

from multihop_benchmark.cli import main
from multihop_benchmark.datasets import multihop_dataset_loader
from multihop_benchmark.datasets.multihop_dataset_loader import OUTPUT_NAMES, DatasetDownloadError


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "laya_data"
    monkeypatch.setenv("PROJECT_DATA_ROOT", str(root))
    monkeypatch.setenv("PROJECT_RUNTIME_ROOT", str(tmp_path / "laya_runtime"))
    return root


def synthetic_hotpot_rows(prefix, count):
    return [
        {
            "id": f"{prefix}{i:05d}",
            "question": f"question {i}?",
            "answer": f"answer {i}",
            "type": "bridge",
            "level": "easy",
            # 每 100 題一題帶越界 sent_id 3（S{i}b 只有 1 句）
            "supporting_facts": {
                "title": [f"S{i}a", f"S{i}b"] + ([f"S{i}b"] if i % 100 == 0 else []),
                "sent_id": [0, 0] + ([3] if i % 100 == 0 else []),
            },
            "context": {
                "title": [f"S{i}a", f"D{i % 7}", f"S{i}b"],
                "sentences": [[f"Support a of {i}."], [f"Shared distractor {i % 7}."], [f"Support b of {i}."]],
            },
        }
        for i in range(count)
    ]


@pytest.fixture
def fake_hub(monkeypatch):
    from datasets import Dataset

    calls = []
    splits = {
        "train": Dataset.from_list(synthetic_hotpot_rows("t", 1005)),
        "validation": Dataset.from_list(synthetic_hotpot_rows("v", 205)),
    }

    def load_raw_dataset(dataset, *, logger, **kwargs):
        calls.append(dataset)
        if dataset == "musique":
            raise DatasetDownloadError("network down")
        return splits

    monkeypatch.setattr(multihop_dataset_loader, "load_raw_dataset", load_raw_dataset)
    return calls


def log_events(data_root):
    lines = (data_root / "logs" / "prepare_data.jsonl").read_text("utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_prepare_data_writes_outputs_licenses_and_prints_counts(data_root, fake_hub, capsys):
    exit_code = main(["prepare-data", "--datasets", "hotpotqa"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "hotpotqa: sample_200=200 train_800=800 calibration_100=100 test_100=100 corpus=" in out
    dataset_dir = data_root / "datasets" / "hotpotqa"
    for name in OUTPUT_NAMES:
        assert (dataset_dir / f"{name}.jsonl").is_file()
    assert (dataset_dir / "prepare_manifest.json").is_file()
    licenses = (data_root / "datasets" / "LICENSES.md").read_text("utf-8")
    assert "CC BY-SA 4.0" in licenses and "Apache-2.0" in licenses and "CC BY 4.0" in licenses
    events = log_events(data_root)
    prepared = next(e for e in events if e["event"] == "dataset_prepared")
    assert prepared["source"] == "prepare-data" and prepared["seed"] == 42
    assert prepared["counts"]["sample_200"] == 200
    assert events[-1]["event"] == "prepare_data_finished"
    out_of_range = [e for e in events if e["event"] == "supporting_sentence_out_of_range"]
    assert out_of_range and {e["level"] for e in out_of_range} == {"WARNING"}
    assert out_of_range[0]["source"] == "prepare-data" and out_of_range[0]["sent_id"] == 3


def test_prepare_data_skips_complete_outputs_unless_forced(data_root, fake_hub, capsys):
    main(["prepare-data", "--datasets", "hotpotqa"])
    sample = data_root / "datasets" / "hotpotqa" / "sample_200.jsonl"
    before = sample.read_bytes()

    assert main(["prepare-data", "--datasets", "hotpotqa"]) == 0
    assert fake_hub == ["hotpotqa"]
    assert "hotpotqa: sample_200=200" in capsys.readouterr().out
    assert "dataset_outputs_exist" in [e["event"] for e in log_events(data_root)]

    assert main(["prepare-data", "--datasets", "hotpotqa", "--force"]) == 0
    assert fake_hub == ["hotpotqa", "hotpotqa"]
    assert sample.read_bytes() == before


def test_prepare_data_rebuilds_invalid_outputs_instead_of_skipping(data_root, fake_hub, capsys):
    dataset_dir = data_root / "datasets" / "hotpotqa"
    dataset_dir.mkdir(parents=True)
    for name in OUTPUT_NAMES:
        (dataset_dir / f"{name}.jsonl").write_text("", encoding="utf-8")

    assert main(["prepare-data", "--datasets", "hotpotqa"]) == 0

    assert fake_hub == ["hotpotqa"]
    out = capsys.readouterr().out
    assert "既有輸出不完整" in out and "（已存在，略過）" not in out
    assert "hotpotqa: sample_200=200 train_800=800 calibration_100=100 test_100=100 corpus=" in out
    invalid = [e for e in log_events(data_root) if e["event"] == "dataset_outputs_invalid"]
    assert invalid and invalid[0]["level"] == "WARNING" and "prepare_manifest.json" in invalid[0]["problem"]


def test_prepare_data_rebuilds_when_an_answer_was_edited(data_root, fake_hub, capsys):
    main(["prepare-data", "--datasets", "hotpotqa"])
    sample = data_root / "datasets" / "hotpotqa" / "sample_200.jsonl"
    original = sample.read_bytes()
    lines = sample.read_text("utf-8").splitlines(True)
    row = json.loads(lines[0])
    row["answer"] = "__REVIEWER_CORRUPTED_GOLD__"
    sample.write_text(json.dumps(row, ensure_ascii=False) + "\n" + "".join(lines[1:]), encoding="utf-8")
    capsys.readouterr()

    assert main(["prepare-data", "--datasets", "hotpotqa"]) == 0

    assert fake_hub == ["hotpotqa", "hotpotqa"]
    out = capsys.readouterr().out
    assert "既有輸出不完整" in out and "sample_200.jsonl" in out and "（已存在，略過）" not in out
    assert sample.read_bytes() == original
    invalid = [e for e in log_events(data_root) if e["event"] == "dataset_outputs_invalid"]
    assert invalid and invalid[-1]["level"] == "WARNING" and "sample_200.jsonl" in invalid[-1]["problem"]


def test_prepare_data_continues_after_download_failure_and_exits_nonzero(data_root, fake_hub, capsys):
    exit_code = main(["prepare-data", "--datasets", "musique", "hotpotqa"])

    assert exit_code == 1
    assert fake_hub == ["musique", "hotpotqa"]
    assert (data_root / "datasets" / "hotpotqa" / "corpus.jsonl").is_file()
    assert not (data_root / "datasets" / "musique").exists()
    assert "musique" in capsys.readouterr().out


def test_prepare_data_rejects_unknown_dataset(data_root, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["prepare-data", "--datasets", "squad"])
    assert excinfo.value.code != 0
    assert "--datasets" in capsys.readouterr().err
