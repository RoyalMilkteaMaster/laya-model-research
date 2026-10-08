from datetime import date, timedelta

import pytest

from multihop_benchmark.cli import main


@pytest.fixture
def roots(tmp_path, monkeypatch):
    data_root = tmp_path / "laya_data"
    runtime_root = tmp_path / "laya_runtime"
    monkeypatch.setenv("PROJECT_DATA_ROOT", str(data_root))
    monkeypatch.setenv("PROJECT_RUNTIME_ROOT", str(runtime_root))
    return data_root, runtime_root


def dated(name, days_ago):
    return f"{name}-{date.today() - timedelta(days=days_ago):%Y-%m-%d}.jsonl"


def test_clean_logs_deletes_only_rotated_files_older_than_threshold(roots):
    logs_dir = roots[0] / "logs"
    logs_dir.mkdir(parents=True)
    old = [dated("benchmark", 91), dated("train_laya", 200)]
    kept = [
        "benchmark.jsonl",
        "prepare_data.jsonl",
        dated("benchmark", 90),
        dated("prepare_data", 1),
        "notes-2000-01-01.txt",
    ]
    for name in old + kept:
        (logs_dir / name).write_text("{}\n", encoding="utf-8")

    exit_code = main(["clean-logs", "--older-than", "90"])

    assert exit_code == 0
    remaining = {p.name for p in logs_dir.iterdir()}
    assert set(kept) <= remaining
    assert not set(old) & remaining


def test_clean_logs_requires_older_than(roots, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["clean-logs"])
    assert excinfo.value.code != 0
    assert "--older-than" in capsys.readouterr().err


def test_clean_logs_rejects_negative_days(roots, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["clean-logs", "--older-than", "-1"])
    assert excinfo.value.code != 0
    assert "--older-than" in capsys.readouterr().err


def test_setup_check_fails_and_names_missing_items(roots, monkeypatch, capsys):
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")

    exit_code = main(["setup-check"])

    output = capsys.readouterr().out
    assert exit_code != 0
    assert "Data Root" in output and str(roots[0]) in output
    assert "Runtime Root" in output and str(roots[1]) in output
    assert "Ollama" in output and "http://127.0.0.1:9" in output
    assert "缺少" in output


def test_setup_check_reports_missing_settings(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("PROJECT_DATA_ROOT", raising=False)
    monkeypatch.delenv("PROJECT_RUNTIME_ROOT", raising=False)
    monkeypatch.setattr("multihop_benchmark.settings.DEFAULT_ENV_FILE", tmp_path / "none.env")

    exit_code = main(["setup-check"])

    assert exit_code != 0
    assert "PROJECT_DATA_ROOT" in capsys.readouterr().out


@pytest.mark.integration
def test_setup_check_passes_on_prepared_machine(capsys):
    exit_code = main(["setup-check"])

    output = capsys.readouterr().out
    assert exit_code == 0, output
    assert "MiB" in output
