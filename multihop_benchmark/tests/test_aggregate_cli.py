import json

import pytest

from multihop_benchmark.cli import main

from test_results_aggregator import LAYA_GROUP, LLM_GROUP, PROBE, RUN_ID, TABLE_STEMS, write_records


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "laya_data"
    monkeypatch.setenv("PROJECT_DATA_ROOT", str(root))
    monkeypatch.setenv("PROJECT_RUNTIME_ROOT", str(tmp_path / "laya_runtime"))
    return root


def log_events(data_root):
    path = data_root / "logs" / "benchmark.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_aggregate_writes_summary_tables_figures_and_logs(data_root, tmp_path, capsys):
    run_dir = write_records(data_root / "runs" / RUN_ID, [LAYA_GROUP, LLM_GROUP])
    (run_dir / "probe_results.json").write_text(json.dumps(PROBE), encoding="utf-8")
    paper = tmp_path / "paper"

    exit_code = main(["aggregate", "--run-id", RUN_ID, "--paper-dir", str(paper)])

    assert exit_code == 0
    assert (run_dir / "summary.csv").is_file()
    assert {p.name for p in (paper / "tables").iterdir()} == {f"{stem}.tex" for stem in (*TABLE_STEMS, "probe")}
    assert len(list((paper / "figures").glob("*.pdf"))) == 5
    output = capsys.readouterr().out
    assert str(run_dir / "summary.csv") in output
    (event,) = [e for e in log_events(data_root) if e["event"] == "aggregate_finished"]
    assert event["run_id"] == RUN_ID and event["source"] == "aggregate" and event["level"] == "INFO"


def test_aggregate_probe_only(data_root, tmp_path):
    run_dir = data_root / "runs" / RUN_ID
    run_dir.mkdir(parents=True)
    (run_dir / "probe_results.json").write_text(json.dumps(PROBE), encoding="utf-8")
    paper = tmp_path / "paper"

    exit_code = main(["aggregate", "--run-id", RUN_ID, "--probe", "--paper-dir", str(paper)])

    assert exit_code == 0
    assert [p.name for p in (paper / "tables").iterdir()] == ["probe.tex"]
    assert not (paper / "figures").exists() and not (run_dir / "summary.csv").exists()


def test_aggregate_unknown_run_fails_with_message(data_root, tmp_path, capsys):
    exit_code = main(["aggregate", "--run-id", "nope", "--paper-dir", str(tmp_path / "paper")])

    assert exit_code == 1
    assert "records" in capsys.readouterr().out
    (event,) = [e for e in log_events(data_root) if e["event"] == "aggregate_failed"]
    assert event["level"] == "ERROR" and event["run_id"] == "nope"


def test_aggregate_requires_run_id(data_root, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["aggregate"])
    assert excinfo.value.code != 0
    assert "--run-id" in capsys.readouterr().err


def test_aggregate_corrupt_record_line_exits_nonzero(data_root, tmp_path, capsys):
    run_dir = write_records(data_root / "runs" / RUN_ID, [LLM_GROUP])
    (path,) = list((run_dir / "records").iterdir())
    path.write_text(path.read_text(encoding="utf-8").replace("\n", "\n{broken\n", 1), encoding="utf-8")

    exit_code = main(["aggregate", "--run-id", RUN_ID, "--paper-dir", str(tmp_path / "paper")])

    assert exit_code == 1
    assert f"{path.name}:2" in capsys.readouterr().out
    assert not (run_dir / "summary.csv").exists()


def test_aggregate_reports_removed_stale_probe_table(data_root, tmp_path, capsys):
    write_records(data_root / "runs" / RUN_ID, [LLM_GROUP])
    stale = tmp_path / "paper" / "tables" / "probe.tex"
    stale.parent.mkdir(parents=True)
    stale.write_text("% old run\n", encoding="utf-8")

    exit_code = main(["aggregate", "--run-id", RUN_ID, "--paper-dir", str(tmp_path / "paper")])

    assert exit_code == 0 and not stale.exists()
    assert f"已移除 {stale}" in capsys.readouterr().out
