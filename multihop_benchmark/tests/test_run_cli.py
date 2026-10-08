"""`cli run` 與 `cli probe-models`：參數、退出碼與輸出（以假 Backends 取代 Ollama／GPU）。"""

from __future__ import annotations

import json

import pytest

from fakes import tool_reply
from multihop_benchmark.agent.protocols import LLMConnectionError
from multihop_benchmark.cli import main
from multihop_benchmark.runs import benchmark_runner, probe
from test_benchmark_runner import FakeOllama, FakeSampler, OneRetriever, OneShotDecision, PolicyLLM, write_questions


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    root = tmp_path / "laya_data"
    monkeypatch.setenv("PROJECT_DATA_ROOT", str(root))
    monkeypatch.setenv("PROJECT_RUNTIME_ROOT", str(tmp_path / "laya_runtime"))
    write_questions(root)
    state = {"fail": None, "calls": []}

    def fake_backends(settings, **kwargs):
        state["calls"].append(kwargs)
        return benchmark_runner.Backends(
            llm_factory=lambda model, num_ctx: PolicyLLM(model, num_ctx, [], state["fail"]),
            ollama=FakeOllama([], digests={m: f"sha-{m}" for m in benchmark_runner.MODELS}),
            decision_factory=OneShotDecision, retriever_factory=lambda d: OneRetriever(),
            sampler_factory=FakeSampler, laya_model_hash=lambda: "h", git_commit=lambda: "c",
        )

    monkeypatch.setattr(benchmark_runner, "production_backends", fake_backends)
    return root, state


def test_run_subset_writes_records_and_returns_zero(cli_env, capsys):
    root, _ = cli_env
    code = main(["run", "--run-id", "cli", "--limit", "2", "--models", "qwen3.5:4b", "qwen3.5:2b",
                 "--datasets", "2wiki", "--conditions", "rag_llm", "recall_laya", "--timeout-s", "30"])
    assert code == 0
    files = sorted(p.name for p in (root / "runs" / "cli" / "records").iterdir())
    assert files == ["2wiki__qwen3.5-2b__rag_llm.jsonl", "2wiki__qwen3.5-2b__recall_laya.jsonl",
                     "2wiki__qwen3.5-4b__rag_llm.jsonl", "2wiki__qwen3.5-4b__recall_laya.jsonl"]
    manifest = json.loads((root / "runs" / "cli" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["loop"]["timeout_s"] == 30 and manifest["questions_per_combo"] == 2
    assert "本次執行 8 題" in capsys.readouterr().out


def test_run_defaults_cover_full_grid(cli_env):
    root, _ = cli_env
    assert main(["run", "--run-id", "grid", "--limit", "1"]) == 0
    manifest = json.loads((root / "runs" / "grid" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["models"] == list(benchmark_runner.MODELS)
    assert manifest["datasets"] == list(benchmark_runner.DATASETS)
    assert manifest["conditions"] == list(benchmark_runner.CONDITIONS)
    assert manifest["loop"]["timeout_s"] == 300
    assert len(list((root / "runs" / "grid" / "records").iterdir())) == 60


def test_run_connection_failure_exits_non_zero(cli_env, capsys):
    _, state = cli_env
    state["fail"] = lambda model, num_ctx, n: LLMConnectionError("Ollama down")
    assert main(["run", "--run-id", "down", "--limit", "1"]) == 1
    assert "Ollama down" in capsys.readouterr().out


def test_run_missing_questions_exits_non_zero(cli_env, capsys):
    root, _ = cli_env
    (root / "datasets" / "musique" / "sample_200.jsonl").unlink()
    assert main(["run", "--run-id", "nodata", "--limit", "1"]) == 1
    assert "prepare-data" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["--datasets", "nope"], ["--conditions", "rag"], ["--limit", "0"]])
def test_run_rejects_bad_arguments(cli_env, argv):
    with pytest.raises(SystemExit):
        main(["run", "--run-id", "bad", *argv])


def test_probe_models_writes_probe_results(cli_env, monkeypatch, capsys):
    root, _ = cli_env
    monkeypatch.setattr(PolicyLLM, "chat", _probe_chat)
    assert main(["probe-models", "--run-id", "p", "--models", "qwen3.5:4b", "qwen3.5:0.8b"]) == 0
    data = json.loads((root / "runs" / "p" / "probe_results.json").read_text(encoding="utf-8"))
    assert {tag: (m["valid_tool_call_rate"], m["n"]) for tag, m in data["models"].items()} == {
        "qwen3.5:4b": (1.0, 20), "qwen3.5:0.8b": (1.0, 20)}
    assert "qwen3.5:0.8b" in capsys.readouterr().out


async def _probe_chat(self, messages, tools=None):
    return tool_reply(probe.PROBE_TOOL_NAME, {"query": "x"})


def test_probe_models_missing_model_exits_non_zero(cli_env, capsys):
    assert main(["probe-models", "--run-id", "p", "--models", "llama:1b"]) == 1
    assert "llama:1b" in capsys.readouterr().out


# ---- 輸入邊界（Review r1 Finding）：CLI 在任何寫入前以非零退出 ----
@pytest.mark.parametrize("command", ["run", "probe-models"])
@pytest.mark.parametrize("bad", ["..", "a/b", "a\\b", "/tmp/escaped-t09-cli"])
def test_cli_rejects_bad_run_id(cli_env, tmp_path, command, bad):
    root, state = cli_env
    with pytest.raises(SystemExit) as exit_info:
        main([command, "--run-id", bad])
    assert exit_info.value.code != 0
    assert not (root / "runs").exists() and state["calls"] == []
    assert not (tmp_path.parent / "escaped-t09-cli").exists()


@pytest.mark.parametrize("argv", [["--datasets", "2wiki", "2wiki"], ["--conditions", "rag_llm", "rag_llm"],
                                  ["--models", "qwen3.5:4b", "qwen3.5:4b"]])
def test_cli_run_rejects_duplicate_selectors(cli_env, argv, capsys):
    root, _ = cli_env
    assert main(["run", "--run-id", "dup", "--limit", "1", *argv]) != 0
    assert "重複" in capsys.readouterr().out
    assert not (root / "runs" / "dup" / "records").exists()
