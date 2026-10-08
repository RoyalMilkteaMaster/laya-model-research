"""benchmark_runner：依模型分組排程、續跑、OOM 重試、連線中止、manifest（假 LLM／Laya／檢索／Ollama）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fakes import decision, final, passage, sub_q, text
from multihop_benchmark.agent.protocols import LLMConnectionError, LLMOutOfMemoryError
from multihop_benchmark.benchmark_logger import get_logger
from multihop_benchmark.runs import benchmark_runner, record_store
from multihop_benchmark.runs.benchmark_runner import Backends

RUN_ID = "unit"
MODELS = ["qwen3.5:9b", "qwen3.5:0.8b"]
DATASETS = ["hotpotqa", "musique"]
CONDITIONS = ["recall_llm", "rag_llm", "recall_laya", "rag_laya"]


# ---- 假物件 ------------------------------------------------------------------


class PolicyLLM:
    """依請求型態回應：有 final_answer 工具 → 停止；只有 ask_sub_question → 提子問題；純文字 → 回憶／作答。

    `fail` 可為 callable(model, num_ctx, call_index) → 要拋出的例外或 None。
    """

    def __init__(self, model, num_ctx, log, fail=None):
        self.model, self.num_ctx, self.log, self.fail = model, num_ctx, log, fail

    async def chat(self, messages, tools=None):
        names = [t["function"]["name"] for t in tools or []]
        self.log.append(("chat", self.model, self.num_ctx))
        error = self.fail(self.model, self.num_ctx, len(self.log)) if self.fail else None
        if error is not None:
            raise error
        if "final_answer" in names:
            return final("Paris")
        if names:
            return sub_q("Where?")
        return text("Paris")


class FakeOllama:
    def __init__(self, log, digests=None, running=()):
        self.log = log
        self.digests_map = digests if digests is not None else {m: f"sha-{m}" for m in MODELS}
        self.loaded = list(running)

    def digests(self):
        return dict(self.digests_map)

    def version(self):
        return "0.0-fake"

    def running(self):
        return list(self.loaded)

    def load(self, model, num_ctx):
        self.log.append(("load", model, num_ctx))
        self.loaded.append(model)

    def unload(self, model):
        self.log.append(("unload", model))
        self.loaded = [m for m in self.loaded if m != model]


class FakeSampler:
    def __init__(self, peak=4096):
        self.peak, self.error = peak, None

    def start(self):
        pass

    def stop(self):
        return self.peak


class OneShotDecision:
    """每題第一跳就 sufficient，Laya 條件 hops=1。"""

    def decide(self, state):
        return decision(0.9, "A", 0)


class OneRetriever:
    def retrieve(self, query):
        return [passage("p1")]


def write_questions(data_root: Path, n: int = 4) -> None:
    for dataset in ("hotpotqa", "2wiki", "musique"):
        path = data_root / "datasets" / dataset / "sample_200.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [{"id": f"{dataset}-q{i}", "question": f"Q{i}?", "answer": "Paris", "answer_aliases": [],
                 "paragraphs": []} for i in range(n)]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def env(tmp_path):
    write_questions(tmp_path)
    log: list = []
    state = {"fail": None, "retrievers": [], "decisions": 0}

    def llm_factory(model, num_ctx):
        log.append(("client", model, num_ctx))
        return PolicyLLM(model, num_ctx, log, state["fail"])

    def decision_factory():
        state["decisions"] += 1
        return OneShotDecision()

    def retriever_factory(dataset):
        state["retrievers"].append(dataset)
        return OneRetriever()

    backends = Backends(
        llm_factory=llm_factory, ollama=FakeOllama(log), decision_factory=decision_factory,
        retriever_factory=retriever_factory, sampler_factory=FakeSampler,
        laya_model_hash=lambda: "laya-hash", git_commit=lambda: "abc123",
    )
    logger = get_logger("run", log_dir=tmp_path / "logs")
    return tmp_path, backends, log, state, logger


def run(env, **overrides):
    data_root, backends, _, _, logger = env
    kwargs = dict(datasets=DATASETS, models=MODELS, conditions=CONDITIONS, limit=2, timeout_s=30,
                  data_root=data_root, backends=backends, logger=logger)
    kwargs.update(overrides)
    return benchmark_runner.run(RUN_ID, **kwargs)


def log_events(data_root: Path) -> list[dict]:
    path = data_root / "logs" / "benchmark.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def records(data_root: Path, combo: str) -> list[dict]:
    path = record_store.records_path(RUN_ID, combo, data_root=data_root)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# ---- 行為 --------------------------------------------------------------------


def test_combo_name_uses_model_slug():
    assert benchmark_runner.combo_name("2wiki", "qwen3.5:0.8b", "rag_laya") == "2wiki__qwen3.5-0.8b__rag_laya"


def test_runs_every_combo_grouped_by_model_and_writes_records(env):
    data_root, _, log, _, _ = env
    summary = run(env)

    started = [e["combo"] for e in log_events(data_root) if e["event"] == "combo_started"]
    expected = [benchmark_runner.combo_name(d, m, c) for m in MODELS for d in DATASETS for c in CONDITIONS]
    assert started == expected  # 同模型 8 組連續，順序依 models → datasets → conditions
    assert summary["questions_run"] == len(expected) * 2

    loads = [entry for entry in log if entry[0] in ("load", "unload")]
    assert loads == [("load", "qwen3.5:9b", 8192), ("unload", "qwen3.5:9b"),
                     ("load", "qwen3.5:0.8b", 8192), ("unload", "qwen3.5:0.8b")]

    rows = records(data_root, "musique__qwen3.5-0.8b__rag_laya")
    assert [r["question_id"] for r in rows] == ["musique-q0", "musique-q1"]
    for row in rows:
        assert row["model"] == "qwen3.5:0.8b" and row["condition"] == "rag_laya" and row["dataset"] == "musique"
        assert row["status"] == "done" and row["em"] == 1
        assert row["vram_peak_mb"] == 4096
        assert row["finished_at"].endswith("Z")
        record_store.validate_record(row)


def test_log_events_carry_run_id_and_combo_and_model_loaded_once_per_model(env):
    data_root = env[0]
    run(env)
    events = log_events(data_root)
    for event in events:
        if event["event"] in ("combo_started", "combo_finished"):
            assert event["run_id"] == RUN_ID and event["combo"]
    loaded = [e["model"] for e in events if e["event"] == "model_loaded"]
    assert loaded == MODELS
    finished = [e for e in events if e["event"] == "combo_finished"]
    assert len(finished) == len(MODELS) * len(DATASETS) * len(CONDITIONS)
    assert finished[0]["counts"] == {"done": 2}


def test_manifest_has_contract_fields(env):
    data_root = env[0]
    probe = data_root / "runs" / RUN_ID / "probe_results.json"
    probe.parent.mkdir(parents=True)
    probe.write_text(json.dumps({"models": {"qwen3.5:9b": {
        "valid_tool_call_rate": 0.95, "n": 20, "mean_latency_ms": 321.0, "items": []}}}), encoding="utf-8")
    run(env)

    manifest = json.loads((data_root / "runs" / RUN_ID / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["datasets"] == DATASETS
    assert manifest["models"] == MODELS
    assert manifest["conditions"] == CONDITIONS
    assert manifest["questions_per_combo"] == 2
    assert manifest["model_digests"] == {m: f"sha-{m}" for m in MODELS}
    assert manifest["seed"] == 42
    assert manifest["git_commit"] == "abc123"
    assert manifest["laya_model_hash"] == "laya-hash"
    assert manifest["retrieval_device"] == "cuda"
    assert manifest["loop"]["max_hops"] == 4 and manifest["loop"]["timeout_s"] == 30
    assert manifest["loop"]["num_ctx"] == 8192 and manifest["loop"]["oom_retry_num_ctx"] == 4096
    assert manifest["probe_summary"]["qwen3.5:9b"] == {"valid_tool_call_rate": 0.95, "n": 20, "mean_latency_ms": 321.0}
    assert manifest["probe_summary"]["qwen3.5:0.8b"] is None


def test_questions_per_combo_without_limit_is_whole_sample(env):
    run(env, limit=None, models=MODELS[:1], datasets=DATASETS[:1], conditions=CONDITIONS[:1])
    manifest = json.loads((env[0] / "runs" / RUN_ID / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["questions_per_combo"] == 4
    assert len(records(env[0], "hotpotqa__qwen3.5-9b__recall_llm")) == 4


def test_resume_skips_done_questions_and_logs_resumed(env):
    data_root, _, log, _, _ = env
    run(env, limit=1)
    log.clear()
    run(env, limit=2)

    rows = records(data_root, "hotpotqa__qwen3.5-9b__recall_llm")
    assert [r["question_id"] for r in rows] == ["hotpotqa-q0", "hotpotqa-q1"]  # q0 沒有重跑
    resumed = [e for e in log_events(data_root) if e["event"] == "resumed"]
    assert len(resumed) == 16 and all(e["skipped"] == 1 for e in resumed)


def test_fully_done_model_is_not_loaded_again(env):
    data_root, _, log, _, _ = env
    run(env)
    log.clear()
    summary = run(env)
    assert summary["questions_run"] == 0
    assert not [entry for entry in log if entry[0] == "load"]
    assert not [e for e in log_events(data_root)[-20:] if e["event"] == "combo_started"]


def test_non_done_questions_are_retried_on_resume(env):
    data_root, _, _, state, _ = env
    state["fail"] = lambda model, num_ctx, n: ValueError("boom")
    run(env, models=MODELS[:1], datasets=DATASETS[:1], conditions=CONDITIONS[:1], limit=1)
    state["fail"] = None
    run(env, models=MODELS[:1], datasets=DATASETS[:1], conditions=CONDITIONS[:1], limit=1)
    assert [r["status"] for r in records(data_root, "hotpotqa__qwen3.5-9b__recall_llm")] == ["error", "done"]


def test_oom_is_retried_once_with_small_context(env):
    data_root, _, log, state, _ = env
    state["fail"] = lambda model, num_ctx, n: LLMOutOfMemoryError("CUDA out of memory") if num_ctx == 8192 else None
    run(env, models=MODELS[:1], datasets=DATASETS[:1], conditions=["recall_llm"], limit=1)

    rows = records(data_root, "hotpotqa__qwen3.5-9b__recall_llm")
    assert len(rows) == 1 and rows[0]["status"] == "done"
    assert ("client", "qwen3.5:9b", 4096) in log
    retry = [e for e in log_events(data_root) if e["event"] == "oom_retry"]
    assert len(retry) == 1 and retry[0]["question_id"] == "hotpotqa-q0"


def test_oom_twice_keeps_oom_record_and_continues(env):
    data_root, _, log, state, _ = env
    state["fail"] = lambda model, num_ctx, n: LLMOutOfMemoryError("CUDA out of memory")
    run(env, models=MODELS[:1], datasets=DATASETS[:1], conditions=["recall_llm"], limit=2)

    rows = records(data_root, "hotpotqa__qwen3.5-9b__recall_llm")
    assert [r["status"] for r in rows] == ["oom", "oom"]  # 每題只寫最後結果一行
    assert sum(1 for entry in log if entry == ("client", "qwen3.5:9b", 4096)) == 2  # 每題恰重試一次
    failed = [e for e in log_events(data_root) if e["event"] == "question_failed"]
    assert [e["errorCode"] for e in failed] == ["QUESTION_OOM", "QUESTION_OOM"]


def test_connection_failure_aborts_whole_run_with_error_log(env):
    data_root, _, _, state, _ = env
    state["fail"] = lambda model, num_ctx, n: LLMConnectionError("Ollama down")
    with pytest.raises(LLMConnectionError):
        run(env)

    assert not record_store.records_path(RUN_ID, "hotpotqa__qwen3.5-9b__recall_llm", data_root=data_root).exists()
    events = log_events(data_root)
    aborted = [e for e in events if e["event"] == "run_aborted"]
    assert len(aborted) == 1
    assert aborted[0]["level"] == "ERROR" and aborted[0]["errorCode"] == "OLLAMA_CONNECTION_FAILED"
    assert aborted[0]["combo"] == "hotpotqa__qwen3.5-9b__recall_llm"
    assert len([e for e in events if e["event"] == "combo_started"]) == 1  # 後續組別不再執行


def test_timeout_question_is_recorded_and_run_continues(env):
    data_root, backends, _, _, _ = env

    class SlowLLM(PolicyLLM):
        async def chat(self, messages, tools=None):
            import asyncio

            await asyncio.sleep(0.5)
            return await super().chat(messages, tools)

    backends.llm_factory = lambda model, num_ctx: SlowLLM(model, num_ctx, [])
    summary = run(env, models=MODELS[:1], datasets=DATASETS[:1], conditions=["recall_llm", "rag_llm"],
                  limit=1, timeout_s=0.2)
    assert summary["questions_run"] == 2
    for condition in ("recall_llm", "rag_llm"):
        rows = records(data_root, f"hotpotqa__qwen3.5-9b__{condition}")
        assert [r["status"] for r in rows] == ["timeout"]


def test_other_models_are_unloaded_before_loading_next(env):
    _, backends, log, _, _ = env
    backends.ollama.loaded = ["qwen3.5:27b"]
    run(env, models=MODELS[:1], datasets=DATASETS[:1], conditions=CONDITIONS[:1], limit=1)
    assert log.index(("unload", "qwen3.5:27b")) < log.index(("load", "qwen3.5:9b", 8192))


def test_missing_model_fails_before_any_work(env):
    data_root, backends, log, _, _ = env
    backends.ollama.digests_map = {"qwen3.5:9b": "sha"}
    with pytest.raises(benchmark_runner.RunError, match="qwen3.5:0.8b"):
        run(env)
    assert not (data_root / "runs" / RUN_ID / "records").exists()
    assert not [entry for entry in log if entry[0] == "load"]


def test_retrieval_and_laya_loaded_only_when_needed(env):
    _, _, _, state, _ = env
    run(env, models=MODELS[:1], conditions=["recall_llm"], limit=1)
    assert state["retrievers"] == [] and state["decisions"] == 0
    run(env, models=MODELS[:1], conditions=["rag_laya"], limit=1)
    assert sorted(state["retrievers"]) == sorted(DATASETS) and state["decisions"] == 1


def test_manifest_created_at_survives_restart(env):
    data_root = env[0]
    run(env, limit=1)
    path = data_root / "runs" / RUN_ID / "manifest.json"
    first = json.loads(path.read_text(encoding="utf-8"))
    run(env, limit=2)
    second = json.loads(path.read_text(encoding="utf-8"))
    assert second["created_at"] == first["created_at"]
    # 同一 run_id 的設定以第一次為準（見 _write_manifest）；本次參數記在 launches
    assert len(second["launches"]) == 2 and second["questions_per_combo"] == 1
    assert [launch["questions_per_combo"] for launch in second["launches"]] == [1, 2]


def test_rejects_unknown_dataset_or_condition(env):
    with pytest.raises(ValueError, match="nope"):
        run(env, datasets=["nope"])
    with pytest.raises(ValueError, match="rag_both"):
        run(env, conditions=["rag_both"])


@pytest.mark.parametrize("stage", ["digests", "load"])
def test_connection_failure_in_model_management_is_logged_and_aborts(env, stage):
    data_root, backends, _, _, _ = env

    def down(*args):
        raise LLMConnectionError("Ollama down")

    setattr(backends.ollama, stage, down)
    with pytest.raises(LLMConnectionError):
        run(env)
    aborted = [e for e in log_events(data_root) if e["event"] == "run_aborted"]
    assert len(aborted) == 1 and aborted[0]["errorCode"] == "OLLAMA_CONNECTION_FAILED"


# ---- 輸入邊界（Review r1 Finding）----
BAD_RUN_IDS = ["", ".", "..", "../escape", "a/b", "a\\b", "nul\0byte"]


@pytest.mark.parametrize("bad", BAD_RUN_IDS)
def test_rejects_run_id_that_is_not_a_single_path_segment_before_any_write(env, bad):
    data_root, _, log, _, _ = env
    before = sorted(p.relative_to(data_root.parent) for p in data_root.parent.rglob("*"))
    with pytest.raises(ValueError, match="run_id"):
        benchmark_runner.run(bad, DATASETS, MODELS, CONDITIONS, 1, data_root=data_root, backends=env[1], logger=env[4])
    after = sorted(p.relative_to(data_root.parent) for p in data_root.parent.rglob("*"))
    assert after == before and log == []  # 沒有任何檔案或 Ollama 動作


def test_rejects_absolute_run_id_without_writing_outside_data_root(env, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "escaped-run"
    with pytest.raises(ValueError, match="run_id"):
        benchmark_runner.run(str(outside), DATASETS, MODELS, CONDITIONS, 1, data_root=env[0], backends=env[1],
                             logger=env[4])
    assert not outside.exists()


@pytest.mark.parametrize("selector", ["datasets", "models", "conditions"])
def test_rejects_duplicate_selectors_before_scheduling(env, selector):
    data_root, _, log, _, _ = env
    values = {"datasets": ["hotpotqa", "musique", "hotpotqa"], "models": ["qwen3.5:9b", "qwen3.5:9b"],
              "conditions": ["recall_llm", "recall_llm"]}[selector]
    with pytest.raises(ValueError, match="重複"):
        run(env, **{selector: values})
    assert not (data_root / "runs" / RUN_ID).exists() and log == []


# ---- 同一 run_id 重啟時 manifest 合併而非覆寫（Ticket 12 執行中發現）----
ALL_MODELS = ["qwen3.5:27b", "qwen3.5:9b", "qwen3.5:4b", "qwen3.5:2b", "qwen3.5:0.8b"]
ALL_DATASETS = ["hotpotqa", "2wiki", "musique"]


def read_manifest(data_root):
    return json.loads((data_root / "runs" / RUN_ID / "manifest.json").read_text(encoding="utf-8"))


def test_restart_with_subset_keeps_union_of_grid_in_manifest(env):
    data_root, backends, _, _, _ = env
    backends.ollama.digests_map = {m: f"sha-{m}" for m in ALL_MODELS}
    probe = data_root / "runs" / RUN_ID / "probe_results.json"
    probe.parent.mkdir(parents=True)
    probe.write_text(json.dumps({"models": {m: {"valid_tool_call_rate": 1.0, "n": 20, "mean_latency_ms": 1.0}
                                            for m in ALL_MODELS}}), encoding="utf-8")
    run(env, models=ALL_MODELS, datasets=ALL_DATASETS, conditions=CONDITIONS, limit=1)
    run(env, models=["qwen3.5:0.8b", "qwen3.5:2b", "qwen3.5:4b", "qwen3.5:9b"], datasets=["2wiki", "musique"],
        conditions=["rag_llm"], limit=1)

    manifest = read_manifest(data_root)
    assert manifest["models"] == ALL_MODELS
    assert list(manifest["model_digests"]) == ALL_MODELS
    assert manifest["model_digests"] == {m: f"sha-{m}" for m in ALL_MODELS}
    assert manifest["datasets"] == ALL_DATASETS
    assert manifest["conditions"] == CONDITIONS
    assert len(manifest["launches"]) == 2
    assert manifest["launches"][1]["models"] == ["qwen3.5:0.8b", "qwen3.5:2b", "qwen3.5:4b", "qwen3.5:9b"]
    assert list(manifest["probe_summary"]) == ALL_MODELS and all(manifest["probe_summary"].values())


def test_restart_appends_new_selectors_in_stable_order(env):
    data_root, backends, _, _, _ = env
    backends.ollama.digests_map = {m: f"sha-{m}" for m in ALL_MODELS}
    run(env, models=["qwen3.5:4b", "qwen3.5:2b"], datasets=["musique"], conditions=["rag_laya"], limit=1)
    run(env, models=["qwen3.5:27b", "qwen3.5:2b"], datasets=["hotpotqa", "musique"],
        conditions=["recall_llm", "rag_laya"], limit=1)

    manifest = read_manifest(data_root)
    assert manifest["models"] == ["qwen3.5:4b", "qwen3.5:2b", "qwen3.5:27b"]
    assert list(manifest["model_digests"]) == ["qwen3.5:4b", "qwen3.5:2b", "qwen3.5:27b"]
    assert manifest["datasets"] == ["musique", "hotpotqa"]
    assert manifest["conditions"] == ["rag_laya", "recall_llm"]


def test_restart_recovers_models_recorded_only_in_earlier_launches(env):
    """r2 已覆寫過的 manifest（頂層少了某模型、launches 仍有）在下一次啟動時補回。"""
    data_root, backends, _, _, _ = env
    backends.ollama.digests_map = {m: f"sha-{m}" for m in ALL_MODELS}
    path = data_root / "runs" / RUN_ID / "manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "run_id": RUN_ID, "created_at": "2026-09-29T19:59:20.610Z", "datasets": ALL_DATASETS,
        "models": ["qwen3.5:0.8b", "qwen3.5:9b"], "conditions": CONDITIONS, "questions_per_combo": 1, "limit": 1,
        "model_digests": {"qwen3.5:0.8b": "sha-qwen3.5:0.8b", "qwen3.5:9b": "sha-qwen3.5:9b"},
        "launches": [{"started_at": "2026-09-29T19:59:20.610Z", "datasets": ALL_DATASETS,
                      "models": ["qwen3.5:27b", "qwen3.5:9b", "qwen3.5:0.8b"], "conditions": CONDITIONS, "limit": 1},
                     {"started_at": "2026-09-29T23:40:29.080Z", "datasets": ALL_DATASETS,
                      "models": ["qwen3.5:0.8b", "qwen3.5:9b"], "conditions": CONDITIONS, "limit": 1}],
    }), encoding="utf-8")
    run(env, models=["qwen3.5:27b"], datasets=["2wiki", "musique"], conditions=CONDITIONS, limit=1)

    manifest = read_manifest(data_root)
    assert manifest["models"] == ["qwen3.5:0.8b", "qwen3.5:9b", "qwen3.5:27b"]
    assert manifest["model_digests"] == {m: f"sha-{m}" for m in ["qwen3.5:0.8b", "qwen3.5:9b", "qwen3.5:27b"]}
    assert manifest["datasets"] == ALL_DATASETS and manifest["created_at"] == "2026-09-29T19:59:20.610Z"
    assert len(manifest["launches"]) == 3


def test_restart_with_different_settings_keeps_first_values_and_warns(env):
    data_root, backends, _, _, _ = env
    run(env, models=MODELS[:1], datasets=DATASETS[:1], conditions=CONDITIONS[:1], limit=1, timeout_s=30)
    backends.ollama.digests_map = {m: f"new-{m}" for m in MODELS}  # 模型被重新 pull
    run(env, models=MODELS[:1], datasets=DATASETS[:1], conditions=CONDITIONS[:1], limit=2, timeout_s=60)

    manifest = read_manifest(data_root)
    assert manifest["questions_per_combo"] == 1 and manifest["limit"] == 1
    assert manifest["loop"]["timeout_s"] == 30
    assert manifest["model_digests"] == {"qwen3.5:9b": "sha-qwen3.5:9b"}
    latest = manifest["launches"][-1]
    assert latest["questions_per_combo"] == 2 and latest["limit"] == 2 and latest["loop"]["timeout_s"] == 60
    assert latest["model_digests"] == {"qwen3.5:9b": "new-qwen3.5:9b"}
    warnings = [e for e in log_events(data_root) if e["event"] == "manifest_settings_kept"]
    assert len(warnings) == 1 and warnings[0]["level"] == "WARNING"
    assert {"questions_per_combo", "limit", "loop", "model_digests.qwen3.5:9b"} <= set(warnings[0]["keys"])


def test_restart_with_identical_settings_does_not_warn(env):
    run(env, limit=1)
    run(env, limit=1)
    assert not [e for e in log_events(env[0]) if e["event"] == "manifest_settings_kept"]


def test_limit_none_is_a_real_setting_but_laya_fields_fill_in(env):
    data_root = env[0]
    run(env, models=MODELS[:1], datasets=DATASETS[:1], conditions=["recall_llm"], limit=None)
    run(env, models=MODELS[:1], datasets=DATASETS[:1], conditions=["rag_laya"], limit=1)

    manifest = read_manifest(data_root)
    assert manifest["limit"] is None and manifest["questions_per_combo"] == 4  # 全部題目仍是本 run 的設定
    assert manifest["laya_model_hash"] == "laya-hash" and manifest["laya_device"] == "cuda"  # 第一次未用 Laya：補上
    warning = next(e for e in log_events(data_root) if e["event"] == "manifest_settings_kept")
    assert set(warning["keys"]) == {"limit", "questions_per_combo"}


# ---- 從舊 launches 補回的模型也要有 digest（第 3 輪 Review Finding）----
def write_r2_manifest_missing_27b(data_root):
    """r2 覆寫過的形狀：頂層只剩 0.8b，舊 launch 仍有 27b，且舊 launch 沒記 model_digests。"""
    path = data_root / "runs" / RUN_ID / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "run_id": RUN_ID, "created_at": "2026-09-29T19:59:20.610Z", "datasets": ["hotpotqa"],
        "models": ["qwen3.5:0.8b"], "conditions": ["recall_llm"], "questions_per_combo": 1, "limit": 1,
        "model_digests": {"qwen3.5:0.8b": "sha-qwen3.5:0.8b"},
        "launches": [{"started_at": "2026-09-29T19:59:20.610Z", "datasets": ["hotpotqa"],
                      "models": ["qwen3.5:27b", "qwen3.5:0.8b"], "conditions": ["recall_llm"], "limit": 1}],
    }), encoding="utf-8")


def restart_subset_without_27b(env):
    run(env, models=["qwen3.5:0.8b"], datasets=["hotpotqa"], conditions=["recall_llm"], limit=1)
    return read_manifest(env[0])


def test_recovered_model_not_in_this_launch_gets_digest_from_installed_models(env):
    data_root, backends, _, _, _ = env
    backends.ollama.digests_map = {m: f"sha-{m}" for m in ALL_MODELS}
    write_r2_manifest_missing_27b(data_root)
    manifest = restart_subset_without_27b(env)
    assert manifest["models"] == ["qwen3.5:0.8b", "qwen3.5:27b"]
    assert manifest["model_digests"] == {"qwen3.5:0.8b": "sha-qwen3.5:0.8b", "qwen3.5:27b": "sha-qwen3.5:27b"}
    assert not [e for e in log_events(data_root) if e["event"] == "manifest_digest_missing"]


def test_recovered_model_prefers_digest_recorded_by_this_runs_probe(env):
    data_root, backends, _, _, _ = env
    backends.ollama.digests_map = {m: f"sha-{m}" for m in ALL_MODELS}  # 目前安裝的 27b 可能已重新 pull
    write_r2_manifest_missing_27b(data_root)
    (data_root / "runs" / RUN_ID / "probe_results.json").write_text(json.dumps({"models": {
        "qwen3.5:27b": {"valid_tool_call_rate": 1.0, "n": 20, "mean_latency_ms": 1.0, "digest": "probe-27b"}}}),
        encoding="utf-8")
    manifest = restart_subset_without_27b(env)
    assert manifest["model_digests"]["qwen3.5:27b"] == "probe-27b"


def test_digest_that_cannot_be_recovered_is_null_with_warning(env):
    data_root, backends, _, _, _ = env
    backends.ollama.digests_map = {"qwen3.5:0.8b": "sha-qwen3.5:0.8b"}  # 27b 已不在本機、也沒有探針紀錄
    write_r2_manifest_missing_27b(data_root)
    manifest = restart_subset_without_27b(env)
    assert manifest["models"] == ["qwen3.5:0.8b", "qwen3.5:27b"]
    assert manifest["model_digests"] == {"qwen3.5:0.8b": "sha-qwen3.5:0.8b", "qwen3.5:27b": None}
    warnings = [e for e in log_events(data_root) if e["event"] == "manifest_digest_missing"]
    assert len(warnings) == 1 and warnings[0]["level"] == "WARNING"
    assert warnings[0]["models"] == ["qwen3.5:27b"] and warnings[0]["errorCode"] == "MANIFEST_DIGEST_MISSING"
