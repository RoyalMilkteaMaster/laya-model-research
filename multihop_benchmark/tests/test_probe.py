"""probe-models：固定 20 題工具呼叫探針的合法率、平均延遲與 probe_results.json（假 LLM／Ollama）。"""

from __future__ import annotations

import json

import pytest

from fakes import text, tool_reply
from multihop_benchmark.agent.protocols import InvalidToolCallError, LLMConnectionError, LLMResponse
from multihop_benchmark.benchmark_logger import get_logger
from multihop_benchmark.runs import probe

RUN_ID = "probe-unit"


class ScriptLLM:
    """每次呼叫依 `reply(index)` 回應或拋例外；記錄收到的工具名。"""

    def __init__(self, reply):
        self.reply, self.calls = reply, []

    async def chat(self, messages, tools=None):
        self.calls.append([t["function"]["name"] for t in tools or []])
        item = self.reply(len(self.calls) - 1)
        if isinstance(item, BaseException):
            raise item
        return item


class FakeOllama:
    def __init__(self):
        self.log = []

    def digests(self):
        return {"qwen3.5:4b": "sha-4b", "qwen3.5:2b": "sha-2b"}

    def running(self):
        return []

    def load(self, model, num_ctx):
        self.log.append(("load", model, num_ctx))

    def unload(self, model):
        self.log.append(("unload", model))


def good(latency=100.0):
    reply = tool_reply(probe.PROBE_TOOL_NAME, {"query": "capital of France"})
    reply.latency_ms = latency
    return reply


def test_twenty_fixed_questions():
    assert len(probe.PROBE_QUESTIONS) == 20 == len(set(probe.PROBE_QUESTIONS))


@pytest.mark.parametrize("response, valid", [
    (good(), True),
    (tool_reply("search", '{"query": "x"}'), True),  # JSON 字串參數可解析
    (tool_reply("search", {"query": "  "}), False),  # 空查詢
    (tool_reply("search", {"q": "x"}), False),  # 缺必要參數
    (tool_reply("lookup", {"query": "x"}), False),  # 未知工具
    (tool_reply("search", "not json"), False),
    (text("The capital is Paris."), False),  # 沒有工具呼叫
])
def test_valid_tool_call_rule(response, valid):
    assert probe.is_valid_tool_call(response) is valid


def test_probe_model_statistics():
    # 20 題：前 15 題合法（延遲 100）、3 題只回文字（延遲 200）、2 題伺服器無法解析工具呼叫
    def reply(i):
        if i < 15:
            return good(100.0)
        if i < 18:
            return LLMResponse(content="Paris", latency_ms=200.0)
        return InvalidToolCallError("error parsing tool call")

    llm = ScriptLLM(reply)
    result = probe.probe_model(llm)
    assert all(names == [probe.PROBE_TOOL_NAME] for names in llm.calls)
    assert result["n"] == 20 and result["valid"] == 15
    assert result["valid_tool_call_rate"] == pytest.approx(0.75)
    assert len(result["items"]) == 20
    assert result["items"][19]["valid"] is False and "parsing" in result["items"][19]["error"]
    # 平均延遲：合法與不合法都計入；例外題以實測時間（接近 0）計
    assert result["mean_latency_ms"] == pytest.approx((15 * 100 + 3 * 200 + result["items"][18]["latency_ms"]
                                                      + result["items"][19]["latency_ms"]) / 20, abs=0.01)


def test_probe_timeout_counts_as_invalid():
    import asyncio

    class Slow:
        async def chat(self, messages, tools=None):
            await asyncio.sleep(1)

    result = probe.probe_model(Slow(), questions=probe.PROBE_QUESTIONS[:2], timeout_s=0.05)
    assert result["valid"] == 0 and result["n"] == 2
    assert "逾時" in result["items"][0]["error"]


def test_connection_error_aborts_probe():
    with pytest.raises(LLMConnectionError):
        probe.probe_model(ScriptLLM(lambda i: LLMConnectionError("down")))


def test_run_probe_writes_results_and_logs(tmp_path):
    ollama = FakeOllama()
    clients = []

    def llm_factory(model, num_ctx):
        clients.append((model, num_ctx))
        return ScriptLLM(lambda i: good(50.0) if model == "qwen3.5:4b" or i % 2 else text("no tool"))

    logger = get_logger("probe-models", log_dir=tmp_path / "logs")
    path = probe.run_probe(RUN_ID, ["qwen3.5:4b", "qwen3.5:2b"], data_root=tmp_path, llm_factory=llm_factory,
                           ollama=ollama, logger=logger)

    assert path == tmp_path / "runs" / RUN_ID / "probe_results.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data["models"]) == {"qwen3.5:4b", "qwen3.5:2b"}
    four, two = data["models"]["qwen3.5:4b"], data["models"]["qwen3.5:2b"]
    assert (four["valid_tool_call_rate"], four["n"], four["mean_latency_ms"]) == (1.0, 20, 50.0)
    assert two["valid_tool_call_rate"] == 0.5 and two["digest"] == "sha-2b"
    assert ollama.log == [("load", "qwen3.5:4b", 8192), ("unload", "qwen3.5:4b"),
                          ("load", "qwen3.5:2b", 8192), ("unload", "qwen3.5:2b")]
    assert clients == [("qwen3.5:4b", 8192), ("qwen3.5:2b", 8192)]

    events = [json.loads(line) for line in (tmp_path / "logs" / "benchmark.jsonl").read_text().splitlines()]
    started = [e["model"] for e in events if e["event"] == "probe_started"]
    finished = [e for e in events if e["event"] == "probe_finished"]
    assert started == ["qwen3.5:4b", "qwen3.5:2b"]
    assert [(e["model"], e["valid_tool_call_rate"]) for e in finished] == [("qwen3.5:4b", 1.0), ("qwen3.5:2b", 0.5)]
    assert all(e["run_id"] == RUN_ID for e in finished)


def test_run_probe_merges_with_existing_models(tmp_path):
    path = tmp_path / "runs" / RUN_ID / "probe_results.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"models": {"qwen3.5:27b": {"valid_tool_call_rate": 1.0, "n": 20,
                                                             "mean_latency_ms": 9.0}}}), encoding="utf-8")
    probe.run_probe(RUN_ID, ["qwen3.5:4b"], data_root=tmp_path, llm_factory=lambda m, c: ScriptLLM(lambda i: good()),
                    ollama=FakeOllama(), logger=get_logger("probe-models", log_dir=tmp_path / "logs"))
    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data["models"]) == {"qwen3.5:27b", "qwen3.5:4b"}


def test_run_probe_rejects_missing_model(tmp_path):
    with pytest.raises(probe.ProbeError, match="qwen3.5:0.8b"):
        probe.run_probe(RUN_ID, ["qwen3.5:0.8b"], data_root=tmp_path, llm_factory=lambda m, c: None,
                        ollama=FakeOllama(), logger=get_logger("probe-models", log_dir=tmp_path / "logs"))


# ---- 輸入邊界（Review r1 Finding）----
@pytest.mark.parametrize("bad", ["", ".", "..", "../escape", "a/b", "a\\b", "nul\0byte"])
def test_run_probe_rejects_bad_run_id_before_any_write(tmp_path, bad):
    ollama = FakeOllama()
    with pytest.raises(ValueError, match="run_id"):
        probe.run_probe(bad, ["qwen3.5:4b"], data_root=tmp_path / "data", llm_factory=lambda m, c: None,
                        ollama=ollama, logger=get_logger("probe-models", log_dir=tmp_path / "logs"))
    assert not (tmp_path / "data").exists() and ollama.log == []  # logs/ 由測試建立 logger 時產生


def test_run_probe_rejects_absolute_run_id(tmp_path):
    outside = tmp_path / "outside" / "escaped-probe"
    with pytest.raises(ValueError, match="run_id"):
        probe.run_probe(str(outside), ["qwen3.5:4b"], data_root=tmp_path / "data", llm_factory=lambda m, c: None,
                        ollama=FakeOllama(), logger=get_logger("probe-models", log_dir=tmp_path / "logs"))
    assert not (tmp_path / "outside").exists()


def test_run_probe_rejects_duplicate_models(tmp_path):
    ollama = FakeOllama()
    with pytest.raises(ValueError, match="重複"):
        probe.run_probe("p", ["qwen3.5:4b", "qwen3.5:4b"], data_root=tmp_path / "data", llm_factory=lambda m, c: None,
                        ollama=ollama, logger=get_logger("probe-models", log_dir=tmp_path / "logs"))
    assert ollama.log == [] and not (tmp_path / "data").exists()
