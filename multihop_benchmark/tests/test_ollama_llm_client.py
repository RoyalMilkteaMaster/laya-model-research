"""OllamaLLMClient：以假的 ollama AsyncClient 驗證呼叫參數、回應轉換、重試與錯誤分類（不需 Ollama）。"""

from __future__ import annotations

import asyncio

import httpx
import ollama
import pytest

from multihop_benchmark.agent.ollama_llm_client import OllamaLLMClient
from multihop_benchmark.agent.protocols import (
    InvalidToolCallError,
    LLMClient,
    LLMConnectionError,
    LLMOutOfMemoryError,
    ToolCall,
)


def ok_response(content="Tallinn", tool_calls=None, prompt_eval_count=31, eval_count=4):
    return ollama.ChatResponse(
        model="qwen3.5:4b",
        message=ollama.Message(role="assistant", content=content, tool_calls=tool_calls),
        prompt_eval_count=prompt_eval_count,
        eval_count=eval_count,
    )


class FakeAsyncClient:
    instances: list["FakeAsyncClient"] = []

    def __init__(self, outcomes, host=None, delay_s=0.0):
        self.outcomes = outcomes
        self.host = host
        self.delay_s = delay_s
        self.requests = []
        self.closed = False
        FakeAsyncClient.instances.append(self)

    async def chat(self, **kwargs):
        assert not self.closed, "已關閉的 client 不可再使用"
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        return outcome

    async def close(self):
        self.closed = True

    @classmethod
    def all_requests(cls):
        return [r for instance in cls.instances for r in instance.requests]


def make_client(outcomes, *, delay_s=0.0, **kwargs):
    FakeAsyncClient.instances = []
    kwargs.setdefault("retry_delays_s", (0, 0, 0))
    return OllamaLLMClient(
        "qwen3.5:4b", "http://ollama.test:11434",
        client_factory=lambda host: FakeAsyncClient(outcomes, host, delay_s), **kwargs,
    )


def test_chat_sends_fixed_options_and_maps_response():
    tool_call = ollama.Message.ToolCall(function=ollama.Message.ToolCall.Function(
        name="ask_sub_question", arguments={"sub_question": "Who?"}))
    client = make_client([ok_response(content="", tool_calls=[tool_call])], num_ctx=4096, seed=7)
    tools = [{"type": "function", "function": {"name": "ask_sub_question"}}]

    response = asyncio.run(client.chat([{"role": "user", "content": "hi"}], tools=tools))

    assert isinstance(client, LLMClient)
    request = FakeAsyncClient.all_requests()[0]
    assert FakeAsyncClient.instances[0].host == "http://ollama.test:11434"
    assert request["model"] == "qwen3.5:4b" and request["think"] is False and request["tools"] == tools
    assert request["options"] == {"num_ctx": 4096, "temperature": 0, "seed": 7}
    assert response.tool_calls == [ToolCall("ask_sub_question", {"sub_question": "Who?"})]
    assert (response.content, response.prompt_tokens, response.output_tokens) == ("", 31, 4)
    assert response.latency_ms >= 0


def test_defaults_are_num_ctx_8192_seed_42_and_missing_counts_are_zero():
    client = make_client([ok_response(prompt_eval_count=None, eval_count=None)])
    response = asyncio.run(client.chat([{"role": "user", "content": "hi"}]))
    request = FakeAsyncClient.all_requests()[0]
    assert request["options"] == {"num_ctx": 8192, "temperature": 0, "seed": 42}
    assert "tools" not in request or request["tools"] is None
    assert (response.content, response.prompt_tokens, response.output_tokens) == ("Tallinn", 0, 0)


def test_connection_failure_retries_three_times_then_raises():
    outcomes = [ConnectionError("refused")] * 4
    client = make_client(outcomes)
    with pytest.raises(LLMConnectionError):
        asyncio.run(client.chat([{"role": "user", "content": "hi"}]))
    assert len(FakeAsyncClient.all_requests()) == 4  # 1 次 + 重試 3 次


def test_connection_recovers_within_retries():
    client = make_client([httpx.ConnectError("refused"), ConnectionError("refused"), ok_response()])
    response = asyncio.run(client.chat([{"role": "user", "content": "hi"}]))
    assert response.content == "Tallinn"
    assert len(FakeAsyncClient.all_requests()) == 3


def test_latency_covers_failed_attempts_and_backoff():
    client = make_client([ConnectionError("refused"), ok_response()], delay_s=0.02, retry_delays_s=(0.08,))
    response = asyncio.run(client.chat([{"role": "user", "content": "hi"}]))
    assert response.latency_ms >= 100  # 80 ms backoff + 20 ms 成功請求


@pytest.mark.parametrize("message,expected", [
    ("model 'qwen3.5:99b' not found", None),
    ("model requires more system memory (20 GiB) than is available (12 GiB)", LLMOutOfMemoryError),
    ("llama runner process has terminated: cudaMalloc failed: out of memory", LLMOutOfMemoryError),
    ("CUDA error: out of memory", LLMOutOfMemoryError),
    ("error parsing tool call: raw='{\"name\": ...' err=unexpected end of JSON input", InvalidToolCallError),
])
def test_response_errors_are_classified(message, expected):
    client = make_client([ollama.ResponseError(message, 500)])
    with pytest.raises(expected or ollama.ResponseError):
        asyncio.run(client.chat([{"role": "user", "content": "hi"}]))


def test_every_created_client_is_closed_across_event_loops_and_errors():
    client = make_client([ok_response(), ok_response(), ollama.ResponseError("CUDA error: out of memory", 500),
                          ConnectionError("refused"), ConnectionError("refused")], retry_delays_s=(0,))
    asyncio.run(client.chat([{"role": "user", "content": "a"}]))
    asyncio.run(client.chat([{"role": "user", "content": "b"}]))
    with pytest.raises(LLMOutOfMemoryError):
        asyncio.run(client.chat([{"role": "user", "content": "c"}]))
    with pytest.raises(LLMConnectionError):
        asyncio.run(client.chat([{"role": "user", "content": "d"}]))
    assert FakeAsyncClient.instances and all(i.closed for i in FakeAsyncClient.instances)


def test_client_closed_when_request_is_cancelled_by_timeout():
    client = make_client([ok_response()], delay_s=1.0)

    async def main():
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(client.chat([{"role": "user", "content": "a"}]), 0.05)

    asyncio.run(main())
    assert all(i.closed for i in FakeAsyncClient.instances)
