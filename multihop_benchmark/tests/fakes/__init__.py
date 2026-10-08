"""可腳本化的假 `LLMClient`／`DecisionModel`／`Retriever`，供迴圈與執行器測試使用。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable, Mapping
from typing import Any

from multihop_benchmark.agent.protocols import LLMResponse, ToolCall


def tool_reply(name: str, arguments: dict[str, Any] | str, *, prompt_tokens: int = 10, output_tokens: int = 5) -> LLMResponse:
    return LLMResponse(content="", tool_calls=[ToolCall(name, arguments)],
                       prompt_tokens=prompt_tokens, output_tokens=output_tokens, latency_ms=1.0)


def sub_q(text: str, **tokens: int) -> LLMResponse:
    return tool_reply("ask_sub_question", {"sub_question": text}, **tokens)


def final(answer: str, **tokens: int) -> LLMResponse:
    return tool_reply("final_answer", {"answer": answer}, **tokens)


def text(content: str, *, prompt_tokens: int = 10, output_tokens: int = 5) -> LLMResponse:
    return LLMResponse(content=content, prompt_tokens=prompt_tokens, output_tokens=output_tokens, latency_ms=1.0)


class FakeLLM:
    """依序回傳腳本中的回應；項目為 `LLMResponse` 或要拋出的例外。每次呼叫前睡 `sleep_s` 秒。"""

    def __init__(self, script: Iterable[LLMResponse | BaseException], *, sleep_s: float = 0.0) -> None:
        self.script = list(script)
        self.sleep_s = sleep_s
        self.calls: list[dict[str, Any]] = []
        self.returned: list[LLMResponse] = []

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        self.calls.append({"messages": messages, "tools": [t["function"]["name"] for t in tools or []]})
        if self.sleep_s:
            await asyncio.sleep(self.sleep_s)
        if not self.script:
            raise AssertionError(f"FakeLLM 腳本用完（第 {len(self.calls)} 次呼叫）")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        self.returned.append(item)
        return item

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.returned)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(r.prompt_tokens for r in self.returned)


class FakeDecision:
    """依序回傳固定決策；序列用完後重複最後一個。`completed` 為已執行完畢的呼叫數（睡眠結束才加 1）。"""

    def __init__(self, decisions: Iterable[Mapping[str, Any]], *, sleep_s: float = 0.0) -> None:
        self.decisions = [dict(d) for d in decisions]
        self.sleep_s = sleep_s
        self.states: list[dict[str, Any]] = []
        self.completed = 0

    def decide(self, state: Mapping[str, Any]) -> dict[str, Any]:
        self.states.append({"question": state["question"], "sub_questions": list(state["sub_questions"]),
                            "evidence": list(state["evidence"])})
        if self.sleep_s:
            time.sleep(self.sleep_s)
        index = min(len(self.states) - 1, len(self.decisions) - 1)
        self.completed += 1
        return dict(self.decisions[index])


def decision(sufficient_p: float, next_action: Any = "B", remaining: Any = 1, latency_ms: float = 3.0) -> dict[str, Any]:
    return {"sufficient_p": sufficient_p, "next_action": next_action, "remaining": remaining, "latency_ms": latency_ms}


class FakeRetriever:
    """依序回傳段落清單；序列用完後回空清單。`completed` 為已執行完畢的呼叫數（睡眠結束才加 1）。"""

    def __init__(self, results: Iterable[list[Mapping[str, str]]], *, sleep_s: float = 0.0) -> None:
        self.results = [list(r) for r in results]
        self.sleep_s = sleep_s
        self.queries: list[str] = []
        self.completed = 0

    def retrieve(self, query: str) -> list[dict[str, str]]:
        self.queries.append(query)
        if self.sleep_s:
            time.sleep(self.sleep_s)
        self.completed += 1
        return [dict(p) for p in self.results.pop(0)] if self.results else []


def passage(pid: str) -> dict[str, str]:
    return {"pid": pid, "title": f"Title {pid}", "text": f"Text of {pid}."}
