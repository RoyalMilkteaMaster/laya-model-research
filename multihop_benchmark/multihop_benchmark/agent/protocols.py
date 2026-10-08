"""多跳迴圈的三個注入接縫：`LLMClient`、`DecisionModel`、`Retriever`，以及 LLM 回應型別與例外。

- `LLMClient.chat` 是 async（逾時以 `asyncio.wait_for` 取消）；正式實作為 `OllamaLLMClient`。
- `DecisionModel.decide` 與 `Retriever.retrieve` 是同步呼叫（GPU 推論），迴圈在每題專用的單執行緒 executor
  執行；逾時時等進行中的呼叫自然結束後才回傳（見 `multihop_agent_loop` 模組說明）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict, runtime_checkable


@dataclass(slots=True)
class ToolCall:
    """一次工具呼叫。`arguments` 通常是 dict；模型輸出無法解析時可為原始字串。"""

    name: str
    arguments: dict[str, Any] | str


@dataclass(slots=True)
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


class Passage(TypedDict):
    pid: str
    title: str
    text: str


class Decision(TypedDict):
    """`next_action` ∈ {A, B, C}（直接作答／再拆子問題／換查詢）；`remaining` ∈ {0, 1, 2}（2 代表 ≥ 2）。"""

    sufficient_p: float
    next_action: str
    remaining: int
    latency_ms: float


class DecisionState(TypedDict):
    """傳給 `DecisionModel.decide` 的原始 state；實作端以 `decision_state_builder.build_state(**state)` 組裝文字。

    `evidence` 為目前累積的 evidence（`{pid, title, text}`；recall 條件的 title 是子問題、text 是回憶內容）。
    """

    question: str
    sub_questions: list[str]
    evidence: list[Passage]


@runtime_checkable
class LLMClient(Protocol):
    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        """送出一次對話；`tools` 為 Ollama／OpenAI function 格式。

        必須回報該次呼叫的 prompt／output token 與延遲。連線失敗重試後仍失敗拋 `LLMConnectionError`；
        顯示卡記憶體不足拋 `LLMOutOfMemoryError`；伺服器無法解析模型的工具呼叫拋 `InvalidToolCallError`。
        """
        ...


@runtime_checkable
class DecisionModel(Protocol):
    def decide(self, state: DecisionState) -> Decision: ...


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str) -> list[Passage]:
        """回傳依相關度排序的段落（dense top-20 → rerank）；迴圈只取前 3 段。"""
        ...


class InvalidToolCallError(Exception):
    """模型的工具呼叫無效：未知工具名、參數不合法、非 JSON，或需要工具時沒有呼叫。"""


class LLMConnectionError(ConnectionError):
    """Ollama 連線重試用盡；呼叫端（benchmark_runner）據此中止整個 run。"""


class LLMOutOfMemoryError(RuntimeError):
    """推論時顯示卡記憶體不足。"""
