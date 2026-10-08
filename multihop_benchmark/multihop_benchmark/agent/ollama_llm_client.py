"""`LLMClient` 的 Ollama 實作（Spike 回退結論：直接用 ollama Python client）。

固定 `think=False`、`options={num_ctx, temperature: 0, seed}`；工具呼叫用 Ollama 原生 `tools` 參數；
token 取 `prompt_eval_count`／`eval_count`（Ollama 省略時記 0）；`latency_ms` 為整次 `chat()` 的實際經過時間，
包含連線失敗的嘗試與重試等待。

錯誤分類：連線失敗（`ConnectionError`、httpx 傳輸錯誤）以 `retry_delays_s` 間隔重試 3 次，仍失敗拋
`LLMConnectionError`；伺服器回報記憶體不足拋 `LLMOutOfMemoryError`；無法解析模型工具呼叫拋
`InvalidToolCallError`；其他 `ollama.ResponseError` 原樣拋出。

每次 `chat()` 各建一個 `ollama.AsyncClient`，結束時（含例外與逾時取消）一律關閉：client 綁定建立時的事件迴圈，
而 `run_question` 每題 `asyncio.run` 一次；逐次建立可避免跨迴圈重用，也不需要呼叫端管理關閉。對本機 Ollama
而言，每次多一次 TCP 連線的成本遠小於模型推論時間。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from typing import Any

import httpx
import ollama

from multihop_benchmark.agent.protocols import (
    InvalidToolCallError,
    LLMConnectionError,
    LLMOutOfMemoryError,
    LLMResponse,
    ToolCall,
)

_OOM_MARKERS = ("out of memory", "cudamalloc", "more system memory", "insufficient memory")


def _classify(exc: ollama.ResponseError) -> Exception:
    message = str(exc.error).lower()
    if any(marker in message for marker in _OOM_MARKERS):
        return LLMOutOfMemoryError(str(exc.error))
    if "tool call" in message and ("pars" in message or "invalid" in message):
        return InvalidToolCallError(str(exc.error))
    return exc


class OllamaLLMClient:
    def __init__(
        self,
        model: str,
        host: str | None = None,
        num_ctx: int = 8192,
        seed: int = 42,
        *,
        keep_alive: str | float | None = None,
        retry_delays_s: Sequence[float] = (1.0, 2.0, 4.0),
        client_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if host is None:
            from multihop_benchmark.settings import load_settings

            host = load_settings().ollama_host
        self.model = model
        self.host = host
        self.num_ctx = num_ctx
        self.seed = seed
        self.keep_alive = keep_alive
        self.retry_delays_s = tuple(retry_delays_s)
        self._client_factory = client_factory or (lambda h: ollama.AsyncClient(host=h))

    @property
    def options(self) -> dict[str, Any]:
        return {"num_ctx": self.num_ctx, "temperature": 0, "seed": self.seed}

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        request: dict[str, Any] = {
            "model": self.model, "messages": messages, "think": False, "options": self.options,
        }
        if tools:
            request["tools"] = tools
        if self.keep_alive is not None:
            request["keep_alive"] = self.keep_alive

        started = time.perf_counter()
        client = self._client_factory(self.host)
        try:
            response = await self._request_with_retry(client, request)
        finally:
            await client.close()
        latency_ms = round((time.perf_counter() - started) * 1000, 3)

        message = response.message
        return LLMResponse(
            content=message.content or "",
            tool_calls=[ToolCall(call.function.name, dict(call.function.arguments))
                        for call in message.tool_calls or []],
            prompt_tokens=int(response.prompt_eval_count or 0),
            output_tokens=int(response.eval_count or 0),
            latency_ms=latency_ms,
        )

    async def _request_with_retry(self, client: Any, request: dict[str, Any]) -> Any:
        for attempt in range(len(self.retry_delays_s) + 1):
            try:
                return await client.chat(**request)
            except ollama.ResponseError as exc:
                raise _classify(exc) from exc
            except (ConnectionError, httpx.TransportError) as exc:
                if attempt == len(self.retry_delays_s):
                    raise LLMConnectionError(
                        f"Ollama {self.host} 連線失敗，已重試 {attempt} 次：{type(exc).__name__}: {exc}") from exc
                await asyncio.sleep(self.retry_delays_s[attempt])
        raise AssertionError("unreachable")
