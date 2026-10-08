"""function calling 探針：每個模型固定 20 題，要求呼叫一個簡單工具 `search(query)`。

合法呼叫 = 回應的第一個工具呼叫名稱為 `search`，參數是物件（或可解析為物件的 JSON 字串）且 `query` 為非空字串。
只回文字、未知工具、參數不合法、伺服器無法解析工具呼叫、記憶體不足、逾時與其他 Ollama 錯誤一律計為不合法；
`LLMConnectionError`（client 已重試 3 次）往外拋、中止探針。

每題延遲取 client 回報的 `latency_ms`；拋例外的題以實測經過時間計，平均延遲含所有 20 題。
模型探針前以正式 `num_ctx` 預載（不把載入時間算進第一題），結束後卸載。

輸出 Data Root `runs/<run_id>/probe_results.json`，形狀
`{"models": {tag: {valid_tool_call_rate, n, valid, mean_latency_ms, digest, probed_at, items[]}}, ...}`；
已存在時以模型為單位合併（只覆寫本次探針的模型）。
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from multihop_benchmark.agent.protocols import LLMClient, LLMConnectionError, LLMResponse
from multihop_benchmark.benchmark_logger import BenchmarkLogger
from multihop_benchmark.runs.benchmark_runner import NUM_CTX, SEED, require_unique, utc_now_iso, validate_run_id

PROBE_TOOL_NAME = "search"
PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": PROBE_TOOL_NAME,
        "description": "Search an encyclopedia for passages about a single fact.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "A short search query."}},
            "required": ["query"],
        },
    },
}
SYSTEM_PROMPT = "You are a research assistant. Use the search tool to look up facts before answering."
PROBE_QUESTIONS = (
    "What is the capital city of Australia?",
    "Who wrote the novel Pride and Prejudice?",
    "In which year did the Berlin Wall fall?",
    "What is the tallest mountain in Africa?",
    "Who painted The Starry Night?",
    "Which river flows through Baghdad?",
    "What is the chemical symbol for tungsten?",
    "Who was the first person to walk on the Moon?",
    "Which country hosted the 2016 Summer Olympics?",
    "What is the official language of Brazil?",
    "Who composed The Four Seasons?",
    "In which city was Albert Einstein born?",
    "What is the largest desert in Asia?",
    "Which company developed the Windows operating system?",
    "Who directed the film Spirited Away?",
    "What is the currency of Switzerland?",
    "Which planet has the most known moons?",
    "Who founded the city of Alexandria?",
    "What is the national animal of Scotland?",
    "Which university did Isaac Newton attend?",
)
PROBE_TIMEOUT_S = 120.0


class ProbeError(RuntimeError):
    """探針無法開始（例如模型未安裝）。"""


def is_valid_tool_call(response: LLMResponse) -> bool:
    if not response.tool_calls:
        return False
    call = response.tool_calls[0]
    arguments: Any = call.arguments
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return False
    query = arguments.get("query") if isinstance(arguments, dict) else None
    return call.name == PROBE_TOOL_NAME and isinstance(query, str) and bool(query.strip())


async def _ask(llm: LLMClient, question: str, timeout_s: float) -> dict[str, Any]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{question}\nCall the {PROBE_TOOL_NAME} tool to find the answer."}]
    started = time.perf_counter()
    try:
        response = await asyncio.wait_for(llm.chat(messages, tools=[PROBE_TOOL]), timeout_s)
    except LLMConnectionError:
        raise
    except TimeoutError:
        error = f"逾時（>{timeout_s} s）"
    except Exception as exc:  # noqa: BLE001 - 無效工具呼叫、OOM、其他 Ollama 錯誤皆計為不合法
        error = f"{type(exc).__name__}: {exc}"
    else:
        calls = [{"name": c.name, "arguments": c.arguments} for c in response.tool_calls]
        return {"valid": is_valid_tool_call(response), "latency_ms": round(float(response.latency_ms), 3),
                "tool_calls": calls, "content": response.content[:200]}
    return {"valid": False, "latency_ms": round((time.perf_counter() - started) * 1000, 3), "error": error}


def probe_model(llm: LLMClient, *, questions: Sequence[str] = PROBE_QUESTIONS,
                timeout_s: float = PROBE_TIMEOUT_S) -> dict[str, Any]:
    """對一個模型跑探針題，回傳 `{valid_tool_call_rate, n, valid, mean_latency_ms, items}`。"""
    items = [{"question": q, **asyncio.run(_ask(llm, q, timeout_s))} for q in questions]
    valid = sum(item["valid"] for item in items)
    return {
        "valid_tool_call_rate": valid / len(items),
        "n": len(items),
        "valid": valid,
        "mean_latency_ms": round(sum(item["latency_ms"] for item in items) / len(items), 3),
        "items": items,
    }


def run_probe(
    run_id: str,
    models: Sequence[str],
    *,
    data_root: Path,
    llm_factory: Callable[[str, int], LLMClient],
    ollama: Any,
    logger: BenchmarkLogger,
) -> Path:
    """依序探針每個模型並合併寫入 `runs/<run_id>/probe_results.json`；回傳檔案路徑。"""
    validate_run_id(run_id)
    require_unique("models", models)
    digests = ollama.digests()
    missing = [model for model in models if model not in digests]
    if missing:
        message = f"Ollama 未安裝模型：{', '.join(missing)}（請先 ollama pull）"
        logger.error(message, event="probe_failed", run_id=run_id, errorCode="MODEL_MISSING", missing=missing)
        raise ProbeError(message)

    path = Path(data_root) / "runs" / run_id / "probe_results.json"
    results: dict[str, Any] = {}
    if path.is_file():
        try:
            results = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            results = {}
    results.setdefault("models", {})

    for model in models:
        for other in ollama.running():
            if other != model:
                ollama.unload(other)
        logger.info(f"探針開始 {model}：{len(PROBE_QUESTIONS)} 題", event="probe_started", run_id=run_id,
                    model=model, n=len(PROBE_QUESTIONS))
        ollama.load(model, NUM_CTX)
        try:
            result = probe_model(llm_factory(model, NUM_CTX))
        finally:
            ollama.unload(model)
        results["models"][model] = {**result, "digest": digests[model], "probed_at": utc_now_iso()}
        results.update({
            "run_id": run_id, "updated_at": utc_now_iso(), "tool": PROBE_TOOL_NAME,
            "questions": list(PROBE_QUESTIONS), "num_ctx": NUM_CTX, "temperature": 0, "think": False, "seed": SEED,
        })
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(".probe_results.json.tmp")
        tmp.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)  # 每個模型完成就寫入，中斷時保留已完成的模型
        logger.info(f"探針完成 {model}：合法率 {result['valid_tool_call_rate']:.2f}", event="probe_finished",
                    run_id=run_id, model=model, n=result["n"], valid=result["valid"],
                    valid_tool_call_rate=result["valid_tool_call_rate"], mean_latency_ms=result["mean_latency_ms"])
    return path
