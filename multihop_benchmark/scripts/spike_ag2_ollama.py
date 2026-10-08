"""Ticket 03 spike：AG2 1.x + Ollama 五項證據與 27B＋檢索堆疊 VRAM 共存驗證。

臨時程式，不屬於正式模組；依賴裝在獨立 spike venv（不動共用 venv）：

    uv venv ~/laya_runtime/environments/spike --python 3.12
    uv pip install --python ~/laya_runtime/environments/spike/bin/python \
        "ag2[ollama]==1.1.0" ollama sentence-transformers laya torch

執行（於 Code Root，Ollama 已在 tmux `laya-ollama` 執行）：

    source multihop_benchmark/scripts/env.sh
    ~/laya_runtime/environments/spike/bin/python multihop_benchmark/scripts/spike_ag2_ollama.py all

子命令：`agent`（qwen3.5:4b 五項證據）、`vram`（27B＋BGE-M3＋reranker＋Laya 同時上 GPU）、`all`。
輸出到 `$PROJECT_DATA_ROOT/runs/spike/`：`spike_results.json`（固定結構，只跑單一子命令時保留
另一部分的舊結果）與 `trace_<scenario>.jsonl`（一題一行）。可重跑，每次覆寫。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ollama
from ag2 import Agent, Middleware, ToolResult, observer, tool
from ag2 import __version__ as AG2_VERSION
from ag2.config import OllamaConfig
from ag2.events import ModelReasoning, ToolResultEvent
from ag2.middleware import BaseMiddleware

SMALL_MODEL = "qwen3.5:4b"
LARGE_MODEL = "qwen3.5:27b"
NUM_CTX = 8192
TEMPERATURE = 0.0
SEED = 42
MAIN_HOP_LIMIT = 4
MAIN_TIMEOUT_S = 300.0
TEST_HOP_LIMIT = 2
TEST_TIMEOUT_S = 5.0
TIMEOUT_SEARCH_DELAY_S = 3.0  # 模擬慢速檢索，讓 3 跳題目必定超過 5 秒
THINKING_REPEATS = 3
SCHEMA_VERSION = 1
EVIDENCE_KEYS = ("step_trace", "token_counts", "hop_limit_abort", "timeout_abort", "thinking_off")

SYSTEM_PROMPT = (
    "You answer multi-hop questions about fictional entities. You know nothing about them. "
    "Use the search tool to look up facts; each search returns one passage, so search once per fact. "
    "When you know the answer, call final_answer with a short answer (a name, place or number). "
    "Never answer from memory and never reply with plain text instead of calling final_answer."
)

# 固定假段落：search 第 i 次呼叫回第 i 段，用完回 "No further results."
QUESTIONS: list[dict[str, Any]] = [
    {
        "id": "spike_q1",
        "question": "In which city was the founder of Zorvex Labs born?",
        "passages": [
            "Zorvex Labs is a robotics company founded in 2011 by Mira Oduya.",
            "Mira Oduya was born in Tallinn, Estonia, in 1979.",
        ],
        "answer": "Tallinn",
    },
    {
        "id": "spike_q2",
        "question": "What river flows through the hometown of the painter of 'Blue Lantern at Dusk'?",
        "passages": [
            "'Blue Lantern at Dusk' is an oil painting by Henrik Salvo.",
            "Henrik Salvo grew up in the town of Varnholm.",
            "Varnholm lies on the banks of the river Esk.",
        ],
        "answer": "Esk",
    },
    {
        "id": "spike_q3",
        "question": "Which instrument does the lead singer of the band Copper Finch play?",
        "passages": [
            "Copper Finch is an indie band whose lead singer is Talia Brenner.",
            "Talia Brenner plays the cello on most Copper Finch recordings.",
        ],
        "answer": "cello",
    },
    {
        "id": "spike_q4",
        "question": "In what year was the university attended by the author of 'The Glass Orchard' founded?",
        "passages": [
            "'The Glass Orchard' is a novel written by Idris Kallan.",
            "Idris Kallan studied literature at Morrow University.",
            "Morrow University was founded in 1872.",
        ],
        "answer": "1872",
    },
    {
        "id": "spike_q5",
        "question": "What is the capital of the country where the inventor of the Pellar engine was born?",
        "passages": [
            "The Pellar engine was invented by Oskar Venn.",
            "Oskar Venn was born in the country of Freland.",
            "The capital of Freland is Ostrava Nova.",
        ],
        "answer": "Ostrava Nova",
    },
]


# ---------------------------------------------------------------------------
# AG2 設定：OllamaConfig 本身沒有 think／num_ctx 欄位，以 ModelConfig 擴充點補上
# ---------------------------------------------------------------------------


class _ChatProxy:
    """包住 ollama.AsyncClient，注入 think 並記下 Ollama 原始回應的計數與耗時。"""

    def __init__(self, inner: ollama.AsyncClient, think: bool | None, raw_log: list[dict[str, Any]] | None) -> None:
        self._inner = inner
        self._think = think
        self._raw_log = raw_log

    async def chat(self, **kwargs: Any) -> Any:
        if self._think is not None:
            kwargs["think"] = self._think
        response = await self._inner.chat(**kwargs)
        if self._raw_log is not None:
            self._raw_log.append({
                "options_sent": kwargs.get("options"),
                "think_sent": kwargs.get("think"),
                "prompt_eval_count": response.prompt_eval_count,
                "eval_count": response.eval_count,
                "total_duration_ms": _ns_to_ms(response.total_duration),
                "load_duration_ms": _ns_to_ms(response.load_duration),
                "eval_duration_ms": _ns_to_ms(response.eval_duration),
                "thinking_chars": len(response.message.thinking or ""),
            })
        return response


@dataclass(slots=True)
class SpikeOllamaConfig(OllamaConfig):
    think: bool | None = None
    num_ctx: int | None = None
    raw_log: list[dict[str, Any]] | None = None

    def create(self):  # type: ignore[override]
        client = OllamaConfig.create(self)
        # 依賴 ag2 1.1.0 OllamaClient 的私有屬性 _create_options／_client。
        if self.num_ctx is not None:
            client._create_options["num_ctx"] = self.num_ctx
        client._client = _ChatProxy(client._client, self.think, self.raw_log)
        return client


def spike_config(think: bool | None, raw_log: list[dict[str, Any]] | None = None) -> SpikeOllamaConfig:
    return SpikeOllamaConfig(
        model=SMALL_MODEL, host=ollama_host(), temperature=TEMPERATURE, seed=SEED,
        think=think, num_ctx=NUM_CTX, raw_log=raw_log,
    )


# ---------------------------------------------------------------------------
# Trace middleware 與假工具
# ---------------------------------------------------------------------------


class Trace:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.events: list[dict[str, Any]] = []
        self.search_attempts = 0
        self.search_executions = 0
        self.final_answer: str | None = None
        self.hop_limit_hit = False
        self.reasoning_chars = 0
        self.raw_ollama: list[dict[str, Any]] = []

    def add(self, kind: str, **data: Any) -> None:
        self.events.append({"seq": len(self.events) + 1, "t_ms": self.elapsed_ms(), "kind": kind, **data})

    def elapsed_ms(self) -> float:
        return round((time.perf_counter() - self.t0) * 1000, 1)


class TraceMiddleware(BaseMiddleware):
    def __init__(self, event: Any, context: Any, trace: Trace, hop_limit: int) -> None:
        super().__init__(event, context)
        self.trace = trace
        self.hop_limit = hop_limit
        self.step = 0

    async def on_turn(self, call_next, event, context):
        self.trace.add("turn_start")
        try:
            response = await call_next(event, context)
        except BaseException as exc:  # 含 wait_for 造成的 CancelledError
            self.trace.add("turn_interrupted", error=type(exc).__name__)
            raise
        self.trace.add("turn_end", content=response.content)
        return response

    async def on_llm_call(self, call_next, events, context):
        self.step += 1
        started = time.perf_counter()
        response = await call_next(events, context)
        self.trace.add(
            "llm_call",
            step=self.step,
            input_events=len(events),
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            content=response.content,
            tool_calls=[{"name": c.name, "arguments": c.serialized_arguments} for c in response.tool_calls.calls],
            prompt_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            finish_reason=response.finish_reason,
        )
        return response

    async def on_tool_execution(self, call_next, event, context):
        arguments = event.serialized_arguments
        if event.name == "search":
            self.trace.search_attempts += 1
            if self.trace.search_attempts > self.hop_limit:
                self.trace.hop_limit_hit = True
                self.trace.add("hop_limit_abort", tool=event.name, arguments=arguments,
                               attempt=self.trace.search_attempts, hop_limit=self.hop_limit)
                # final=True 讓 AG2 直接結束本回合，不再呼叫 LLM。
                return ToolResultEvent.from_call(event, ToolResult("ABORTED: hop limit reached.", final=True))
        started = time.perf_counter()
        result = await call_next(event, context)
        self.trace.add(
            "tool_call",
            tool=event.name,
            arguments=arguments,
            result="".join(getattr(p, "content", str(p)) for p in result.result.parts)[:500],
            is_error=type(result).__name__ != "ToolResultEvent",
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return result


def make_tools(question: dict[str, Any], trace: Trace, search_delay_s: float) -> list[Any]:
    @tool
    async def search(query: str) -> str:
        """Search the knowledge base for one passage relevant to the query."""
        if search_delay_s:
            await asyncio.sleep(search_delay_s)
        index = trace.search_executions
        trace.search_executions += 1
        passages = question["passages"]
        return passages[index] if index < len(passages) else "No further results."

    @tool
    def final_answer(answer: str) -> ToolResult:
        """Submit the final short answer to the question."""
        trace.final_answer = answer
        return ToolResult(answer, final=True)

    return [search, final_answer]


async def run_question(
    question: dict[str, Any], *, config: OllamaConfig, config_label: str, hop_limit: int,
    timeout_s: float, search_delay_s: float = 0.0, cancel_grace_s: float = 0.0,
) -> dict[str, Any]:
    trace = Trace()
    if isinstance(config, SpikeOllamaConfig):
        config = config.copy()
        config.raw_log = trace.raw_ollama

    async def on_reasoning(event: ModelReasoning) -> None:
        trace.reasoning_chars += len(event.content)

    agent = Agent(
        "SpikeAgent", prompt=SYSTEM_PROMPT, config=config,
        tools=make_tools(question, trace, search_delay_s),
        middleware=[Middleware(TraceMiddleware, trace=trace, hop_limit=hop_limit)],
        observers=[observer(ModelReasoning, on_reasoning)],
    )
    status, error, reply_text = "done", None, None
    try:
        reply = await asyncio.wait_for(agent.ask(question["question"]), timeout_s)
        reply_text = await reply.content()
        if trace.hop_limit_hit:
            status = "hop_limit"
        elif trace.final_answer is None:
            status = "no_final_answer"
    except TimeoutError:
        status = "timeout"
        trace.add("timeout_abort", timeout_s=timeout_s)
    except Exception as exc:  # noqa: BLE001 - spike 需記錄所有失敗
        status, error = "error", f"{type(exc).__name__}: {exc}"
        trace.add("error", error=error)
    latency_ms = trace.elapsed_ms()

    events_after_abort = None
    if status == "timeout" and cancel_grace_s:
        count_at_abort = len(trace.events)
        await asyncio.sleep(cancel_grace_s)
        events_after_abort = len(trace.events) - count_at_abort

    llm_calls = [e for e in trace.events if e["kind"] == "llm_call"]
    answer = trace.final_answer if trace.final_answer is not None else reply_text
    return {
        "question_id": question["id"],
        "question": question["question"],
        "gold_answer": question["answer"],
        "config": config_label,
        "hop_limit": hop_limit,
        "timeout_s": timeout_s,
        "search_delay_s": search_delay_s,
        "status": status,
        "error": error,
        "final_answer": answer,
        "correct": bool(answer) and question["answer"].lower() in str(answer).lower(),
        "latency_ms": latency_ms,
        "search_attempts": trace.search_attempts,
        "search_executions": trace.search_executions,
        "llm_calls": len(llm_calls),
        "prompt_tokens_total": sum(e["prompt_tokens"] or 0 for e in llm_calls),
        "output_tokens_total": sum(e["output_tokens"] or 0 for e in llm_calls),
        "reasoning_chars": trace.reasoning_chars,
        "events_after_abort": events_after_abort,
        "events": trace.events,
        "raw_ollama": trace.raw_ollama,
    }


# ---------------------------------------------------------------------------
# 五項證據
# ---------------------------------------------------------------------------


def evidence(passed: bool, summary: str, **details: Any) -> dict[str, Any]:
    return {"status": "pass" if passed else "fail", "summary": summary, **details}


async def run_agent_stage(out_dir: Path) -> dict[str, Any]:
    await warm_up(SMALL_MODEL)

    main = [await run_question(q, config=spike_config(False), config_label="spike_config_think_false",
                               hop_limit=MAIN_HOP_LIMIT, timeout_s=MAIN_TIMEOUT_S) for q in QUESTIONS]
    write_jsonl(out_dir / "trace_main.jsonl", main)

    hop = [await run_question(q, config=spike_config(False), config_label="spike_config_think_false",
                              hop_limit=TEST_HOP_LIMIT, timeout_s=MAIN_TIMEOUT_S) for q in QUESTIONS]
    write_jsonl(out_dir / "trace_hop_limit.jsonl", hop)

    three_hop = [q for q in QUESTIONS if len(q["passages"]) >= 3]
    timeout = [await run_question(q, config=spike_config(False), config_label="spike_config_think_false",
                                  hop_limit=MAIN_HOP_LIMIT, timeout_s=TEST_TIMEOUT_S,
                                  search_delay_s=TIMEOUT_SEARCH_DELAY_S, cancel_grace_s=5.0)
               for q in three_hop]
    write_jsonl(out_dir / "trace_timeout.jsonl", timeout)

    thinking_rows, thinking = await run_thinking(QUESTIONS[0])
    write_jsonl(out_dir / "trace_thinking.jsonl", thinking_rows)

    return {
        "evidence": {
            "step_trace": judge_step_trace(main),
            "token_counts": judge_token_counts(main),
            "hop_limit_abort": judge_hop_limit(hop),
            "timeout_abort": judge_timeout(timeout),
            "thinking_off": thinking,
        },
        "agent_runs": {
            "main": summarize_rows(main),
            "hop_limit": summarize_rows(hop),
            "timeout": summarize_rows(timeout),
        },
    }


def judge_step_trace(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def complete(row: dict[str, Any]) -> bool:
        kinds = [e["kind"] for e in row["events"]]
        tools_ok = all("arguments" in e and "result" in e for e in row["events"] if e["kind"] == "tool_call")
        return "llm_call" in kinds and "tool_call" in kinds and tools_ok and kinds[0] == "turn_start"

    ok = [r["question_id"] for r in rows if complete(r)]
    sequences = {r["question_id"]: " > ".join(
        e["kind"] if e["kind"] != "tool_call" else f"tool:{e['tool']}" for e in r["events"]) for r in rows}
    return evidence(
        len(ok) == len(rows),
        f"{len(ok)}/{len(rows)} 題有依序的 llm_call（含 tool_calls 參數）與 tool_call（含參數與結果）事件；"
        f"trace 由 AG2 middleware on_turn／on_llm_call／on_tool_execution 產生",
        trace_file="trace_main.jsonl",
        sequences=sequences,
    )


def judge_token_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    calls, mismatches = [], []
    for row in rows:
        llm = [e for e in row["events"] if e["kind"] == "llm_call"]
        for event, raw in zip(llm, row["raw_ollama"], strict=False):
            calls.append((event["prompt_tokens"], event["output_tokens"]))
            if (event["prompt_tokens"], event["output_tokens"]) != (raw["prompt_eval_count"], raw["eval_count"]):
                mismatches.append({"question_id": row["question_id"], "step": event["step"]})
        if len(llm) != len(row["raw_ollama"]):
            mismatches.append({"question_id": row["question_id"], "reason": "llm_call 與原始回應數量不同"})
    positive = all((p or 0) > 0 and (o or 0) > 0 for p, o in calls)
    return evidence(
        bool(calls) and positive and not mismatches,
        f"{len(calls)} 次 LLM 呼叫皆有 prompt／output token；來源為 AG2 ModelResponse.usage"
        f"（ag2 OllamaClient 由 Ollama prompt_eval_count／eval_count 轉入），與原始回應比對不一致 {len(mismatches)} 次",
        source="ag2 ModelResponse.usage <- ollama prompt_eval_count/eval_count",
        llm_calls=len(calls),
        prompt_tokens_total=sum(p or 0 for p, _ in calls),
        output_tokens_total=sum(o or 0 for _, o in calls),
        mismatches=mismatches,
        note="on_llm_call 只拿得到 usage；eval_duration 等原始耗時需包住 ollama client 才取得（見 raw_ollama）",
    )


def judge_hop_limit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hit = [r for r in rows if r["status"] == "hop_limit"]
    clean = [r for r in hit if r["search_executions"] == TEST_HOP_LIMIT
             and not any(e["kind"] == "llm_call" and e["seq"] > next(
                 x["seq"] for x in r["events"] if x["kind"] == "hop_limit_abort") for e in r["events"])]
    return evidence(
        bool(hit) and len(clean) == len(hit),
        f"上限 {TEST_HOP_LIMIT}：{len(hit)}/{len(rows)} 題在第 {TEST_HOP_LIMIT + 1} 次 search 被 middleware 中止"
        f"（記錄 hop_limit_abort，search 實際只執行 {TEST_HOP_LIMIT} 次，中止後無 LLM 呼叫）；"
        f"其餘題目未發出第 3 次 search",
        trace_file="trace_hop_limit.jsonl",
        aborted_questions=[r["question_id"] for r in hit],
        clean_aborts=[r["question_id"] for r in clean],
    )


def judge_timeout(rows: list[dict[str, Any]]) -> dict[str, Any]:
    timed_out = [r for r in rows if r["status"] == "timeout"]
    clean = [r for r in timed_out if r["latency_ms"] < (TEST_TIMEOUT_S + 1) * 1000 and r["events_after_abort"] == 0
             and any(e["kind"] == "turn_interrupted" for e in r["events"])]
    return evidence(
        bool(rows) and len(clean) == len(rows),
        f"逾時 {TEST_TIMEOUT_S:g} 秒（search 模擬 {TIMEOUT_SEARCH_DELAY_S:g} 秒延遲）：{len(clean)}/{len(rows)} 題由 "
        f"asyncio.wait_for 中止，記錄 timeout_abort 與 turn_interrupted，中止後 5 秒內無新事件",
        trace_file="trace_timeout.jsonl",
        latencies_ms={r["question_id"]: r["latency_ms"] for r in rows},
    )


async def run_thinking(question: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # 1) 直接用 AG2 OllamaConfig 傳 think
    try:
        OllamaConfig(model=SMALL_MODEL, think=False)  # type: ignore[call-arg]
        stock_accepts_think, stock_error = True, None
    except TypeError as exc:
        stock_accepts_think, stock_error = False, str(exc)

    variants = {
        "stock_ollama_config": OllamaConfig(model=SMALL_MODEL, host=ollama_host(), temperature=TEMPERATURE, seed=SEED),
        "spike_config_think_true": spike_config(True),
        "spike_config_think_false": spike_config(False),
    }
    rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    for label, config in variants.items():
        runs = []
        for repeat in range(THINKING_REPEATS + 1):  # 第 0 次為暖機（含 num_ctx 改變時的重新載入）
            row = await run_question(question, config=config, config_label=label,
                                     hop_limit=MAIN_HOP_LIMIT, timeout_s=MAIN_TIMEOUT_S)
            row["repeat"] = repeat
            row["warm_up"] = repeat == 0
            rows.append(row)
            if repeat:
                runs.append(row)
        stats[label] = {
            "median_latency_ms": statistics.median(r["latency_ms"] for r in runs),
            "reasoning_chars": [r["reasoning_chars"] for r in runs],
            "output_tokens": [r["output_tokens_total"] for r in runs],
            "statuses": [r["status"] for r in runs],
        }
    fallback = await run_fallback_reference(question)

    off, on = stats["spike_config_think_false"], stats["spike_config_think_true"]
    workaround_ok = all(c == 0 for c in off["reasoning_chars"]) and all(c > 0 for c in on["reasoning_chars"]) \
        and off["median_latency_ms"] < 0.7 * on["median_latency_ms"]
    return rows, evidence(
        stock_accepts_think and workaround_ok,
        ("AG2 OllamaConfig " + ("可" if stock_accepts_think else "不可") + " 傳 think"
         + (f"（{stock_error}）" if stock_error else "")
         + f"；未傳 think 時回應含 thinking（reasoning 字元 {stats['stock_ollama_config']['reasoning_chars']}）"
         + f"。子類別 workaround：think=False 中位延遲 {off['median_latency_ms']} ms、thinking 字元 {off['reasoning_chars']}；"
         + f"think=True 中位延遲 {on['median_latency_ms']} ms、thinking 字元 {on['reasoning_chars']}"),
        trace_file="trace_thinking.jsonl",
        stock_ollama_config_accepts_think=stock_accepts_think,
        stock_ollama_config_error=stock_error,
        stock_ollama_config_accepts_num_ctx="num_ctx" in OllamaConfig.__dataclass_fields__,
        workaround_status="pass" if workaround_ok else "fail",
        workaround_note="SpikeOllamaConfig 繼承 OllamaConfig，改寫 create() 並依賴私有屬性 _create_options／_client",
        per_config=stats,
        ollama_client_reference=fallback,
    )


async def run_fallback_reference(question: dict[str, Any]) -> dict[str, Any]:
    """回退路徑參考：直接用 ollama AsyncClient 發第一步請求，量 think True／False。"""
    client = ollama.AsyncClient(host=ollama_host())
    tools = [{"type": "function", "function": {
        "name": "search", "description": "Search the knowledge base for one passage relevant to the query.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question["question"]}]
    result: dict[str, Any] = {}
    for think in (True, False):
        samples = []
        for repeat in range(THINKING_REPEATS + 1):
            started = time.perf_counter()
            response = await client.chat(model=SMALL_MODEL, messages=messages, tools=tools, think=think,
                                         options=call_options())
            if repeat:
                samples.append({
                    "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                    "thinking_chars": len(response.message.thinking or ""),
                    "eval_count": response.eval_count,
                    "tool_calls": [{"name": c.function.name, "arguments": c.function.arguments}
                                   for c in response.message.tool_calls or []],
                })
        result[f"think_{str(think).lower()}"] = {
            "median_latency_ms": statistics.median(s["latency_ms"] for s in samples),
            "samples": samples,
        }
    return result


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("question_id", "status", "final_answer", "correct", "latency_ms", "search_attempts",
            "search_executions", "llm_calls", "prompt_tokens_total", "output_tokens_total", "reasoning_chars")
    return [{k: r[k] for k in keys} for r in rows]


# ---------------------------------------------------------------------------
# VRAM：檢索堆疊先上 GPU（與正式 run 相同順序），再載入 27B
# ---------------------------------------------------------------------------


def run_vram_stage() -> dict[str, Any]:
    # torch 2.14 的 torch._native 會把 ModernBERT RoPE 的 bmm 送到 Triton kernel，需 C 編譯器；
    # WSL 無 gcc 且不可 sudo，關閉後走原生 CUDA kernel。
    os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
    import laya
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    result: dict[str, Any] = {
        "model": LARGE_MODEL, "num_ctx": NUM_CTX,
        "load_order": ["BAAI/bge-m3", "BAAI/bge-reranker-v2-m3", "convaiinnovations/laya-multilingual", LARGE_MODEL],
        "nvidia_smi_before": nvidia_smi_used(),
    }
    embedder = SentenceTransformer("BAAI/bge-m3", device="cuda")
    embedder.encode(["warm up passage"])
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda")
    reranker.predict([("warm up query", "warm up passage")])
    decision = laya.load("convaiinnovations/laya-multilingual", device="cuda")
    decision.predict(state="Question: warm up", questions={"sufficient": {
        "type": "noul", "instructions": "Is the evidence sufficient to answer the question?",
        "criteria": {"false": "not sufficient", "true": "sufficient"}}})
    torch.cuda.synchronize()
    result["torch_memory_allocated_mib"] = round(torch.cuda.memory_allocated() / 2**20)
    result["torch_memory_reserved_mib"] = round(torch.cuda.memory_reserved() / 2**20)
    result["nvidia_smi_after_retrieval_stack"] = nvidia_smi_used()

    client = ollama.Client(host=ollama_host())
    started = time.perf_counter()
    response = client.chat(model=LARGE_MODEL, messages=[{"role": "user", "content": "Reply with OK."}],
                           think=False, options=call_options(), keep_alive="10m")
    result["first_request_s"] = round(time.perf_counter() - started, 1)
    result["first_request_reply"] = response.message.content
    result["ollama_ps_raw"] = run_cmd(["ollama", "ps"])
    result["nvidia_smi_memory_used_raw"] = run_cmd(["nvidia-smi", "--query-gpu=memory.used", "--format=csv"])
    result["nvidia_smi_27b_loaded"] = nvidia_smi_used()
    ps = [m for m in client.ps().models if m.model == LARGE_MODEL or m.name == LARGE_MODEL]
    size = ps[0].size if ps else None
    size_vram = ps[0].size_vram if ps else None
    result["api_ps"] = {"size": size, "size_vram": size_vram,
                        "context_length": getattr(ps[0], "context_length", None) if ps else None}
    result["processor"] = parse_processor(result["ollama_ps_raw"], LARGE_MODEL)
    offload = None if not size else round(1 - (size_vram or 0) / size, 4)
    result["cpu_offload_ratio"] = offload
    result["verdict"] = "100% GPU" if result["processor"] == "100% GPU" else f"offload: {result['processor']}"

    # 27B 常駐時跑一次接近正式迴圈的負載：dense 編碼、top-20 rerank、Laya 推論、再問 27B。
    workload: dict[str, Any] = {}
    try:
        torch.cuda.reset_peak_memory_stats()
        passage = " ".join(["The committee reviewed the historical records of the river town in detail."] * 30)
        embedder.encode(["Which river flows through the painter's hometown?"] + [passage] * 20, batch_size=32)
        reranker.predict([("Which river flows through the painter's hometown?", passage)] * 20, batch_size=20)
        decision.predict(state="Question: warm up\nEvidence: " + passage[:1500], questions={"sufficient": {
            "type": "noul", "instructions": "Is the evidence sufficient to answer the question?",
            "criteria": {"false": "not sufficient", "true": "sufficient"}}})
        torch.cuda.synchronize()
        workload["torch_max_memory_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
        workload["nvidia_smi_after_workload"] = nvidia_smi_used()
        started = time.perf_counter()
        client.chat(model=LARGE_MODEL, messages=[{"role": "user", "content": "Reply with OK."}],
                    think=False, options=call_options(), keep_alive="10m")
        workload["second_request_s"] = round(time.perf_counter() - started, 1)
        workload["ollama_ps_raw"] = run_cmd(["ollama", "ps"])
        workload["processor"] = parse_processor(workload["ollama_ps_raw"], LARGE_MODEL)
        workload["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - OOM 也是要記錄的結果
        workload["status"] = f"error: {type(exc).__name__}: {str(exc)[:300]}"
    result["workload_check"] = workload

    client.chat(model=LARGE_MODEL, messages=[], keep_alive=0)  # 卸載 27B
    del embedder, reranker, decision
    torch.cuda.empty_cache()
    time.sleep(3)
    result["ollama_ps_after_unload"] = run_cmd(["ollama", "ps"])
    return result


def parse_processor(ps_raw: str, model: str) -> str | None:
    for line in ps_raw.splitlines():
        if line.startswith(model + " "):
            match = re.search(r"(\d+%/\d+% CPU/GPU|\d+% GPU|\d+% CPU)", line)
            return match.group(1) if match else None
    return None


# ---------------------------------------------------------------------------
# 共用
# ---------------------------------------------------------------------------


def conclusion(results: dict[str, Any]) -> dict[str, Any]:
    ev = results.get("evidence") or {}
    failed = [k for k in EVIDENCE_KEYS if (ev.get(k) or {}).get("status") != "pass"]
    vram = results.get("vram") or {}
    out: dict[str, Any] = {"failed_evidence": failed if ev else None}
    if ev:
        out["agent_loop"] = "ag2" if not failed else "fallback_ollama_client"
        out["agent_loop_text"] = "沿用 AG2 1.x" if not failed else "回退：直接用 ollama Python client"
    else:
        out["agent_loop"] = out["agent_loop_text"] = None
    if vram:
        on_gpu = vram.get("verdict") == "100% GPU"
        out["retrieval_stack_device"] = "cuda" if on_gpu else "cpu"
        out["retrieval_stack_text"] = ("BGE-M3 與 reranker 放 GPU" if on_gpu
                                       else "五模型一律 BGE-M3 與 reranker 放 CPU（Laya 保留 GPU）")
    else:
        out["retrieval_stack_device"] = out["retrieval_stack_text"] = None
    return out


def empty_results() -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "generated_at": None, "environment": None, "call_options": None,
            "evidence": {k: None for k in EVIDENCE_KEYS}, "agent_runs": None, "vram": None, "conclusion": None}


def environment() -> dict[str, Any]:
    client = ollama.Client(host=ollama_host())
    models = {m.model: {"digest": m.digest, "size": m.size} for m in client.list().models}
    try:
        server_version = client._request_raw("GET", "/api/version").json().get("version")
    except Exception:  # noqa: BLE001
        server_version = None
    return {
        "ag2_version": AG2_VERSION,
        "ollama_python_version": _pkg_version("ollama"),
        "ollama_server_version": server_version,
        "ollama_host": ollama_host(),
        "python": sys.version.split()[0],
        "models": {k: v for k, v in models.items() if k in (SMALL_MODEL, LARGE_MODEL)},
    }


def call_options() -> dict[str, Any]:
    return {"num_ctx": NUM_CTX, "temperature": TEMPERATURE, "seed": SEED}


async def warm_up(model: str) -> None:
    await ollama.AsyncClient(host=ollama_host()).chat(
        model=model, messages=[{"role": "user", "content": "hi"}], think=False, options=call_options())


def nvidia_smi_used() -> str:
    return run_cmd(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"]).strip()


def run_cmd(cmd: list[str]) -> str:
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return completed.stdout + completed.stderr


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def _ns_to_ms(value: int | None) -> float | None:
    return None if value is None else round(value / 1e6, 1)


def _pkg_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("agent", "vram", "all"))
    parser.add_argument("--out-dir", type=Path, default=None, help="預設 $PROJECT_DATA_ROOT/runs/spike")
    args = parser.parse_args()

    out_dir = args.out_dir
    if out_dir is None:
        data_root = os.environ.get("PROJECT_DATA_ROOT")
        if not data_root:
            print("缺少 PROJECT_DATA_ROOT，請先 source multihop_benchmark/scripts/env.sh", file=sys.stderr)
            return 2
        out_dir = Path(data_root) / "runs" / "spike"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "spike_results.json"
    results = empty_results()
    if results_path.exists():
        previous = json.loads(results_path.read_text(encoding="utf-8"))
        if previous.get("schema_version") == SCHEMA_VERSION:
            results.update({k: previous[k] for k in results if k in previous})

    if args.stage in ("agent", "all"):
        results.update(asyncio.run(run_agent_stage(out_dir)))
    if args.stage in ("vram", "all"):
        results["vram"] = run_vram_stage()
    results["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    results["environment"] = environment()
    results["call_options"] = call_options()
    results["conclusion"] = conclusion(results)
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"results": str(results_path), "conclusion": results["conclusion"],
                      "evidence": {k: (v or {}).get("status") for k, v in results["evidence"].items()},
                      "vram_verdict": (results["vram"] or {}).get("verdict")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
