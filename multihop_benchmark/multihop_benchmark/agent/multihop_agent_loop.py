"""四條件共用的多跳拆解迴圈（ADR 0001；Spike 回退後以自寫迴圈取代 AG2，ADR 0002／Spec 補充）。

兩個開關：`evidence_source ∈ {recall, rag}`、`controller ∈ {llm, laya}`，condition = `<evidence_source>_<controller>`。

每一跳先由控制者決定「停止作答或再問一個子問題」，兩種控制者的決策點相同，跳數可直接比較：

- `controller=laya`：迴圈強制呼叫 `decision.decide(state)`；`sufficient_p ≥ 0.5` 立即停止。否則請 LLM 以
  `ask_sub_question` 工具提出子問題，prompt 附上 Laya 的 `next_action`／`remaining` 提示。
- `controller=llm`：請 LLM 在 `ask_sub_question` 與 `final_answer` 兩個工具中擇一；呼叫 `final_answer` 即停止
  （它的參數不作為答案，作答一律由下方最終作答呼叫產生，四條件的作答方式相同）。
- 停止的那一跳仍記為一跳（`sub_question=""`、`evidence_added=[]`）；`hops = len(steps)`。
- 取得子問題後：`rag` 取 `retriever.retrieve(sub_q)` 前 3 段；`recall` 再呼叫 LLM 閉卷回憶一次，回憶文字作為一段
  evidence（`pid = recall-<hop>`、`title = 子問題`）。evidence 依 `pid` 去重、總數上限 8，超過的新段落捨棄。
- 迴圈結束（停止或到 `max_hops`）後，LLM 依 question＋evidence 作答一次（純文字，取第一個非空行）。

LLM 呼叫順序（假 LLM 依此腳本化）：每跳「子問題呼叫 →（recall 時）回憶呼叫」，停止的那跳只有 controller=llm 的
子問題呼叫（回傳 final_answer），最後一次作答呼叫。

token 與延遲：`steps[i].llm` 為第 i 跳全部 LLM 呼叫（子問題、回憶）的 prompt／output token 與延遲加總；最終作答
呼叫併入最後一跳。因此 `sum(steps[].llm.*)` 即該題全部 LLM 用量。

狀態：
- `done`：正常作答。
- `invalid_tool`：未知工具名、參數不合法、參數非 JSON、需要工具時沒有工具呼叫，或伺服器回報無法解析工具呼叫；
  prediction 為空字串、視為答錯。
- `timeout`：整題超過 `timeout_s`；保留已完成的 steps，`error` 註明逾時發生點（例 `decision.decide @ hop 2`）。
- `oom`：LLM 拋 `LLMOutOfMemoryError`，或 Laya／檢索拋 torch `OutOfMemoryError`。本模組不重試；benchmark_runner
  看到 `status == "oom"` 時以 `num_ctx=4096` 的新 client 重跑一次，仍為 oom 才寫入。
- `error`：其他例外，訊息寫入 `error`。
- `LLMConnectionError`（Ollama 連線重試用盡）不轉成 record，直接往外拋，由 benchmark_runner 中止整個 run。

逾時策略：async 的 LLM 呼叫在 deadline 直接取消（Ollama 請求隨連線關閉中止）。同步的 `decision.decide`／
`retriever.retrieve` 是 GPU 運算，無法中途取消，因此在每題專用的單執行緒 executor 中執行；逾時若發生在它們
執行中，回傳前會等它自然結束並關閉 executor，確保不殘留 worker、不與下一題的 GPU 工作重疊。此時
`latency_ms` 是含等待的實際 wall time（大於 `timeout_s`），`run_question` 最晚在 deadline 加上該次同步呼叫
的剩餘時間後返回。

`vram_peak_mb` 固定為 None、不設 `finished_at`，兩者由 benchmark_runner 填入。
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from multihop_benchmark.agent.protocols import (
    DecisionModel,
    InvalidToolCallError,
    LLMClient,
    LLMConnectionError,
    LLMOutOfMemoryError,
    LLMResponse,
    Retriever,
)
from multihop_benchmark.evaluation import answer_scorer
from multihop_benchmark.runs.record_store import LAYA_NEXT_ACTIONS, LAYA_REMAINING_HOPS, validate_record

EVIDENCE_SOURCES = ("recall", "rag")
CONTROLLERS = ("llm", "laya")
SUFFICIENT_THRESHOLD = 0.5
RAG_TOP_N = 3
MAX_EVIDENCE = 8

ASK_TOOL = "ask_sub_question"
FINAL_TOOL = "final_answer"


def _tool(name: str, description: str, arg: str, arg_description: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {arg: {"type": "string", "description": arg_description}},
                "required": [arg],
            },
        },
    }


TOOLS = {
    ASK_TOOL: _tool(ASK_TOOL, "Ask the next single-hop sub-question needed to answer the original question.",
                    "sub_question", "One short, self-contained factual question."),
    FINAL_TOOL: _tool(FINAL_TOOL, "Stop gathering evidence because it is enough to answer the original question.",
                      "answer", "A short answer: an entity, date, number, or yes/no."),
}
TOOL_ARG = {ASK_TOOL: "sub_question", FINAL_TOOL: "answer"}

SYSTEM_PROMPT = (
    "You answer multi-hop questions by breaking them into single-hop sub-questions. "
    "Each sub-question gathers one fact. Always respond by calling one of the provided tools."
)
LAYA_HINTS = {
    "A": "the evidence may already be close to enough; ask for the single most important missing fact",
    "B": "ask a new sub-question for the next missing fact",
    "C": "the last query did not find useful evidence; rephrase it or ask about the same fact differently",
}


class _Trace:
    """迴圈進行中的狀態；逾時或例外時仍保留已記下的 steps。"""

    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []
        self.sub_questions: list[str] = []
        self.evidence: list[dict[str, str]] = []
        self.prediction = ""
        self.phase = "start"  # 目前進行中的呼叫，逾時時寫入 error
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="multihop-sync")
        self._pending: Future | None = None

    async def call_sync(self, phase: str, fn: Callable[[Any], Any], arg: Any) -> Any:
        """在本題專用執行緒跑同步的 Laya／檢索，並記下 future 供逾時後等待。"""
        self.phase = phase
        self._pending = self._executor.submit(fn, arg)
        return await asyncio.wrap_future(self._pending)

    async def drain(self) -> float:
        """等待被逾時打斷、仍在執行的同步呼叫結束；回傳等待毫秒數。其結果或例外一律捨棄。"""
        if self._pending is None or self._pending.done():
            return 0.0
        started = time.perf_counter()
        try:
            await asyncio.wrap_future(self._pending)
        except Exception:  # noqa: BLE001 - 題目已判定逾時，worker 的結果不再使用
            pass
        return (time.perf_counter() - started) * 1000

    def close(self) -> None:
        # 正常路徑 worker 已閒置，join 立即完成；外部取消而未 drain 時不阻塞事件迴圈。
        self._executor.shutdown(wait=self._pending is None or self._pending.done())

    def new_step(self, hop: int) -> dict[str, Any]:
        step = {"hop": hop, "sub_question": "", "evidence_added": [],
                "llm": {"prompt_tokens": 0, "output_tokens": 0, "latency_ms": 0.0}}
        self.steps.append(step)
        return step


def _add_usage(step: dict[str, Any], response: LLMResponse) -> None:
    usage = step["llm"]
    usage["prompt_tokens"] += int(response.prompt_tokens or 0)
    usage["output_tokens"] += int(response.output_tokens or 0)
    usage["latency_ms"] = round(usage["latency_ms"] + float(response.latency_ms or 0.0), 3)


def _format_context(question: str, sub_questions: list[str], evidence: list[dict[str, str]]) -> str:
    lines = [f"Question: {question}"]
    if sub_questions:
        lines.append("Sub-questions asked so far:")
        lines += [f"{i}. {q}" for i, q in enumerate(sub_questions, start=1)]
    lines.append("Evidence:" if evidence else "Evidence: (none yet)")
    lines += [f"[{i}] {p['title']}: {p['text']}" for i, p in enumerate(evidence, start=1)]
    return "\n".join(lines)


def _parse_tool_call(response: LLMResponse, allowed: tuple[str, ...]) -> tuple[str, str]:
    """取第一個工具呼叫，驗證工具名與唯一必要字串參數；不合法拋 InvalidToolCallError。"""
    if not response.tool_calls:
        raise InvalidToolCallError(f"需要工具呼叫（{'/'.join(allowed)}），模型只回傳文字：{response.content[:200]!r}")
    call = response.tool_calls[0]
    if call.name not in allowed:
        raise InvalidToolCallError(f"未知工具 {call.name!r}（允許 {'/'.join(allowed)}）")
    arguments: Any = call.arguments
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise InvalidToolCallError(f"{call.name} 參數不是 JSON：{call.arguments[:200]!r}") from exc
    arg = TOOL_ARG[call.name]
    value = arguments.get(arg) if isinstance(arguments, Mapping) else None
    if not isinstance(value, str) or not value.strip():
        raise InvalidToolCallError(f"{call.name} 參數不合法，需要非空字串 {arg}：{call.arguments!r}")
    return call.name, value.strip()


def _normalize_decision(raw: Mapping[str, Any], measured_ms: float) -> dict[str, Any]:
    """把 DecisionModel 回傳值正規化為 record 的 laya 欄位；無法辨識時拋 ValueError。"""
    p = float(raw["sufficient_p"])
    if p != p:  # NaN
        raise ValueError("Laya sufficient_p 為 NaN")
    action = raw["next_action"]
    if isinstance(action, int) and not isinstance(action, bool) and 0 <= action < len(LAYA_NEXT_ACTIONS):
        action = LAYA_NEXT_ACTIONS[action]
    action = str(action).strip().upper()
    if action not in LAYA_NEXT_ACTIONS:
        raise ValueError(f"Laya next_action 無法辨識：{raw['next_action']!r}")
    remaining = int(round(float(raw["remaining"])))
    latency = raw.get("latency_ms")
    latency = float(latency) if isinstance(latency, (int, float)) and not isinstance(latency, bool) else measured_ms
    return {
        "sufficient_p": min(max(p, 0.0), 1.0),
        "next_action": action,
        "remaining": min(max(remaining, LAYA_REMAINING_HOPS[0]), LAYA_REMAINING_HOPS[-1]),
        "latency_ms": round(max(latency, 0.0), 3),
    }


def _merge_evidence(trace: _Trace, passages: list[Mapping[str, Any]]) -> list[dict[str, str]]:
    added: list[dict[str, str]] = []
    seen = {p["pid"] for p in trace.evidence}
    for raw in passages:
        if len(trace.evidence) >= MAX_EVIDENCE:
            break
        item = {"pid": str(raw["pid"]), "title": str(raw.get("title", "")), "text": str(raw.get("text", ""))}
        if item["pid"] in seen:
            continue
        seen.add(item["pid"])
        trace.evidence.append(item)
        added.append(item)
    return added


async def _loop(
    trace: _Trace, question: str, evidence_source: str, controller: str,
    llm: LLMClient, decision: DecisionModel | None, retriever: Retriever | None, max_hops: int,
) -> None:
    for hop in range(1, max_hops + 1):
        step = trace.new_step(hop)
        context = _format_context(question, trace.sub_questions, trace.evidence)

        if controller == "laya":
            state = {"question": question, "sub_questions": list(trace.sub_questions),
                     "evidence": [dict(p) for p in trace.evidence]}
            started = time.perf_counter()
            raw = await trace.call_sync(f"decision.decide @ hop {hop}", decision.decide, state)
            laya = _normalize_decision(raw, (time.perf_counter() - started) * 1000)
            step["laya"] = laya
            if laya["sufficient_p"] >= SUFFICIENT_THRESHOLD:
                return
            allowed: tuple[str, ...] = (ASK_TOOL,)
            instruction = (f"Decision layer hint: next action {laya['next_action']} "
                           f"({LAYA_HINTS[laya['next_action']]}); about {laya['remaining']} key piece(s) "
                           f"of evidence still missing.\nCall {ASK_TOOL} with the next sub-question.")
        else:
            allowed = (ASK_TOOL, FINAL_TOOL)
            instruction = (f"If the evidence is enough to answer the question, call {FINAL_TOOL}. "
                           f"Otherwise call {ASK_TOOL} with the next sub-question.")

        trace.phase = f"llm.chat（子問題）@ hop {hop}"
        response = await llm.chat(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"{context}\n\n{instruction}"}],
            tools=[TOOLS[name] for name in allowed],
        )
        _add_usage(step, response)
        tool_name, value = _parse_tool_call(response, allowed)
        if tool_name == FINAL_TOOL:
            return

        step["sub_question"] = value
        trace.sub_questions.append(value)
        if evidence_source == "rag":
            passages = await trace.call_sync(f"retriever.retrieve @ hop {hop}", retriever.retrieve, value)
            step["evidence_added"] = _merge_evidence(trace, list(passages)[:RAG_TOP_N])
        else:
            trace.phase = f"llm.chat（回憶）@ hop {hop}"
            recall = await llm.chat([{"role": "user", "content": (
                f"Original question: {question}\nSub-question: {value}\n\n"
                "Using only your own knowledge, state the facts that answer the sub-question "
                "in one to three sentences.")}])
            _add_usage(step, recall)
            step["evidence_added"] = _merge_evidence(
                trace, [{"pid": f"recall-{hop}", "title": value, "text": recall.content.strip()}])

    # 到 max_hops 仍未停止：強制作答。


async def _answer(trace: _Trace, question: str, llm: LLMClient) -> None:
    context = _format_context(question, trace.sub_questions, trace.evidence)
    trace.phase = f"llm.chat（最終作答）@ hop {len(trace.steps)}"
    response = await llm.chat([{"role": "user", "content": (
        f"{context}\n\nAnswer the original question using the evidence: {question}\nReply with only the short answer "
        "(an entity, date, number, or yes/no), no explanation.")}])
    _add_usage(trace.steps[-1], response)
    lines = [line.strip() for line in response.content.strip().splitlines() if line.strip()]
    trace.prediction = lines[0] if lines else ""


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, LLMOutOfMemoryError) or type(exc).__name__ == "OutOfMemoryError"


async def arun_question(
    question: Mapping[str, Any],
    evidence_source: str,
    controller: str,
    llm: LLMClient,
    decision: DecisionModel | None = None,
    retriever: Retriever | None = None,
    max_hops: int = 4,
    timeout_s: float = 300,
    *,
    run_id: str,
    dataset: str,
    model: str,
    condition: str | None = None,
) -> dict[str, Any]:
    """async 版 `run_question`（已有事件迴圈時使用）。"""
    if evidence_source not in EVIDENCE_SOURCES:
        raise ValueError(f"evidence_source 必須是 {EVIDENCE_SOURCES} 之一：{evidence_source!r}")
    if controller not in CONTROLLERS:
        raise ValueError(f"controller 必須是 {CONTROLLERS} 之一：{controller!r}")
    expected = f"{evidence_source}_{controller}"
    if condition is not None and condition != expected:
        raise ValueError(f"condition {condition!r} 與 evidence_source/controller 不符（應為 {expected!r}）")
    if controller == "laya" and decision is None:
        raise ValueError("controller=laya 需要 decision")
    if evidence_source == "rag" and retriever is None:
        raise ValueError("evidence_source=rag 需要 retriever")
    if max_hops < 1:
        raise ValueError(f"max_hops 必須 ≥ 1：{max_hops}")

    trace = _Trace()
    status, error = "done", None
    started = time.perf_counter()

    async def work() -> None:
        await _loop(trace, question["question"], evidence_source, controller, llm, decision, retriever, max_hops)
        await _answer(trace, question["question"], llm)

    try:
        try:
            await asyncio.wait_for(work(), timeout_s)
        except TimeoutError:
            status, error = "timeout", f"超過 timeout_s={timeout_s}，逾時發生於 {trace.phase}"
            waited_ms = await trace.drain()
            if waited_ms:
                error += f"；已等待進行中的同步呼叫結束（額外 {waited_ms:.0f} ms）"
        except InvalidToolCallError as exc:
            status, error = "invalid_tool", str(exc)
        except LLMConnectionError:
            raise
        except Exception as exc:  # noqa: BLE001 - 單題失敗記入 record，不中止 run
            status = "oom" if _is_oom(exc) else "error"
            error = f"{type(exc).__name__}: {exc}"
    finally:
        trace.close()
    latency_ms = round((time.perf_counter() - started) * 1000, 3)

    prediction = trace.prediction if status == "done" else ""
    gold, aliases = question["answer"], list(question.get("answer_aliases") or [])
    record: dict[str, Any] = {
        "run_id": run_id,
        "dataset": dataset,
        "model": model,
        "condition": expected,
        "question_id": question["id"],
        "gold": gold,
        "gold_aliases": aliases,
        "prediction": prediction,
        "em": answer_scorer.exact_match(prediction, gold, aliases),
        "f1": answer_scorer.f1(prediction, gold, aliases),
        "status": status,
        "hops": len(trace.steps),
        "steps": trace.steps,
        "latency_ms": latency_ms,
        "vram_peak_mb": None,
    }
    if error is not None:
        record["error"] = error
    validate_record(record)
    return record


def run_question(
    question: Mapping[str, Any],
    evidence_source: str,
    controller: str,
    llm: LLMClient,
    decision: DecisionModel | None = None,
    retriever: Retriever | None = None,
    max_hops: int = 4,
    timeout_s: float = 300,
    *,
    run_id: str,
    dataset: str,
    model: str,
    condition: str | None = None,
) -> dict[str, Any]:
    """跑一題並回傳已通過 `record_store.validate_record` 的 record（行為見模組說明）。

    `question` 為統一 schema（`id`、`question`、`answer`、`answer_aliases`）。`model` 為真實 Ollama tag；
    `condition` 可省略（由兩個開關推得），給定時必須一致。以 `asyncio.run` 執行，已在事件迴圈內時改用
    `arun_question`。
    """
    return asyncio.run(arun_question(
        question, evidence_source, controller, llm, decision, retriever, max_hops, timeout_s,
        run_id=run_id, dataset=dataset, model=model, condition=condition,
    ))
