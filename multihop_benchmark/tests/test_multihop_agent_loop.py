"""multihop_agent_loop 以假 LLM／Laya／Retriever 驗證四條件的控制流與 record 內容。

假 LLM 依呼叫順序回應（見 `run_question` docstring 的「LLM 呼叫順序」）：
每跳一次子問題呼叫（controller=llm 時可改呼叫 final_answer）→ recall 時再一次回憶呼叫；迴圈結束後一次作答呼叫。
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest
from fakes import FakeDecision, FakeLLM, FakeRetriever, decision, final, passage, sub_q, text, tool_reply

from multihop_benchmark.agent.multihop_agent_loop import arun_question, run_question
from multihop_benchmark.agent.protocols import InvalidToolCallError, LLMConnectionError, LLMOutOfMemoryError
from multihop_benchmark.runs.record_store import validate_record

QUESTION = {
    "id": "q1",
    "question": "In which city was the founder of Zorvex Labs born?",
    "answer": "Tallinn",
    "answer_aliases": ["Tallinn, Estonia"],
    "paragraphs": [],
}


def run(evidence_source, controller, llm, decision_model=None, retriever=None, **kwargs):
    kwargs.setdefault("max_hops", 4)
    kwargs.setdefault("timeout_s", 30)
    return run_question(
        QUESTION, evidence_source, controller, llm, decision_model, retriever,
        run_id="run-test", dataset="hotpotqa", model="qwen3.5:4b", **kwargs,
    )


def four_condition_cases():
    yield "recall", "llm", FakeLLM([sub_q("Who founded Zorvex Labs?"), text("Mira Oduya founded it."),
                                   final("Tallinn"), text("Tallinn")]), None, None
    yield "rag", "llm", FakeLLM([sub_q("Who founded Zorvex Labs?"), final("Tallinn"), text("Tallinn")]), \
        None, FakeRetriever([[passage("p1"), passage("p2")]])
    yield "recall", "laya", FakeLLM([sub_q("Who founded Zorvex Labs?"), text("Mira Oduya founded it."),
                                    text("Tallinn")]), FakeDecision([decision(0.1), decision(0.9, "A", 0)]), None
    yield "rag", "laya", FakeLLM([sub_q("Who founded Zorvex Labs?"), text("Tallinn")]), \
        FakeDecision([decision(0.1), decision(0.9, "A", 0)]), FakeRetriever([[passage("p1"), passage("p2")]])


@pytest.mark.parametrize("evidence_source,controller,llm,decision_model,retriever", list(four_condition_cases()))
def test_each_condition_produces_valid_record(evidence_source, controller, llm, decision_model, retriever):
    record = run(evidence_source, controller, llm, decision_model, retriever)

    validate_record(record)
    assert record["condition"] == f"{evidence_source}_{controller}"
    assert (record["run_id"], record["dataset"], record["model"], record["question_id"]) == \
        ("run-test", "hotpotqa", "qwen3.5:4b", "q1")
    assert (record["gold"], record["gold_aliases"]) == ("Tallinn", ["Tallinn, Estonia"])
    assert (record["status"], record["prediction"], record["em"], record["f1"]) == ("done", "Tallinn", 1, 1.0)
    assert record["hops"] == 2 == len(record["steps"])
    assert record["steps"][0]["sub_question"] == "Who founded Zorvex Labs?"
    assert len(record["steps"][0]["evidence_added"]) == (2 if evidence_source == "rag" else 1)
    assert record["vram_peak_mb"] is None
    assert "finished_at" not in record
    has_laya = [("laya" in step) for step in record["steps"]]
    assert has_laya == ([True, True] if controller == "laya" else [False, False])
    assert not llm.script, "LLM 腳本應剛好用完"


def test_laya_sufficient_on_first_hop_answers_without_retrieval():
    llm = FakeLLM([text("Tallinn", prompt_tokens=40, output_tokens=2)])
    retriever = FakeRetriever([[passage("p1")]])
    laya = FakeDecision([decision(0.8, "A", 0, latency_ms=4.5)])

    record = run("rag", "laya", llm, laya, retriever)

    validate_record(record)
    assert record["status"] == "done" and record["hops"] == 1
    assert retriever.queries == []
    assert len(llm.calls) == 1 and llm.calls[0]["tools"] == []  # 只有作答呼叫
    step = record["steps"][0]
    assert (step["sub_question"], step["evidence_added"]) == ("", [])
    assert step["laya"] == {"sufficient_p": 0.8, "next_action": "A", "remaining": 0, "latency_ms": 4.5}
    assert (step["llm"]["prompt_tokens"], step["llm"]["output_tokens"]) == (40, 2)
    assert laya.states == [{"question": QUESTION["question"], "sub_questions": [], "evidence": []}]


def test_laya_threshold_is_inclusive_at_half():
    record = run("rag", "laya", FakeLLM([text("Tallinn")]), FakeDecision([decision(0.5)]), FakeRetriever([]))
    assert (record["status"], record["prediction"]) == ("done", "Tallinn")
    assert record["hops"] == 1 and record["steps"][0]["sub_question"] == ""


def test_laya_never_sufficient_forces_answer_at_max_hops():
    llm = FakeLLM([sub_q(f"q{i}") for i in range(1, 4)] + [text("Tallinn")])
    retriever = FakeRetriever([[passage(f"p{i}")] for i in range(1, 4)])
    laya = FakeDecision([decision(0.49, "C", 2)])

    record = run("rag", "laya", llm, laya, retriever, max_hops=3)

    validate_record(record)
    assert record["status"] == "done" and record["prediction"] == "Tallinn"
    assert record["hops"] == 3 and [s["hop"] for s in record["steps"]] == [1, 2, 3]
    assert retriever.queries == ["q1", "q2", "q3"]
    assert all(s["laya"]["sufficient_p"] == 0.49 for s in record["steps"])
    # Laya 決策餵回下一跳：第 3 跳看到前兩個子問題與兩段 evidence
    assert laya.states[2]["sub_questions"] == ["q1", "q2"]
    assert [p["pid"] for p in laya.states[2]["evidence"]] == ["p1", "p2"]


def test_laya_hint_is_attached_to_sub_question_prompt():
    llm = FakeLLM([sub_q("q1"), text("Tallinn")])
    run("rag", "laya", llm, FakeDecision([decision(0.2, "C", 2), decision(0.9)]), FakeRetriever([[passage("p1")]]))

    assert llm.calls[0]["tools"] == ["ask_sub_question"]
    prompt = llm.calls[0]["messages"][-1]["content"]
    assert "next action C" in prompt and "about 2 key piece(s)" in prompt


def test_llm_controller_final_answer_on_second_hop_ends_with_two_hops():
    llm = FakeLLM([sub_q("q1"), final("Tallinn"), text("Tallinn")])
    retriever = FakeRetriever([[passage("p1")], [passage("p2")]])

    record = run("rag", "llm", llm, None, retriever)

    assert record["status"] == "done" and record["hops"] == 2
    assert retriever.queries == ["q1"]
    assert record["steps"][1]["sub_question"] == "" and record["steps"][1]["evidence_added"] == []
    assert llm.calls[0]["tools"] == ["ask_sub_question", "final_answer"]


def test_llm_controller_without_final_answer_forced_to_answer_at_max_hops():
    llm = FakeLLM([sub_q("q1"), text("f1"), sub_q("q2"), text("f2"), text("Tallinn")])
    record = run("recall", "llm", llm, max_hops=2)
    assert record["status"] == "done" and record["hops"] == 2 and record["prediction"] == "Tallinn"
    assert [e["text"] for s in record["steps"] for e in s["evidence_added"]] == ["f1", "f2"]


def test_rag_evidence_takes_top3_dedups_pid_and_caps_at_eight():
    results = [
        [passage("p1"), passage("p2"), passage("p3"), passage("p99")],  # 只取前 3
        [passage("p2"), passage("p4"), passage("p5")],  # p2 重複
        [passage("p6"), passage("p7"), passage("p6")],  # 同批內重複
        [passage("p1"), passage("p8"), passage("p9")],  # p1 重複；只剩 1 格，p9 被截斷
    ]
    llm = FakeLLM([sub_q(f"q{i}") for i in range(1, 5)] + [text("Tallinn")])

    record = run("rag", "llm", llm, None, FakeRetriever(results), max_hops=4)

    added = [[p["pid"] for p in s["evidence_added"]] for s in record["steps"]]
    assert added == [["p1", "p2", "p3"], ["p4", "p5"], ["p6", "p7"], ["p8"]]
    all_pids = [pid for pids in added for pid in pids]
    assert len(all_pids) == 8 == len(set(all_pids))
    assert record["steps"][0]["evidence_added"][0] == {"pid": "p1", "title": "Title p1", "text": "Text of p1."}
    # 最終作答 prompt 內恰有 8 段 evidence
    assert llm.calls[-1]["messages"][-1]["content"].count("\n[") == 8


@pytest.mark.parametrize("bad_reply", [
    tool_reply("search_web", {"query": "x"}),  # 未知工具名
    tool_reply("ask_sub_question", {"question": "x"}),  # 缺必要參數
    tool_reply("ask_sub_question", {"sub_question": 42}),  # 參數型別錯誤
    tool_reply("ask_sub_question", {"sub_question": "  "}),  # 空字串
    tool_reply("ask_sub_question", '{"sub_question": "x"'),  # 非 JSON
    tool_reply("ask_sub_question", '["x"]'),  # JSON 但不是物件
    text("I think the answer is Tallinn."),  # 需要工具時只回文字
    InvalidToolCallError("server: error parsing tool call"),  # 伺服器無法解析
], ids=["unknown-name", "missing-arg", "wrong-type", "empty", "non-json", "json-not-object", "no-tool-call",
        "server-parse-error"])
def test_bad_tool_call_yields_invalid_tool(bad_reply):
    llm = FakeLLM([sub_q("q1"), bad_reply])
    record = run("rag", "llm", llm, None, FakeRetriever([[passage("p1")]]))

    validate_record(record)
    assert record["status"] == "invalid_tool"
    assert (record["prediction"], record["em"], record["f1"]) == ("", 0, 0.0)
    assert record["hops"] == 2 and record["steps"][0]["sub_question"] == "q1"
    assert record["error"]


def test_final_answer_tool_is_invalid_under_laya_controller():
    llm = FakeLLM([final("Tallinn")])
    record = run("recall", "laya", llm, FakeDecision([decision(0.1)]))
    assert record["status"] == "invalid_tool" and "final_answer" in record["error"]


def test_llm_sleeping_past_timeout_yields_timeout_with_latency():
    llm = FakeLLM([sub_q("q1"), sub_q("q2"), text("Tallinn")], sleep_s=0.3)
    record = run("rag", "llm", llm, None, FakeRetriever([[passage("p1")]]), timeout_s=0.5)

    validate_record(record)
    assert record["status"] == "timeout"
    assert (record["prediction"], record["em"]) == ("", 0)
    assert 500 <= record["latency_ms"] < 2000
    assert record["hops"] == 2 and record["steps"][0]["sub_question"] == "q1"  # 保留逾時前的 trace


def slow_sync_cases():
    # (evidence_source, controller, llm, decision, retriever, 慢的物件, 逾時發生點)
    slow_decision = FakeDecision([decision(0.9)], sleep_s=0.6)
    yield "recall", "laya", FakeLLM([text("Tallinn")]), slow_decision, None, slow_decision, "decision.decide"
    slow_retriever = FakeRetriever([[passage("p1")]], sleep_s=0.6)
    yield "rag", "llm", FakeLLM([sub_q("q1"), final("x"), text("Tallinn")]), None, slow_retriever, slow_retriever, \
        "retriever.retrieve"


@pytest.mark.parametrize("evidence_source,controller,llm,decision_model,retriever,slow,phase", list(slow_sync_cases()),
                         ids=["slow-decision", "slow-retriever"])
def test_timeout_inside_sync_worker_waits_for_it_and_records_wall_time(
        evidence_source, controller, llm, decision_model, retriever, slow, phase):
    started = time.perf_counter()
    record = run(evidence_source, controller, llm, decision_model, retriever, timeout_s=0.2)
    wall_ms = (time.perf_counter() - started) * 1000

    validate_record(record)
    assert record["status"] == "timeout"
    assert slow.completed == 1, "回傳前必須等逾時中的同步 worker 結束"
    assert not [t for t in threading.enumerate() if t.name.startswith("multihop-sync")], "不得殘留 worker 執行緒"
    assert 600 <= record["latency_ms"] <= wall_ms < 2000  # 記實際 wall time（含等待 worker）
    assert phase in record["error"] and "hop 1" in record["error"]


def test_timeout_in_persistent_event_loop_leaves_no_overlapping_worker():
    slow = FakeDecision([decision(0.9)], sleep_s=0.5)

    async def main():
        record = await arun_question(QUESTION, "recall", "laya", FakeLLM([]), slow, None, timeout_s=0.1,
                                     run_id="run-test", dataset="hotpotqa", model="qwen3.5:4b")
        return record, slow.completed  # 下一題開始前的狀態

    record, completed_at_return = asyncio.run(main())
    assert record["status"] == "timeout" and completed_at_return == 1
    assert record["latency_ms"] >= 500


def test_timeout_in_llm_call_names_the_phase():
    record = run("rag", "llm", FakeLLM([sub_q("q1")], sleep_s=0.5), None, FakeRetriever([]), timeout_s=0.2)
    assert record["status"] == "timeout" and "llm.chat" in record["error"] and "hop 1" in record["error"]
    assert record["latency_ms"] < 450  # async LLM 呼叫可直接取消，不需等待


def test_step_output_tokens_sum_to_all_llm_calls_with_final_answer_in_last_hop():
    llm = FakeLLM([
        sub_q("q1", prompt_tokens=100, output_tokens=7), text("f1", prompt_tokens=50, output_tokens=11),
        sub_q("q2", prompt_tokens=120, output_tokens=13), text("f2", prompt_tokens=60, output_tokens=17),
        final("Tallinn", prompt_tokens=140, output_tokens=19), text("Tallinn", prompt_tokens=90, output_tokens=23),
    ])
    record = run("recall", "llm", llm)

    steps = record["steps"]
    assert sum(s["llm"]["output_tokens"] for s in steps) == llm.total_output_tokens == 90
    assert sum(s["llm"]["prompt_tokens"] for s in steps) == llm.total_prompt_tokens == 560
    assert [s["llm"]["output_tokens"] for s in steps] == [7 + 11, 13 + 17, 19 + 23]
    assert [s["llm"]["latency_ms"] for s in steps] == [2.0, 2.0, 2.0]


def test_decision_output_is_normalized():
    laya = FakeDecision([decision(0.3, "b", 3.7, latency_ms=2.0), decision(1.2, 2, -1)])
    llm = FakeLLM([sub_q("q1"), text("Tallinn")])
    record = run("rag", "laya", llm, laya, FakeRetriever([[passage("p1")]]))

    validate_record(record)
    assert [(s["laya"]["next_action"], s["laya"]["remaining"]) for s in record["steps"]] == [("B", 2), ("C", 0)]
    assert record["steps"][1]["laya"]["sufficient_p"] == 1.0


def test_unrecognized_decision_yields_error_status():
    record = run("rag", "laya", FakeLLM([]), FakeDecision([decision(0.3, "Z")]), FakeRetriever([]))
    assert record["status"] == "error" and "next_action" in record["error"]


def test_out_of_memory_yields_oom_status():
    llm = FakeLLM([LLMOutOfMemoryError("CUDA error: out of memory")])
    record = run("recall", "llm", llm)
    validate_record(record)
    assert record["status"] == "oom" and "out of memory" in record["error"]


def test_other_exception_yields_error_status():
    llm = FakeLLM([sub_q("q1"), RuntimeError("boom")])
    record = run("recall", "llm", llm)
    validate_record(record)
    assert record["status"] == "error" and record["error"] == "RuntimeError: boom" and record["hops"] == 1


def test_connection_failure_propagates_to_abort_run():
    with pytest.raises(LLMConnectionError):
        run("recall", "llm", FakeLLM([LLMConnectionError("down")]))


def test_condition_argument_must_match_switches():
    with pytest.raises(ValueError, match="condition"):
        run("rag", "llm", FakeLLM([]), None, FakeRetriever([]), condition="recall_llm")
