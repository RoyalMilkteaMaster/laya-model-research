"""真實 Ollama＋GPU：qwen3.5:4b、recall＋llm，跑 HotpotQA sample_200 前 3 題（`pytest -m integration -s` 可看摘要）。

只檢查 record 合規與 trace 形狀，不依賴模型輸出文字。
"""

from __future__ import annotations

import json

import pytest

from multihop_benchmark.agent.multihop_agent_loop import run_question
from multihop_benchmark.agent.ollama_llm_client import OllamaLLMClient
from multihop_benchmark.runs.record_store import validate_record
from multihop_benchmark.settings import load_settings

MODEL = "qwen3.5:4b"


@pytest.mark.integration
def test_recall_llm_on_three_hotpotqa_questions_with_real_ollama():
    settings = load_settings()
    sample = settings.data_root / "datasets" / "hotpotqa" / "sample_200.jsonl"
    with sample.open(encoding="utf-8") as fh:
        questions = [json.loads(next(fh)) for _ in range(3)]
    llm = OllamaLLMClient(MODEL, settings.ollama_host)

    records = [
        run_question(q, "recall", "llm", llm, None, None, max_hops=4, timeout_s=300,
                     run_id="integration-t08", dataset="hotpotqa", model=MODEL, condition="recall_llm")
        for q in questions
    ]

    for q, record in zip(questions, records):
        validate_record(record)
        assert record["question_id"] == q["id"] and record["condition"] == "recall_llm"
        assert record["model"] == MODEL
        assert record["steps"], record
        assert sum(s["llm"]["prompt_tokens"] for s in record["steps"]) > 0
        assert sum(s["llm"]["output_tokens"] for s in record["steps"]) > 0
        assert all(s["llm"]["latency_ms"] > 0 for s in record["steps"])
        assert record["latency_ms"] > 0
        tokens = [(s["llm"]["prompt_tokens"], s["llm"]["output_tokens"], s["llm"]["latency_ms"]) for s in record["steps"]]
        print(f"\n{record['question_id']} status={record['status']} hops={record['hops']} em={record['em']} "
              f"f1={record['f1']:.2f} latency_ms={record['latency_ms']:.0f} "
              f"steps(prompt,output,latency_ms)={tokens} prediction={record['prediction']!r} "
              f"gold={record['gold']!r} error={record.get('error')!r}")
