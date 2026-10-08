# Data and result schema

## Frozen inputs

`data/datasets/{hotpotqa,2wiki,musique}/` contains the exact normalized JSONL used
in the study, not the complete upstream datasets:

| File | Questions per dataset | Use |
|---|---:|---|
| `sample_200.jsonl` | 200 | QA evaluation, sampled from validation |
| `train_800.jsonl` | 800 | Simulate controller training states |
| `calibration_100.jsonl` | 100 | Controller temperature calibration |
| `test_100.jsonl` | 100 | Held-out controller evaluation |
| `corpus.jsonl` | Varies | Deduplicated passages from the 200 evaluation questions |
| `prepare_manifest.json` | — | Sources, seed, counts, and file hashes |

The three train-derived splits are disjoint. The evaluation corpus combines supporting
and distractor paragraphs from the 200 validation questions; it is **not all Wikipedia**.
An evaluation question has `id`, `question`, `answer`, `answer_aliases`, and `paragraphs`.
Each paragraph has `pid`, `title`, `text`, `is_supporting`, and `supporting_sentences`.
The paragraph ID is the first 16 hex digits of SHA256(title + newline + text).

## Predictions and traces

`data/runs/main-20260930/records/` has one JSONL file per configuration.
Each row contains:

| Field | Meaning |
|---|---|
| `dataset`, `model`, `condition`, `question_id` | Join keys |
| `gold`, `gold_aliases`, `prediction` | Reference and generated answers |
| `em`, `f1` | Per-question scores in [0, 1] |
| `status` | `done`, `invalid_tool`, `error`, or a recorded earlier failure |
| `steps` | Sub-questions, retrieved/remembered evidence, controller decisions, tokens |
| `hops` | Executed loop steps, including a final-answer step when applicable |
| `latency_ms`, `vram_peak_mb` | Original observed runtime measurements |
| `finished_at` | Original completion timestamp in UTC |

Files retain retry history. The last row for the same dataset/model/condition/question
is authoritative. There are 12,000 final records: **600 distinct evaluation questions,
each evaluated under 20 model/condition combinations**. They are not 12,000 distinct
questions. Failed final records count as zero EM/F1; they remain in the denominator.

`summary.csv` has 60 rows. EM/F1 are percentages (0–100), rates are proportions (0–1),
and latency is in milliseconds. Tokens are summed across each question's LLM calls
then averaged; VRAM is the configuration maximum. `laya_latency_ms_mean` is per
controller call, not per question. See `results_aggregator.py` for exact formulas.

## Statistics and limitations

Paired bootstrap resamples the same 200 question indices across four conditions,
10,000 times, seed 20261001. Confidence intervals capture uncertainty over sampled
questions, not random seeds, repeated training runs, or machines.

Error taxonomy covers wrong answers in the two RAG conditions. It is a rule-based
diagnostic using supporting-paragraph annotations, not a human causal judgment.
Order: invalid tool → timeout → other execution error → evidence covered → premature
stop → retrieval miss → other. `dataset=all` pools the three datasets (600 questions
per model/condition). See `paper/analysis/error_taxonomy.py`.

The original experiment retried two timeout questions (27B / 2Wiki / RAG+LLM), one twice;
both eventually finished. Final timeout rates therefore equal zero. These additional
retries are part of the archived run, not evidence that timeout never occurred.
The historical decision-layer report provides aggregate metrics, not per-decision outputs.

## Redistribution and attribution

Dataset transformations here comprise deterministic sampling, normalization, passage
hashing, corpus deduplication, and retrieval/answer traces. Original questions, answer
annotations, and retrieved passages retain their upstream terms. No common repository
code license overrides those terms.

| Material and upstream authors | Original source | Terms |
|---|---|---|
| HotpotQA — Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, Christopher D. Manning | [Project](https://hotpotqa.github.io/), [HF input](https://huggingface.co/datasets/hotpotqa/hotpot_qa) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| 2WikiMultihopQA — Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, Akiko Aizawa | [Project and license](https://github.com/Alab-NII/2wikimultihop), [HF input](https://huggingface.co/datasets/framolfese/2WikiMultihopQA) | [Apache-2.0](../LICENSES/2Wiki-Apache-2.0.txt) |
| MuSiQue — Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, Ashish Sabharwal | [Project and license](https://github.com/StonyBrookNLP/musique), [HF input](https://huggingface.co/datasets/dgslibisey/MuSiQue) | [CC BY 4.0](../LICENSES/MuSiQue-CC-BY-4.0.txt) |

HotpotQA-derived data and trace text are shared under CC BY-SA 4.0. MuSiQue-derived
material retains CC BY 4.0. 2Wiki-derived material retains its upstream notices;
its Wikipedia passages may carry additional attribution/share-alike terms.
Passage titles, original question IDs, and upstream links are retained for provenance.
Upstream papers are cited in `paper/refs.bib`. The sources above were checked on 2026-10-09.

Dependencies, pretrained checkpoints, and paper-format files have their own licenses;
see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
