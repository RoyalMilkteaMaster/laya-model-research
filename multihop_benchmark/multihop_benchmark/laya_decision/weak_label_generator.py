"""狀態模擬弱標籤：Data Root `datasets/<dataset>/{train_800,calibration_100,test_100}.jsonl` →
`laya_decisions/{train,calibration,test}.jsonl`（三集合併，依 hotpotqa、2wiki、musique 與檔內題序輸出）。

每行沿用 `laya_zh_tw/02_convert_dataset.py` 的 `{state, type, instructions, criteria, target, label}`；
三個固定問題見 `DECISION_QUESTIONS`，Laya 推論端（laya_decision_client）必須用同一組文字。

狀態（每題以黃金段 G = `is_supporting` 段落、干擾段 D = 其餘段落組成）：

- `question_only`：沒有子問題與 evidence。
- `partial_gold`：隨機 1..|G|-1 段黃金 + 0–2 段干擾。
- `full_gold`：全部黃金 + 0–2 段干擾。
- `distractor_only`：1–3 段干擾。
- evidence 一律打亂順序；子問題為每段 evidence 一句 `What is known about <title>?`（黃金、干擾同一模板，
  不洩漏標籤）。

標籤規則（Spec）：`sufficient` = 黃金全到齊；`next_action` = 全到齊→A、只有干擾→C、其餘（缺黃金）→B；
`remaining_hops` = min(缺的黃金段數, 2)。`question_only` 缺全部黃金、沒有干擾，因此為 B、remaining 2。

evidence 兩種寫法（同一組段落、同一組標籤各產一份；`question_only` 沒有 evidence，只產一份，寫法記為 `none`）：

- `full_paragraph`：段落全文（`build_state` 再截前 2 句），模擬 RAG 檢索到的段落。
- `supporting_only`：`supporting_sentences` 為非空清單時只用這些句子（模擬 recall 條件的短事實）；
  為 `null`（MuSiQue 沒有句子層級標註）時退回整段，該筆計入統計的 `fallback_records`；
  為 `[]`（HotpotQA／2Wiki 的干擾段）時同樣用整段，不計入 fallback。

每題取樣（固定 seed，`random.Random(f"{seed}|{dataset}|{id}")`，與檔案順序無關）：50% 機率含 `question_only`，
再從可行的三種 evidence 狀態中不重複取 2 種。每題平均 (0.5 + 2×2) × 3 = 13.5 筆，train 2400 題約 3.2 萬筆。
|G| = 0 的題目略過；|G| = 1 不產生 `partial_gold`；沒有干擾段時不產生 `distractor_only`。
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from multihop_benchmark.datasets.multihop_dataset_loader import DATASET_SPECS, write_jsonl
from multihop_benchmark.laya_decision.decision_state_builder import (
    MAX_STATE_TOKENS,
    Tokenizer,
    build_state,
    count_tokens,
    load_laya_tokenizer,
)

SEED = 42
OUTPUT_DIR_NAME = "laya_decisions"
SPLITS = {"train": "train_800", "calibration": "calibration_100", "test": "test_100"}
DATASETS = tuple(DATASET_SPECS)
STATE_KINDS = ("question_only", "partial_gold", "full_gold", "distractor_only")
EVIDENCE_KINDS = STATE_KINDS[1:]
WRITINGS = ("full_paragraph", "supporting_only")
NO_WRITING = "none"
QUESTION_ONLY_PROBABILITY = 0.5
EVIDENCE_KINDS_PER_QUESTION = 2
MAX_EXTRA_DISTRACTORS = 2
MAX_DISTRACTOR_ONLY = 3
ACTIONS = ("A", "B", "C")

DECISION_QUESTIONS: dict[str, dict[str, Any]] = {
    "sufficient": {
        "type": "noul",
        "instructions": "Is the current evidence enough to answer the original question?",
        "criteria": {
            "false": "no, key evidence is still missing",
            "true": "yes, the evidence is enough to answer the original question",
        },
    },
    "next_action": {
        "type": "choice",
        "instructions": "What should the agent do next?",
        "criteria": {
            "A": "answer now",
            "B": "decompose and search another sub-question",
            "C": "rephrase the query and search again",
        },
    },
    "remaining_hops": {
        "type": "score",
        "instructions": "How many key pieces of evidence are still missing to answer the original question?",
        "criteria": ["none are missing", "one is missing", "two or more are missing"],
    },
}


@dataclass(frozen=True)
class Labels:
    sufficient: bool
    next_action: str
    remaining_hops: int


@dataclass(frozen=True)
class Decision:
    """一筆輸出 record 與只用於統計的來源資訊（不寫入檔案）。"""

    record: dict[str, Any]
    kind: str
    writing: str
    gold_total: int
    gold_present: int
    fallback: bool


def labels_for(gold_total: int, gold_present: int, distractors: int) -> Labels:
    """Spec 弱標籤規則。"""
    if gold_present == gold_total:
        action = "A"
    elif gold_present == 0 and distractors > 0:
        action = "C"
    else:
        action = "B"
    return Labels(gold_present == gold_total, action, min(gold_total - gold_present, 2))


def _one_hot(index: int, size: int) -> list[float]:
    return [1.0 if i == index else 0.0 for i in range(size)]


def _records(state: str, labels: Labels) -> list[dict[str, Any]]:
    targets = {
        "sufficient": ([0.0, 1.0] if labels.sufficient else [1.0, 0.0], "true" if labels.sufficient else "false"),
        "next_action": (_one_hot(ACTIONS.index(labels.next_action), 3), labels.next_action),
        "remaining_hops": (_one_hot(labels.remaining_hops, 3), labels.remaining_hops),
    }
    return [
        {"state": state, **DECISION_QUESTIONS[name], "target": target, "label": label}
        for name, (target, label) in targets.items()
    ]


def _paragraph_text(paragraph: Mapping[str, Any], writing: str) -> tuple[str, bool]:
    """回傳 (text, 是否因 supporting_sentences=null 退回整段)。"""
    sentences = paragraph.get("supporting_sentences")
    if writing == "supporting_only" and sentences:
        return " ".join(sentences), False
    return paragraph["text"], writing == "supporting_only" and sentences is None


def _feasible_kinds(gold: Sequence[Any], distractors: Sequence[Any]) -> list[str]:
    return [kind for kind in EVIDENCE_KINDS
            if not (kind == "partial_gold" and len(gold) < 2)
            and not (kind == "distractor_only" and not distractors)]


def _simulate(row: Mapping[str, Any], kind: str, rng: random.Random) -> list[Mapping[str, Any]]:
    gold = [p for p in row["paragraphs"] if p["is_supporting"]]
    distractors = [p for p in row["paragraphs"] if not p["is_supporting"]]
    if kind == "question_only":
        return []
    if kind == "partial_gold":
        chosen = rng.sample(gold, rng.randint(1, len(gold) - 1))
        chosen += rng.sample(distractors, rng.randint(0, min(MAX_EXTRA_DISTRACTORS, len(distractors))))
    elif kind == "full_gold":
        chosen = list(gold) + rng.sample(distractors, rng.randint(0, min(MAX_EXTRA_DISTRACTORS, len(distractors))))
    elif kind == "distractor_only":
        chosen = rng.sample(distractors, rng.randint(1, min(MAX_DISTRACTOR_ONLY, len(distractors))))
    else:
        raise ValueError(f"未知狀態：{kind}")
    rng.shuffle(chosen)
    return chosen


def decisions_for_state(
    row: Mapping[str, Any], kind: str, rng: random.Random, *, tokenizer: Tokenizer | None = None,
) -> list[Decision]:
    """對一題產生指定狀態的 decisions：evidence 狀態兩種寫法各 3 筆，`question_only` 3 筆。"""
    gold_total = sum(1 for p in row["paragraphs"] if p["is_supporting"])
    paragraphs = _simulate(row, kind, rng)
    gold_present = sum(1 for p in paragraphs if p["is_supporting"])
    labels = labels_for(gold_total, gold_present, len(paragraphs) - gold_present)
    sub_questions = [f"What is known about {p['title']}?" for p in paragraphs]
    decisions = []
    for writing in WRITINGS if paragraphs else (NO_WRITING,):
        texts = [_paragraph_text(p, writing) for p in paragraphs]
        evidence = [{"pid": p["pid"], "title": p["title"], "text": text} for p, (text, _) in zip(paragraphs, texts)]
        state = build_state(row["question"], sub_questions, evidence, tokenizer=tokenizer)
        fallback = any(fell_back for _, fell_back in texts)
        decisions += [Decision(record, kind, writing, gold_total, gold_present, fallback)
                      for record in _records(state, labels)]
    return decisions


def question_decisions(
    row: Mapping[str, Any], *, dataset: str, seed: int = SEED, tokenizer: Tokenizer | None = None,
) -> list[Decision]:
    """一題的全部 decisions；取樣只依 (seed, dataset, id)，與題目在檔案中的位置無關。"""
    gold = [p for p in row["paragraphs"] if p["is_supporting"]]
    if not gold:
        return []
    distractors = [p for p in row["paragraphs"] if not p["is_supporting"]]
    rng = random.Random(f"{seed}|{dataset}|{row['id']}")
    kinds = ["question_only"] if rng.random() < QUESTION_ONLY_PROBABILITY else []
    feasible = _feasible_kinds(gold, distractors)
    picked = set(rng.sample(feasible, min(EVIDENCE_KINDS_PER_QUESTION, len(feasible))))
    kinds += [kind for kind in EVIDENCE_KINDS if kind in picked]
    return [d for kind in kinds for d in decisions_for_state(row, kind, rng, tokenizer=tokenizer)]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_split_questions(data_root: Path, split: str) -> dict[str, list[dict[str, Any]]]:
    """讀三集的某個切分（train／calibration／test）；任一檔不存在時丟 FileNotFoundError。"""
    paths = {dataset: Path(data_root) / "datasets" / dataset / f"{SPLITS[split]}.jsonl" for dataset in DATASETS}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"缺少 prepare-data 輸出：{', '.join(missing)}")
    return {dataset: _read_jsonl(path) for dataset, path in paths.items()}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_laya_data(
    data_root: Path, *, seed: int = SEED, tokenizer: Tokenizer | None = None,
) -> dict[str, Any]:
    """產生三個切分檔並回傳統計；state 超過 token 上限時丟 ValueError（不應發生，build_state 已截斷）。

    回傳 `{"seed", "output_dir", "splits": {split: {questions, skipped_questions, records, counts, fallback_records,
    label_counts, max_state_tokens, sha256}}}`；`counts` 的鍵為 (狀態, 型別, 寫法)。
    """
    tokenizer = tokenizer or load_laya_tokenizer()
    questions = {split: load_split_questions(data_root, split) for split in SPLITS}  # 先確認輸入齊全再寫檔
    output_dir = Path(data_root) / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"seed": seed, "output_dir": str(output_dir), "splits": {}}
    for split, by_dataset in questions.items():
        decisions: list[Decision] = []
        skipped = 0
        for dataset, rows in by_dataset.items():
            for row in rows:
                produced = question_decisions(row, dataset=dataset, seed=seed, tokenizer=tokenizer)
                skipped += not produced
                decisions += produced
        state_tokens = {s: count_tokens(s, tokenizer) for s in {d.record["state"] for d in decisions}}
        max_tokens = max(state_tokens.values(), default=0)
        if max_tokens > MAX_STATE_TOKENS:
            raise ValueError(f"{split}: state 最長 {max_tokens} tokens，超過上限 {MAX_STATE_TOKENS}")
        path = output_dir / f"{split}.jsonl"
        write_jsonl(path, (d.record for d in decisions))
        summary["splits"][split] = {
            "questions": sum(len(rows) for rows in by_dataset.values()),
            "skipped_questions": skipped,
            "records": len(decisions),
            "counts": Counter((d.kind, d.record["type"], d.writing) for d in decisions),
            "fallback_records": sum(d.fallback for d in decisions),
            "label_counts": Counter((d.record["type"], str(d.record["label"])) for d in decisions),
            "max_state_tokens": max_tokens,
            "sha256": _sha256(path),
        }
    return summary


# ---- 抽驗：只從 state 文字與原始題目重新推導標籤，不使用產生時的內部資訊 ----

_EVIDENCE_LINE = re.compile(r"^\[(\d+)\] (.*)$")


def _norm(text: str) -> str:
    return " ".join(str(text).split())


def _resolve(body: str, paragraphs: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """由 `<title>: <text>` 找回原段落：取標題相符者中最長的標題；同標題多段時要求 evidence 文字是該段全文或支持句的前綴。

    仍無法區分的多段（例如 MuSiQue 同一題的重複干擾段）若 `is_supporting` 相同，對標籤無影響，取第一段。"""
    candidates = [p for p in paragraphs if body.startswith(_norm(p["title"]) + ":") or body == _norm(p["title"])]
    if not candidates:
        return None
    longest = max(len(_norm(p["title"])) for p in candidates)
    candidates = [p for p in candidates if len(_norm(p["title"])) == longest]
    if len(candidates) > 1:
        text = body[longest + 1:].strip()  # build_state 輸出的是正規化文字的前綴（前 2 句或截斷後的前幾詞）
        candidates = [p for p in candidates if _norm(p["text"]).startswith(text)
                      or _norm(" ".join(p.get("supporting_sentences") or [])).startswith(text)]
    return candidates[0] if candidates and len({p["is_supporting"] for p in candidates}) == 1 else None


def _audit_expected(state: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    lines = state.splitlines()
    evidence_lines = [m.group(2) for m in map(_EVIDENCE_LINE.match, lines) if m]
    for row in rows:
        resolved = [_resolve(body, row["paragraphs"]) for body in evidence_lines]
        if any(p is None for p in resolved):
            continue
        gold_total = sum(1 for p in row["paragraphs"] if p["is_supporting"])
        gold_present = len({p["pid"] for p in resolved if p["is_supporting"]})
        distractors = sum(1 for p in resolved if not p["is_supporting"])
        # 依 Spec 文字獨立重寫的規則（刻意不呼叫 labels_for）。
        if gold_present == gold_total:
            action = "A"
        elif gold_present == 0 and distractors:
            action = "C"
        else:
            action = "B"
        return {"noul": "true" if gold_present == gold_total else "false", "choice": action,
                "score": min(gold_total - gold_present, 2), "gold_total": gold_total, "gold_present": gold_present}
    return None


def audit_labels(
    records: Sequence[Mapping[str, Any]], questions: Mapping[str, Iterable[Mapping[str, Any]]], *,
    sample_size: int = 100, seed: int = 0,
) -> dict[str, Any]:
    """隨機抽 `sample_size` 筆，從 state 的問題行找回原題、從 evidence 行找回段落，重新推導標籤並與 label、
    target 的 argmax 比對。回傳 `{checked, by_type, mismatches[{index, type, reason, expected, label}]}`。"""
    by_question: dict[str, list[Mapping[str, Any]]] = {}
    for rows in questions.values():
        for row in rows:
            by_question.setdefault(_norm(row["question"]), []).append(row)
    indices = sorted(random.Random(seed).sample(range(len(records)), min(sample_size, len(records))))
    mismatches = []
    by_type: Counter[str] = Counter()
    for index in indices:
        record = records[index]
        by_type[record["type"]] += 1
        first = record["state"].splitlines()[0]
        rows = by_question.get(_norm(first.removeprefix("Question: ")), [])
        expected = _audit_expected(record["state"], rows)
        if expected is None:
            mismatches.append({"index": index, "type": record["type"], "reason": "無法由 state 找回原題或段落",
                               "expected": None, "label": record["label"]})
            continue
        want = expected[record["type"]]
        options = {"noul": ["false", "true"], "choice": list(ACTIONS), "score": [0, 1, 2]}[record["type"]]
        target = list(record["target"])
        target_label = options[target.index(max(target))]
        if record["label"] != want or target_label != want:
            mismatches.append({"index": index, "type": record["type"], "reason": "標籤不符規則",
                               "expected": want, "label": record["label"]})
    return {"checked": len(indices), "by_type": dict(by_type), "mismatches": mismatches}
