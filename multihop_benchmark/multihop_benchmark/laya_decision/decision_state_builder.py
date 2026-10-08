"""Laya state 的唯一組裝點；弱標籤產生（weak_label_generator）、Laya 推論（laya_decision_client）與
迴圈（multihop_agent_loop）都呼叫 `build_state`，訓練與推論看到同一種文字。

State 文字格式（英文，與 Laya 的三個固定問題同語言）：

    Question: <question>
    Sub-questions asked:
    1. <sub_question>
    ...
    Evidence:
    [1] <title>: <text>
    [2] <text>                      # 純字串 evidence 沒有 title

沒有子問題或 evidence 時該節寫 `(none yet)`。

- evidence 元素可為 `{pid, title, text}` dict（迴圈傳入的格式；只用 title 與 text）或純字串。
- 每段 evidence 先把空白壓成單一空格，再只保留前 2 句（`first_sentences`）。斷句規則：`.`／`!`／`?`
  （可接引號或右括號）後接空白，或 `。`／`！`／`？`。這是簡單規則，`U.S. Army` 之類縮寫會被誤斷。
- token 量測：`len(tokenizer.encode(state, add_special_tokens=False))`，預設 tokenizer 為
  `convaiinnovations/laya-multilingual` 的 `tokenizer/tokenizer.json`（與 `laya.load()` 同一檔）。
  測試可注入任何有 `encode(text, add_special_tokens=False)` 的物件。
- 截斷策略（確定性，同輸入必同輸出）：
  1. 未超過上限（預設 900）時原樣回傳。
  2. 超過時，對每段 evidence 的 text 套同一個「最多 w 個空白分隔詞」上限（標題、問題、子問題不動），以二分搜尋
     取仍 ≤ 上限的最大 w。所有 evidence 都保留（至少保留標題），不會因截斷而整段消失。
  3. w = 0 仍超過（問題或子問題本身過長）時，把整個 state 從尾端截到仍 ≤ 上限的最長詞數前綴（保留換行），
     此時尾端的子問題或 evidence 可能被截掉。
  二分搜尋只接受實測 ≤ 上限的結果，因此回傳值一定 ≤ 上限。
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from typing import Any, Protocol

LAYA_CHECKPOINT = "convaiinnovations/laya-multilingual"
MAX_STATE_TOKENS = 900
MAX_SENTENCES = 2
NONE_YET = "(none yet)"

_SENTENCE_END = re.compile(r"[.!?][\"'”’)\]]*\s+|[。！？][」』”’）)]*\s*")
_WORD = re.compile(r"\S+")


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = ...) -> list[Any]: ...


@lru_cache(maxsize=1)
def load_laya_tokenizer() -> Tokenizer:
    """載入 Laya checkpoint 的 fast tokenizer（經 `HF_HOME` 快取；只下載 tokenizer.json，不需 GPU）。"""
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import hf_hub_download
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast(tokenizer_file=hf_hub_download(LAYA_CHECKPOINT, "tokenizer/tokenizer.json"))


def count_tokens(text: str, tokenizer: Tokenizer | None = None) -> int:
    tokenizer = tokenizer or load_laya_tokenizer()
    return len(tokenizer.encode(text, add_special_tokens=False))


def first_sentences(text: str, n: int = MAX_SENTENCES) -> str:
    """空白正規化後保留前 n 句；句數不足 n 時回傳全文。"""
    text = " ".join(str(text).split())
    for i, match in enumerate(_SENTENCE_END.finditer(text), start=1):
        if i == n:
            return text[: match.end()].strip()
    return text


def _evidence_item(item: Mapping[str, Any] | str) -> tuple[str, str]:
    if isinstance(item, Mapping):
        return " ".join(str(item.get("title") or "").split()), first_sentences(item.get("text") or "")
    return "", first_sentences(item)


def _assemble(question: str, sub_questions: Sequence[str], items: Sequence[tuple[str, str]]) -> str:
    lines = [f"Question: {question}"]
    if sub_questions:
        lines.append("Sub-questions asked:")
        lines += [f"{i}. {q}" for i, q in enumerate(sub_questions, start=1)]
    else:
        lines.append(f"Sub-questions asked: {NONE_YET}")
    if items:
        lines.append("Evidence:")
        lines += [f"[{i}] {title}: {text}".rstrip() if title else f"[{i}] {text}".rstrip()
                  for i, (title, text) in enumerate(items, start=1)]
    else:
        lines.append(f"Evidence: {NONE_YET}")
    return "\n".join(lines)


def _largest_fitting(high: int, fits: Callable[[int], bool]) -> int | None:
    """回傳 [0, high] 中使 fits 為真的最大值（二分搜尋）；0 也不符合時回傳 None。"""
    if not fits(0):
        return None
    low = 0
    while low < high:
        mid = (low + high + 1) // 2
        if fits(mid):
            low = mid
        else:
            high = mid - 1
    return low


def build_state(
    question: str,
    sub_questions: Sequence[str],
    evidence: Sequence[Mapping[str, Any] | str],
    *,
    tokenizer: Tokenizer | None = None,
    max_tokens: int = MAX_STATE_TOKENS,
) -> str:
    """組裝 Laya state 文字並截到 ≤ `max_tokens`（格式與截斷策略見模組說明）。"""
    if max_tokens <= 0:
        raise ValueError(f"max_tokens 必須為正整數：{max_tokens}")
    tokenizer = tokenizer or load_laya_tokenizer()
    question = " ".join(str(question).split())
    sub_questions = [" ".join(str(q).split()) for q in sub_questions]
    items = [_evidence_item(item) for item in evidence]

    def fits(text: str) -> bool:
        return count_tokens(text, tokenizer) <= max_tokens

    state = _assemble(question, sub_questions, items)
    if fits(state):
        return state

    words = [text.split() for _, text in items]

    def with_word_cap(cap: int) -> str:
        return _assemble(question, sub_questions, [(title, " ".join(w[:cap])) for (title, _), w in zip(items, words)])

    cap = _largest_fitting(max((len(w) for w in words), default=0), lambda c: fits(with_word_cap(c)))
    if cap is not None:
        return with_word_cap(cap)

    state = with_word_cap(0)
    ends = [match.end() for match in _WORD.finditer(state)]
    keep = _largest_fitting(len(ends), lambda k: fits(state[: ends[k - 1]] if k else ""))
    return state[: ends[keep - 1]] if keep else ""
