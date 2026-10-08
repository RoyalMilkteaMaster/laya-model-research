"""EM／F1，採 HotpotQA 官方正規化；gold 與 aliases 取最大值（MuSiQue）。"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Iterable

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = set(string.punctuation)
_SPECIAL_ANSWERS = {"yes", "no", "noanswer"}


def normalize_answer(text: str) -> str:
    """小寫、去標點、去冠詞 a/an/the、合併空白（HotpotQA 官方 normalize_answer）。"""
    text = "".join(ch for ch in text.lower() if ch not in _PUNCTUATION)
    return " ".join(_ARTICLES.sub(" ", text).split())


def _golds(gold: str, aliases: Iterable[str]) -> list[str]:
    return [gold, *aliases]


def exact_match(prediction: str, gold: str, aliases: Iterable[str] = ()) -> int:
    pred = normalize_answer(prediction)
    if not pred:
        return 0
    return int(any(pred == normalize_answer(g) for g in _golds(gold, aliases)))


def _f1_single(pred: str, gold: str) -> float:
    # 官方規則：yes／no／noanswer 必須完全相同才給分。
    if (pred in _SPECIAL_ANSWERS or gold in _SPECIAL_ANSWERS) and pred != gold:
        return 0.0
    pred_tokens, gold_tokens = pred.split(), gold.split()
    common = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def f1(prediction: str, gold: str, aliases: Iterable[str] = ()) -> float:
    pred = normalize_answer(prediction)
    if not pred:
        return 0.0
    return max(_f1_single(pred, normalize_answer(g)) for g in _golds(gold, aliases))
