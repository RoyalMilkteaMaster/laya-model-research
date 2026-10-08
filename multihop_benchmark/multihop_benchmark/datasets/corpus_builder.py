"""由抽樣題目建每集共用語料：全部段落（支持＋干擾）依 pid 去重，chunk = 一個段落。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def build_corpus(questions: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    """回傳 [{pid, title, text}]，依題目與段落的首次出現順序排列。"""
    corpus: dict[str, dict[str, str]] = {}
    for question in questions:
        for paragraph in question["paragraphs"]:
            corpus.setdefault(
                paragraph["pid"], {"pid": paragraph["pid"], "title": paragraph["title"], "text": paragraph["text"]}
            )
    return list(corpus.values())
