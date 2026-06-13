"""Answer-quality and retrieval metrics.

Answer scoring uses SQuAD-style normalization (lowercase, strip articles/punctuation,
collapse whitespace) for both Exact Match and token-level F1. Retrieval is scored as
supporting-chunk recall@k against the gold supporting chunk ids.
"""
from __future__ import annotations

import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = s.translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    common = Counter(pred_toks) & Counter(gold_toks)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pred_toks)
    recall = n_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def support_recall_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    """Fraction of gold supporting chunks present in the top-k retrieved ids."""
    if not gold_ids:
        return float("nan")
    topk = set(retrieved_ids[:k])
    hit = sum(1 for g in gold_ids if g in topk)
    return hit / len(gold_ids)
