"""Shared loaders + tokenizer for the frozen corpus and gold sets."""
from __future__ import annotations

import json
import re

from .. import config

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization shared by BM25 indexing and querying."""
    return _TOKEN.findall(text.lower())


def load_corpus() -> list[dict]:
    """Frozen corpus in stable chunk order (FAISS row i == corpus[i])."""
    with open(config.CORPUS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]
