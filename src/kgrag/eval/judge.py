"""Local-judge faithfulness + citation-correctness rubric.

PROVISIONAL. The judge is a *separate* model (llama3.1:8b-instruct, temp 0) from the
qwen generator, so the system never grades its own output. Judge calibration against
human labels is deferred to P4; these numbers are directional only.

Two binary judgements per answered question:
* faithful          — the answer is fully supported by the supplied context passages
                      (no facts beyond them).
* citations_correct — the cited [chunk_id]s exist in the context and genuinely support
                      the answer, and at least one citation is given.
"""
from __future__ import annotations

import re

from .. import config
from ..ollama_client import generate as ollama_generate

JUDGE_SYSTEM = (
    "You are a strict evaluator. Judge only against the passages provided; do not use "
    "outside knowledge. Answer with the exact labelled format requested, nothing else."
)

JUDGE_TEMPLATE = """You are grading an answer produced from retrieved passages.

Passages shown to the system:
{context}

Question: {question}
System answer: {answer}
Cited chunk_ids: {citations}

Judge two things:
1. FAITHFUL: is every claim in the system answer supported by the passages above?
2. CITATIONS: do the cited chunk_ids exist above AND contain support for the answer,
   with at least one citation present?

Respond in exactly this format:
FAITHFUL: yes|no
CITATIONS: correct|incorrect"""


def _context_block(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{c['chunk_id']}] ({c['source_title']}) {c['text']}" for c in chunks
    )


def judge_answer(question: str, answer: str, citations: list[str], chunks: list[dict]) -> dict:
    prompt = JUDGE_TEMPLATE.format(
        context=_context_block(chunks),
        question=question,
        answer=answer,
        citations=", ".join(citations) if citations else "(none)",
    )
    raw = ollama_generate(
        config.JUDGE_MODEL,
        prompt,
        system=JUDGE_SYSTEM,
        temperature=config.JUDGE_TEMPERATURE,
        num_predict=32,
    )
    faithful = _yes(raw, "faithful")
    citations_correct = _correct(raw, "citations")
    return {"faithful": faithful, "citations_correct": citations_correct, "judge_raw": raw}


def _yes(text: str, key: str) -> bool:
    m = re.search(rf"{key}\s*:\s*(yes|no)", text, re.IGNORECASE)
    return bool(m) and m.group(1).lower() == "yes"


def _correct(text: str, key: str) -> bool:
    m = re.search(rf"{key}\s*:\s*(correct|incorrect)", text, re.IGNORECASE)
    return bool(m) and m.group(1).lower() == "correct"
