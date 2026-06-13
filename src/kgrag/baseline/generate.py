"""Local generation over retrieved context (Ollama qwen2.5:7b-instruct, temp 0).

The model answers using only the provided passages, cites the ``[chunk_id]`` it relied
on, and must emit the abstain sentinel when the context cannot support an answer. The
reply is parsed into a structured record (answer text, citations, abstained flag).
"""
from __future__ import annotations

import re

from .. import config
from ..ollama_client import generate as ollama_generate

# chunk_ids are unique `c#####` tokens; accept them with or without brackets, since
# the model frequently drops the brackets in its Citations line.
_CITE = re.compile(r"c\d{5}")

SYSTEM = (
    "You answer multi-hop questions using ONLY the numbered context passages provided. "
    "Each passage is labelled with its [chunk_id]. Reason across passages as needed, but "
    "never use outside knowledge. Answer with the shortest span that answers the question "
    "— a single name, entity, number, or date — with no explanation. Cite the chunk_id of "
    "every passage you used. If the passages do not contain enough information to answer, "
    f"reply with exactly {config.ABSTAIN_TOKEN}."
)

PROMPT_TEMPLATE = """Context passages:
{context}

Question: {question}

Respond in exactly this format:
Answer: <the short answer, or {abstain} if the passages are insufficient>
Citations: <the [chunk_id]s you used, comma-separated; leave empty if {abstain}>"""


def build_context(chunks: list[dict]) -> str:
    lines = []
    for ch in chunks:
        lines.append(f"[{ch['chunk_id']}] ({ch['source_title']}) {ch['text']}")
    return "\n\n".join(lines)


def parse_response(text: str) -> dict:
    answer_text = ""
    citations: list[str] = []
    answer_line = None
    cite_segment = text
    for line in text.splitlines():
        low = line.strip().lower()
        if low.startswith("answer:"):
            answer_line = line.split(":", 1)[1].strip()
        elif low.startswith("citations:"):
            cite_segment = line.split(":", 1)[1]
    if answer_line is None:
        # model didn't follow format; fall back to first non-empty line
        answer_line = next((l.strip() for l in text.splitlines() if l.strip()), "")

    abstained = config.ABSTAIN_TOKEN.lower() in answer_line.lower()
    answer_text = "" if abstained else answer_line
    # citations come from the citations segment (or whole reply as fallback)
    citations = list(dict.fromkeys(_CITE.findall(cite_segment)))
    if not citations and not abstained:
        citations = list(dict.fromkeys(_CITE.findall(text)))
    return {
        "raw": text,
        "answer": answer_text,
        "citations": citations,
        "abstained": abstained,
    }


def answer_question(question: str, chunks: list[dict]) -> dict:
    prompt = PROMPT_TEMPLATE.format(
        context=build_context(chunks),
        question=question,
        abstain=config.ABSTAIN_TOKEN,
    )
    raw = ollama_generate(
        config.GEN_MODEL,
        prompt,
        system=SYSTEM,
        temperature=config.GEN_TEMPERATURE,
        num_predict=config.GEN_NUM_PREDICT,
    )
    result = parse_response(raw)
    result["context_chunk_ids"] = [c["chunk_id"] for c in chunks]
    return result
