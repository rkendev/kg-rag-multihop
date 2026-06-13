"""Run the flat-RAG baseline end to end over the held-out test slice + no-knowledge set.

For each question: hybrid-retrieve a fused ranking, then generate an answer (with
citations) from the top-k context. Raw per-question records are written to
``data/processed/runs/`` for the scorer to consume — generation and scoring are kept
separate so the (slow, CPU-bound) generation pass is done once.
"""
from __future__ import annotations

import json
import sys
import time

from .. import config
from . import corpus_io, generate
from .retrieve import HybridRetriever


def _run(retriever: HybridRetriever, questions: list[dict], out_path, *, label: str) -> None:
    max_k = max(config.RECALL_KS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as f:
        for i, q in enumerate(questions, 1):
            fused = retriever.search(q["question"])
            retrieved_ids = [cid for cid, _ in fused[:max_k]]
            context = [retriever.by_id[cid] for cid in retrieved_ids[: config.GEN_TOP_K]]
            gen = generate.answer_question(q["question"], context)
            record = {
                "id": q["id"],
                "question": q["question"],
                "gold_answer": q.get("answer"),
                "hop_type": q["hop_type"],
                "hop_count": q["hop_count"],
                "gold_support_chunk_ids": q.get("gold_support_chunk_ids", []),
                "retrieved_ids": retrieved_ids,
                "answer": gen["answer"],
                "citations": gen["citations"],
                "abstained": gen["abstained"],
                "context_chunk_ids": gen["context_chunk_ids"],
                "raw": gen["raw"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(questions):
                rate = (time.time() - t0) / i
                print(f"[{label}] {i}/{len(questions)}  ({rate:.1f}s/q)", flush=True)


def main() -> int:
    retriever = HybridRetriever()

    gold = corpus_io.load_jsonl(config.QUESTIONS_PATH)
    test = [g for g in gold if g.get("split") == "test"]
    print(f"test questions: {len(test)}")
    _run(retriever, test, config.RUNS_DIR / "baseline_test.jsonl", label="test")

    no_knowledge = corpus_io.load_jsonl(config.NO_KNOWLEDGE_PATH)
    print(f"no-knowledge questions: {len(no_knowledge)}")
    _run(
        retriever,
        no_knowledge,
        config.RUNS_DIR / "baseline_no_knowledge.jsonl",
        label="no_knowledge",
    )
    print("baseline run complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
