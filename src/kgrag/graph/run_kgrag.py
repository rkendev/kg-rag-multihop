"""Run the KG-RAG system end to end over the held-out test slice + no-knowledge set.

Same shape as ``kgrag.baseline.run_baseline`` — only the retriever differs. For each
question: KG-fused retrieval (hybrid leg + graph leg, RRF), then the **UNCHANGED** P0
generator over the top-k context. Records mirror the baseline schema (so the same scorer
consumes them) plus a few KG diagnostics (``kg_seeds``, ``kg_linked``, ``graph_leg_size``,
``used_graph``) for the verdict's failure-mode analysis.

Generation is the single multi-hour, CPU-bound LLM stage; it runs ONCE and is persisted.
The run is **resumable**: ids already present in the output file are skipped, so a killed
batch resumes where it stopped. Launch unbuffered:

    PYTHONUNBUFFERED=1 uv run python -u -m kgrag.graph.run_kgrag
"""
from __future__ import annotations

import json
import sys
import time

from .. import config
from ..baseline import corpus_io, generate
from .kg_retrieve import KGRetriever

TEST_OUT = config.RUNS_DIR / "kgrag_test.jsonl"
NK_OUT = config.RUNS_DIR / "kgrag_no_knowledge.jsonl"


def _done_ids(path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _run(retriever: KGRetriever, questions: list[dict], out_path, *, label: str) -> None:
    max_k = max(config.RECALL_KS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_ids(out_path)
    todo = [q for q in questions if q["id"] not in done]
    print(f"[{label}] {len(questions)} questions; {len(done)} done; {len(todo)} to run", flush=True)
    if not todo:
        print(f"[{label}] nothing to do — checkpoint complete", flush=True)
        return

    t0 = time.time()
    # append (line-buffered) so each answer is durable the instant it is produced
    with open(out_path, "a", encoding="utf-8", buffering=1) as f:
        for i, q in enumerate(todo, 1):
            r = retriever.retrieve(q["question"])
            retrieved_ids = [cid for cid, _ in r["fused"][:max_k]]
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
                # --- KG diagnostics (not used by the deterministic metrics) ---
                "kg_intent": r["plan"]["intent"],
                "kg_seeds": r["plan"]["seed_entity_ids"],
                "kg_linked": r["plan"]["linked"],
                "kg_unlinked": [s["text"] for s in r["plan"]["unlinked"]],
                "graph_leg_size": len(r["graph_ranking"]),
                "graph_edges": r["traversal"]["n_edges"],
                "graph_truncated": r["traversal"]["budget_truncated"],
                "used_graph": r["used_graph"],
                "flat_top5": r["flat_ranking"][: config.GEN_TOP_K],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            n_done = len(done) + i
            rate = (time.time() - t0) / i
            print(f"[q] {n_done}/{len(questions)} {q['id'][:8]} "
                  f"used_graph={record['used_graph']} seeds={len(record['kg_seeds'])} "
                  f"glen={record['graph_leg_size']}  ({rate:.1f}s/q)", flush=True)


def main() -> int:
    retriever = KGRetriever()

    gold = corpus_io.load_jsonl(config.QUESTIONS_PATH)
    test = [g for g in gold if g.get("split") == "test"]
    _run(retriever, test, TEST_OUT, label="test")

    no_knowledge = corpus_io.load_jsonl(config.NO_KNOWLEDGE_PATH)
    _run(retriever, no_knowledge, NK_OUT, label="no_knowledge")
    print("kgrag run complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
