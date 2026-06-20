#!/usr/bin/env python3
"""Export per-question flat-RAG vs KG-RAG comparison data for the static demo.

Reads the stored, already-scored runs (no models, no network) and emits one
record per test question plus a chunk-text lookup. Output is written as
``docs/data.js`` assigning ``window.DEMO_DATA`` so the page works over
``file://`` and offline (browsers block ``fetch()`` of a local JSON sibling).

Sources (frozen, read-only):
  - flat baseline (scored) : data/processed/runs/scored_test.jsonl
  - KG-RAG v2 (realistic)  : data/processed/runs/kgrag_scored_test.jsonl
                             (the answer-enriched v1 run is preserved as
                              *_v1_goldsupport.jsonl and is deliberately NOT used)
  - chunk text             : data/processed/corpus.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "data" / "processed" / "runs"
CORPUS = ROOT / "data" / "processed" / "corpus.jsonl"
OUT = ROOT / "docs" / "data.js"

FLAT_RUN = RUNS / "scored_test.jsonl"
KG_RUN = RUNS / "kgrag_scored_test.jsonl"

# Top-k of retrieved context actually shown to the generator (GEN_TOP_K = 5).
TOP_K = 5
# F1 deadband for declaring a per-question win/loss (below this is a "tie").
DEADBAND = 0.1
# Token-F1 threshold for the binary correct/incorrect badge vs gold.
CORRECT_F1 = 0.5


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def is_correct(rec: dict) -> bool:
    return (not rec.get("abstained", False)) and float(rec.get("f1", 0.0)) >= CORRECT_F1


def main() -> None:
    flat = {r["id"]: r for r in load_jsonl(FLAT_RUN)}
    kg = {r["id"]: r for r in load_jsonl(KG_RUN)}
    corpus = {r["chunk_id"]: r for r in load_jsonl(CORPUS)}

    assert set(flat) == set(kg), "flat and KG-RAG runs cover different question ids"

    records: list[dict] = []
    referenced: set[str] = set()
    counts = {"kg_better": 0, "kg_worse": 0, "tie": 0}

    # Preserve the on-disk question order from the flat run.
    for r in load_jsonl(FLAT_RUN):
        qid = r["id"]
        f = flat[qid]
        k = kg[qid]

        flat_ctx = list(f.get("context_chunk_ids", [])[:TOP_K])
        kg_ctx = list(k.get("context_chunk_ids", [])[:TOP_K])
        # Chunks the graph leg surfaced into the model's window that flat didn't show.
        flat_seen = set(f.get("flat_top5") or flat_ctx)
        graph_surfaced = [c for c in kg_ctx if c not in flat_seen]
        gold_support = list(f.get("gold_support_chunk_ids", []))

        df1 = float(k.get("f1", 0.0)) - float(f.get("f1", 0.0))
        if abs(df1) < DEADBAND:
            outcome = "tie"
        elif df1 > 0:
            outcome = "kg_better"
        else:
            outcome = "kg_worse"
        counts[outcome] += 1

        for c in (*flat_ctx, *kg_ctx, *gold_support):
            referenced.add(c)

        flat_correct = is_correct(f)
        kg_correct = is_correct(k)
        gold_set = set(gold_support)
        surfaced_gold = [c for c in graph_surfaced if c in gold_set]

        # Factual, templated takeaway derived only from the stored data.
        if outcome == "kg_better":
            if surfaced_gold:
                caption = (
                    f"Graph traversal surfaced gold-support chunk(s) "
                    f"{', '.join(surfaced_gold)} that flat retrieval missed; "
                    f"KG-RAG answered better (F1 {f['f1']:.2f}→{k['f1']:.2f})."
                )
            else:
                caption = (
                    f"Graph-fused retrieval improved the answer "
                    f"(F1 {f['f1']:.2f}→{k['f1']:.2f})."
                )
        elif outcome == "kg_worse":
            caption = (
                f"Graph traversal pulled extra chunks into the top-5 and the "
                f"answer got worse (F1 {f['f1']:.2f}→{k['f1']:.2f})."
            )
        else:
            if f.get("abstained") and k.get("abstained"):
                caption = "Both systems abstained — neither found a supported answer."
            elif graph_surfaced:
                caption = (
                    f"Graph surfaced {len(graph_surfaced)} new chunk(s) but answer "
                    f"quality was unchanged (F1 {k['f1']:.2f})."
                )
            else:
                caption = f"Retrieval and answer quality were unchanged (F1 {k['f1']:.2f})."

        records.append(
            {
                "id": qid,
                "hop_type": r["hop_type"],
                "hop_count": r["hop_count"],
                "question": r["question"],
                "gold_answer": r["gold_answer"],
                "gold_support_chunk_ids": gold_support,
                "flat": {
                    "answer": f.get("answer", ""),
                    "f1": round(float(f.get("f1", 0.0)), 4),
                    "em": round(float(f.get("em", 0.0)), 4),
                    "abstained": bool(f.get("abstained", False)),
                    "retrieved_chunk_ids": flat_ctx,
                    "correct": flat_correct,
                },
                "kg": {
                    "answer": k.get("answer", ""),
                    "f1": round(float(k.get("f1", 0.0)), 4),
                    "em": round(float(k.get("em", 0.0)), 4),
                    "abstained": bool(k.get("abstained", False)),
                    "retrieved_chunk_ids": kg_ctx,
                    "graph_surfaced_chunk_ids": graph_surfaced,
                    "used_graph": bool(k.get("used_graph", False)),
                    "correct": kg_correct,
                },
                "outcome": outcome,
                "caption": caption,
            }
        )

    chunks = {
        cid: {
            "source_title": corpus[cid]["source_title"],
            "text": corpus[cid]["text"],
        }
        for cid in sorted(referenced)
        if cid in corpus
    }

    payload = {
        "meta": {
            "n": len(records),
            "top_k": TOP_K,
            "deadband": DEADBAND,
            "correct_f1": CORRECT_F1,
            "outcome_counts": counts,
            "source": {
                "flat": "data/processed/runs/scored_test.jsonl",
                "kg": "data/processed/runs/kgrag_scored_test.jsonl",
                "note": "realistic v2 KG-RAG run (851-chunk graph, distractors included)",
            },
        },
        "records": records,
        "chunks": chunks,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    OUT.write_text("window.DEMO_DATA=" + blob + ";\n", encoding="utf-8")

    print(f"wrote {OUT.relative_to(ROOT)}: {len(records)} questions, "
          f"{len(chunks)} chunks referenced")
    print(f"  outcomes: {counts}")


if __name__ == "__main__":
    main()
