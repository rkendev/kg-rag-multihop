"""Regenerate the graph-corpus scope file (``graph_corpus_ids.txt``).

Two scopes are reproducible from the frozen artifacts:

* ``gold``  — the unique gold-support chunks of the 100 held-out test questions
  (the v1 overlay, 246 chunks). This is *answer-enriched*: the graph contains only
  the paragraphs that support the gold answers, which is favourable to the graph leg.
* ``full``  — the union of the 100 test questions' **full context pools** (gold
  support **plus** their distractor paragraphs, exactly as they appear in the frozen
  corpus). 851 chunks. This is the *defensibility* scope: the graph now includes the
  same distractors the flat retriever must contend with, so the KG-vs-flat comparison
  is no longer biased by an answer-only overlay.

Both map a test question's context paragraphs to ``chunk_id`` via the SAME
``(source_title, normalized_text)`` key the corpus build used (``para_text``), so the
mapping is byte-identical to ``build_corpus``. The FAISS/vector index stays over the
full 3,350-chunk corpus regardless of scope — only the graph overlay is scoped.

Deterministic, pure-Python, no LLM. ``--write`` persists; default is a dry-run report.

    uv run python -m kgrag.graph.make_graph_scope --scope full --write
"""
from __future__ import annotations

import argparse
import json
import sys

from .. import config
from ..ingest.build_corpus import para_text


def _corpus_key_index() -> dict[tuple[str, str], str]:
    """(source_title, normalized_text) -> chunk_id over the frozen corpus."""
    idx: dict[tuple[str, str], str] = {}
    with open(config.CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            idx[(c["source_title"], c["text"])] = c["chunk_id"]
    return idx


def _test_ids() -> list[str]:
    return [l.strip() for l in open(config.TEST_IDS_PATH, encoding="utf-8") if l.strip()]


def compute_scope(scope: str) -> tuple[set[str], dict]:
    """Return (chunk_ids, stats) for ``scope`` in {'gold', 'full'}."""
    key_to_chunk = _corpus_key_index()
    with open(config.DEV_JSON, encoding="utf-8") as f:
        by_id = {ex["_id"]: ex for ex in json.load(f)}

    test_ids = _test_ids()
    union: set[str] = set()
    support: set[str] = set()
    n_paras = 0
    unmapped = 0
    for qid in test_ids:
        ex = by_id[qid]
        support_titles = {t for t, _ in ex["supporting_facts"]}
        for title, sents in ex["context"]:
            n_paras += 1
            cid = key_to_chunk.get((title, para_text(sents)))
            if cid is None:
                unmapped += 1
                continue
            if title in support_titles:
                support.add(cid)
            if scope == "full" or title in support_titles:
                union.add(cid)

    stats = {
        "scope": scope,
        "test_questions": len(test_ids),
        "context_paragraphs_pooled": n_paras,
        "unmapped_paragraphs": unmapped,
        "scope_chunks": len(union),
        "gold_support_chunks": len(support),
    }
    return union, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["gold", "full"], default="full")
    ap.add_argument("--write", action="store_true", help="persist to graph_corpus_ids.txt")
    args = ap.parse_args()

    chunk_ids, stats = compute_scope(args.scope)
    print(json.dumps(stats, indent=2))
    if stats["unmapped_paragraphs"]:
        raise SystemExit(
            f"{stats['unmapped_paragraphs']} test context paragraphs did not map to the "
            f"frozen corpus — scope would be incomplete; aborting."
        )

    # Provenance check: the answer-enriched gold-support subset must be contained in the
    # full scope, so an existing checkpoint extracted over the gold subset stays valid.
    if args.scope == "full":
        gold_ids, _ = compute_scope("gold")
        if not gold_ids <= chunk_ids:
            raise SystemExit("gold-support subset is NOT contained in the full scope; aborting.")
        print(f"[scope] gold-support subset ({len(gold_ids)}) ⊆ full scope ({len(chunk_ids)}): OK")

    if args.write:
        out = config.GRAPH_CORPUS_IDS_PATH
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(chunk_ids)) + "\n")
        print(f"[scope] wrote {len(chunk_ids)} chunk_ids -> {out}")
    else:
        print("[scope] dry-run (pass --write to persist)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
