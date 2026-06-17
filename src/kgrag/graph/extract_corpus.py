"""Step 1 — full-corpus triple extraction, resumable and checkpointed per chunk.

Wall-clock is the dominant risk: ~851 chunks of qwen-7B CPU extraction is a multi-hour /
overnight batch. So this writes each chunk's predictions to ``predictions_corpus.jsonl`` the
moment they are produced, and on restart skips any ``chunk_id`` already accounted for. Run as
ONE background job, unbuffered (no auto-restart wrapper needed — a stalled chunk no longer
kills the run):

    PYTHONUNBUFFERED=1 uv run python -u -m kgrag.graph.extract_corpus

Resilience (per-chunk isolation): a single Ollama call can stall past its read timeout (or the
chunk can otherwise raise). That must NOT kill the whole multi-hour run. Each chunk's extraction
is wrapped: on any exception the chunk is recorded as **attempted with 0 triples** and its id is
appended to ``timed_out_chunk_ids.txt`` with the error, and the loop continues to the next chunk.
Skipping a handful of pathological chunks out of 851 is acceptable and documented.

Resume bookkeeping uses BOTH the predictions file and an ``attempted_chunk_ids.txt`` sidecar.
The sidecar records every attempted chunk regardless of triple count, so chunks that legitimately
yield 0 triples (or that timed out) are skipped on resume instead of being re-run every restart.
Resume done-set = chunk_ids in predictions ∪ chunk_ids in the attempted sidecar.

The extractor itself is the FIXED one from ``kgrag.eval.extract_triples`` (Step 0) — identical
prompt, model, provenance, and confidence fields. This LLM pass runs ONCE; scoring and graph
verification later recompute from the persisted file, never re-running the model.
"""
from __future__ import annotations

import json
import sys

from .. import config
from ..baseline import corpus_io
from ..eval.extract_triples import extract_chunk_records, load_ner

TIMED_OUT_PATH = config.EXTRACTION_DIR / "timed_out_chunk_ids.txt"


def _ids_in_jsonl(path) -> set[str]:
    out: set[str] = set()
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.add(json.loads(line)["chunk_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def _ids_in_txt(path) -> set[str]:
    if not path.exists():
        return set()
    return {l.strip() for l in open(path, encoding="utf-8") if l.strip()}


def _done_chunk_ids() -> set[str]:
    """Chunk_ids already accounted for: produced triples OR were attempted (incl. 0-triple/timeout)."""
    return _ids_in_jsonl(config.EXTRACTION_PRED_CORPUS_PATH) | _ids_in_txt(
        config.EXTRACTION_ATTEMPTED_IDS_PATH
    )


def main() -> int:
    out_path = config.EXTRACTION_PRED_CORPUS_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    corpus = corpus_io.load_corpus()
    # Optional scope restriction (the graph corpus ids file).
    if config.GRAPH_CORPUS_IDS_PATH.exists():
        scope = {l.strip() for l in open(config.GRAPH_CORPUS_IDS_PATH) if l.strip()}
        corpus = [c for c in corpus if c["chunk_id"] in scope]
        print(f"[corpus] scoped to {len(corpus)} chunks via {config.GRAPH_CORPUS_IDS_PATH.name}", flush=True)
    total = len(corpus)
    done = _done_chunk_ids()
    todo = [c for c in corpus if c["chunk_id"] not in done]
    print(f"[corpus] {total} chunks; {len(done)} already done; {len(todo)} to extract", flush=True)
    if not todo:
        print("[corpus] nothing to do — checkpoint already complete", flush=True)
        return 0

    ner = load_ner()
    n_triples, n_lowconf, n_failed = 0, 0, 0
    # line-buffered appends so each chunk's state is durable the instant it's written
    with open(out_path, "a", encoding="utf-8", buffering=1) as f, \
         open(config.EXTRACTION_ATTEMPTED_IDS_PATH, "a", encoding="utf-8", buffering=1) as af:
        for i, c in enumerate(todo, 1):
            cid = c["chunk_id"]
            n_global = len(done) + i
            try:
                recs, n_ents = extract_chunk_records(ner, cid, c["text"])
            except Exception as e:  # noqa: BLE001 — one stalled/erroring chunk must never kill the run
                # Record as attempted-with-0-triples so resume skips it; log which chunk failed.
                af.write(cid + "\n")
                with open(TIMED_OUT_PATH, "a", encoding="utf-8") as tf:
                    tf.write(f"{cid}\t{type(e).__name__}\t{str(e)[:200]}\n")
                n_failed += 1
                print(
                    f"[chunk] {n_global}/{total} {cid}: FAILED ({type(e).__name__}) "
                    f"-> 0 triples, marked attempted+skipped",
                    flush=True,
                )
                continue
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_triples += 1
                if r.get("low_confidence"):
                    n_lowconf += 1
            af.write(cid + "\n")  # mark attempted (covers the legitimate 0-triple case too)
            print(
                f"[chunk] {n_global}/{total} {cid}: {n_ents} entities -> {len(recs)} triples"
                + (f"  ({sum(1 for r in recs if r.get('low_confidence'))} low-conf)" if any(r.get("low_confidence") for r in recs) else ""),
                flush=True,
            )

    print(
        f"[corpus] done: wrote {n_triples} triples this run ({n_lowconf} low-confidence); "
        f"{n_failed} chunk(s) failed/timed-out (see {TIMED_OUT_PATH.name}) -> {out_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
