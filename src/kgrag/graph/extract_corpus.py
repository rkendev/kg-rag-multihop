"""Step 1 — full-corpus triple extraction, resumable and checkpointed per chunk.

Wall-clock is the dominant risk: ~3,350 chunks of qwen-7B CPU extraction is a multi-hour /
overnight batch. So this writes each chunk's predictions to ``predictions_corpus.jsonl`` the
moment they are produced, and on restart skips any ``chunk_id`` already present in that file.
Run as one background job, unbuffered:

    PYTHONUNBUFFERED=1 uv run python -u -m kgrag.graph.extract_corpus

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


def _done_chunk_ids(path) -> set[str]:
    """chunk_ids already written to the checkpoint file (resume support)."""
    done: set[str] = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["chunk_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main() -> int:
    out_path = config.EXTRACTION_PRED_CORPUS_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    corpus = corpus_io.load_corpus()
    # Optional scope restriction (v1: the test-question support chunks).
    if config.GRAPH_CORPUS_IDS_PATH.exists():
        scope = {l.strip() for l in open(config.GRAPH_CORPUS_IDS_PATH) if l.strip()}
        corpus = [c for c in corpus if c["chunk_id"] in scope]
        print(f"[corpus] scoped to {len(corpus)} chunks via {config.GRAPH_CORPUS_IDS_PATH.name}", flush=True)
    total = len(corpus)
    done = _done_chunk_ids(out_path)
    todo = [c for c in corpus if c["chunk_id"] not in done]
    print(f"[corpus] {total} chunks; {len(done)} already done; {len(todo)} to extract", flush=True)
    if not todo:
        print("[corpus] nothing to do — checkpoint already complete", flush=True)
        return 0

    ner = load_ner()
    n_triples, n_lowconf = 0, 0
    # line-buffered append so each chunk is durable the instant it's written
    with open(out_path, "a", encoding="utf-8", buffering=1) as f:
        for i, c in enumerate(todo, 1):
            cid = c["chunk_id"]
            recs, n_ents = extract_chunk_records(ner, cid, c["text"])
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_triples += 1
                if r.get("low_confidence"):
                    n_lowconf += 1
            # global progress counter (done-so-far / total), not just this session's slice
            print(
                f"[chunk] {len(done) + i}/{total} {cid}: {n_ents} entities -> {len(recs)} triples"
                + (f"  ({sum(1 for r in recs if r.get('low_confidence'))} low-conf)" if any(r.get("low_confidence") for r in recs) else ""),
                flush=True,
            )

    print(f"[corpus] done: wrote {n_triples} triples this run ({n_lowconf} low-confidence) -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
