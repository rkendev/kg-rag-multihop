"""Step 3 — build the embedded Kùzu graph + FAISS vector index from stored artifacts.

Pure assembly, no LLM: reads the resolved triples + canonical entities (Step 2) and the frozen
corpus, and writes a versioned graph under ``data/graph/v<N>/`` with a ``current`` symlink:

  nodes:  Entity{entity_id, canonical_name, type, aliases[]}
          Chunk{chunk_id, doc_id, text, source_title}
          Source{doc_id}
  edges:  (Entity)-[:RELATES {relation, confidence, source_chunk_id}]->(Entity)
          (Entity)-[:MENTIONED_IN]->(Chunk)
          (Chunk)-[:FROM_SOURCE]->(Source)     # 'FROM' is reserved in Cypher; same semantics

Every RELATES edge carries provenance (source_chunk_id, confidence) — non-negotiable. The FAISS
flat index reuses the P0 corpus embeddings (same pinned bge model — the embedder is not re-run)
and is keyed by chunk_id via a sidecar chunk_ids.json. Integrity is asserted before the build is
blessed: no orphan edges, every RELATES endpoint a real entity, every source_chunk_id a real chunk.
"""
from __future__ import annotations

import json
import shutil
import sys

import numpy as np

from .. import config
from ..baseline import corpus_io


def _doc_id(source_title: str) -> str:
    """Each Wikipedia article title is one Source document."""
    return source_title


def load_inputs():
    corpus = corpus_io.load_corpus()
    entities = corpus_io.load_jsonl(config.RESOLUTION_ENTITIES_PATH)
    triples = corpus_io.load_jsonl(config.RESOLUTION_TRIPLES_PATH)
    return corpus, entities, triples


def integrity_check(corpus, entities, triples) -> dict:
    """Fail loudly before building if anything would create an orphan/dangling edge."""
    chunk_ids = {c["chunk_id"] for c in corpus}
    ent_ids = {e["entity_id"] for e in entities}
    rel = [t for t in triples if t.get("subj_id") and t.get("obj_id")]
    dropped = len(triples) - len(rel)
    bad_chunk = [t for t in rel if t["chunk_id"] not in chunk_ids]
    bad_subj = [t for t in rel if t["subj_id"] not in ent_ids]
    bad_obj = [t for t in rel if t["obj_id"] not in ent_ids]
    if bad_chunk or bad_subj or bad_obj:
        raise SystemExit(
            f"integrity FAIL: {len(bad_chunk)} edges -> missing chunk, "
            f"{len(bad_subj)} -> missing subj entity, {len(bad_obj)} -> missing obj entity"
        )
    return {"relates_edges": len(rel), "dropped_unresolved": dropped}


def build_kuzu(db_dir, corpus, entities, triples):
    import kuzu

    # kuzu may persist as a single file (+ .wal) or a directory depending on version; clear both.
    for p in (db_dir, db_dir.with_name(db_dir.name + ".wal")):
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    db = kuzu.Database(str(db_dir))
    conn = kuzu.Connection(db)

    conn.execute(
        "CREATE NODE TABLE Entity(entity_id STRING, canonical_name STRING, type STRING, "
        "aliases STRING[], PRIMARY KEY(entity_id))"
    )
    conn.execute(
        "CREATE NODE TABLE Chunk(chunk_id STRING, doc_id STRING, text STRING, "
        "source_title STRING, PRIMARY KEY(chunk_id))"
    )
    conn.execute("CREATE NODE TABLE Source(doc_id STRING, PRIMARY KEY(doc_id))")
    conn.execute(
        "CREATE REL TABLE RELATES(FROM Entity TO Entity, relation STRING, "
        "confidence DOUBLE, source_chunk_id STRING)"
    )
    conn.execute("CREATE REL TABLE MENTIONED_IN(FROM Entity TO Chunk)")
    conn.execute("CREATE REL TABLE FROM_SOURCE(FROM Chunk TO Source)")

    # --- nodes ---
    for e in entities:
        conn.execute(
            "CREATE (:Entity {entity_id:$id, canonical_name:$n, type:$t, aliases:$a})",
            {"id": e["entity_id"], "n": e["canonical_name"], "t": e["type"],
             "a": sorted({a["surface"] for a in e["aliases"]})},
        )
    sources = sorted({_doc_id(c["source_title"]) for c in corpus})
    for did in sources:
        conn.execute("CREATE (:Source {doc_id:$d})", {"d": did})
    for c in corpus:
        conn.execute(
            "CREATE (:Chunk {chunk_id:$c, doc_id:$d, text:$x, source_title:$s})",
            {"c": c["chunk_id"], "d": _doc_id(c["source_title"]), "x": c["text"],
             "s": c["source_title"]},
        )

    # --- edges ---
    # Chunk -> Source
    for c in corpus:
        conn.execute(
            "MATCH (ch:Chunk {chunk_id:$c}), (s:Source {doc_id:$d}) "
            "CREATE (ch)-[:FROM_SOURCE]->(s)",
            {"c": c["chunk_id"], "d": _doc_id(c["source_title"])},
        )
    # Entity -> Entity (RELATES, with provenance) and Entity -> Chunk (MENTIONED_IN)
    mentioned = set()
    n_rel = 0
    for t in triples:
        if not (t.get("subj_id") and t.get("obj_id")):
            continue
        conf = t.get("llm_confidence")
        conn.execute(
            "MATCH (a:Entity {entity_id:$s}), (b:Entity {entity_id:$o}) "
            "CREATE (a)-[:RELATES {relation:$r, confidence:$c, source_chunk_id:$ch}]->(b)",
            {"s": t["subj_id"], "o": t["obj_id"], "r": t["relation"],
             "c": float(conf) if conf is not None else -1.0, "ch": t["chunk_id"]},
        )
        n_rel += 1
        for eid in (t["subj_id"], t["obj_id"]):
            link = (eid, t["chunk_id"])
            if link not in mentioned:
                conn.execute(
                    "MATCH (e:Entity {entity_id:$e}), (ch:Chunk {chunk_id:$c}) "
                    "CREATE (e)-[:MENTIONED_IN]->(ch)",
                    {"e": eid, "c": t["chunk_id"]},
                )
                mentioned.add(link)
    return {"entities": len(entities), "chunks": len(corpus), "sources": len(sources),
            "relates": n_rel, "mentioned_in": len(mentioned)}


def build_faiss(db_dir, corpus):
    """Reuse the P0 corpus embeddings (same pinned model — embedder not re-run); persist a flat
    IP index + chunk_id sidecar next to the Kùzu DB."""
    import faiss

    vecs = np.load(config.EMB_PATH)
    if vecs.shape[0] != len(corpus):
        raise SystemExit(f"embedding rows {vecs.shape[0]} != corpus chunks {len(corpus)}")
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, str(db_dir / "faiss.index"))
    with open(db_dir / "chunk_ids.json", "w", encoding="utf-8") as f:
        json.dump([c["chunk_id"] for c in corpus], f)
    return {"faiss_vectors": index.ntotal, "faiss_dim": int(vecs.shape[1])}


def main() -> int:
    corpus, entities, triples = load_inputs()
    integ = integrity_check(corpus, entities, triples)

    version_dir = config.GRAPH_DIR / "v1"
    version_dir.mkdir(parents=True, exist_ok=True)
    kuzu_dir = version_dir / "kuzu"

    counts = build_kuzu(kuzu_dir, corpus, entities, triples)
    faiss_counts = build_faiss(version_dir, corpus)

    manifest = {
        "embed_model": config.EMBED_MODEL,
        "gliner_model": config.GLINER_MODEL,
        "extract_model": config.EXTRACT_MODEL,
        "resolve_fuzz_ratio": config.RESOLVE_FUZZ_RATIO,
        "resolve_cosine": config.RESOLVE_COSINE,
        **counts, **faiss_counts, **integ,
    }
    with open(version_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # rollback symlink: data/graph/current -> v1
    current = config.GRAPH_DIR / "current"
    if current.is_symlink() or current.exists():
        current.unlink()
    current.symlink_to(version_dir.name)

    print("[build] graph + index written to", version_dir)
    for k, v in manifest.items():
        print(f"  {k}: {v}")
    print("[build] current ->", version_dir.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
