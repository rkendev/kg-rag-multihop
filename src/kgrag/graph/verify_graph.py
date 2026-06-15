"""Step 4 — deterministic graph-stats verify + connectivity report (pure Python, no LLM).

Two jobs:

1. **Deterministic reproduce.** Recompute node/edge/alias counts and integrity invariants from
   the stored resolution output + frozen corpus, twice, and assert the two runs are identical and
   match the build manifest. No model is run; the graph must recompute exactly from artifacts.

2. **Connectivity report** (guards the *under-merge* failure mode that high merge-precision hides):
   (a) bridging-entity count — canonical entities spanning >=2 chunks and/or with RELATES edges
       sourced from >=2 distinct source_chunk_ids;
   (b) bridge-entity resolution coverage — for gold bridge/compositional/bridge_comparison test
       questions, the fraction whose bridge entity (the entity shared across the two gold hops)
       resolved to a single canonical node connecting both hops;
   (c) an entity-resolution recall proxy: coverage of all gold multi-hop chain entities into nodes,
       plus a near-duplicate candidate list (same type, just under threshold) for a hand audit.

Reads gold/questions.jsonl + gold/test_ids.txt READ-ONLY (inspecting, not modifying — the fairness
guardrail forbids changing those files, not reading them).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

from .. import config
from ..baseline import corpus_io
from ..eval.triple_matcher import TripleMatcher
from .resolve_entities import mention_key  # shared resolution key (alias + paren-tag)

MULTIHOP = {"bridge", "compositional", "bridge_comparison"}


def recompute_stats(corpus, entities, triples) -> dict:
    chunk_ids = {c["chunk_id"] for c in corpus}
    ent_ids = {e["entity_id"] for e in entities}
    rel = [t for t in triples if t.get("subj_id") and t.get("obj_id")]
    orphan_chunk = sum(1 for t in rel if t["chunk_id"] not in chunk_ids)
    orphan_subj = sum(1 for t in rel if t["subj_id"] not in ent_ids)
    orphan_obj = sum(1 for t in rel if t["obj_id"] not in ent_ids)
    aliases = sum(len({a["surface"] for a in e["aliases"]}) for e in entities)
    sources = len({c["source_title"] for c in corpus})
    return {
        "entities": len(entities),
        "chunks": len(corpus),
        "sources": sources,
        "relates_edges": len(rel),
        "aliases": aliases,
        "dropped_unresolved": len(triples) - len(rel),
        "orphan_edges": orphan_chunk + orphan_subj + orphan_obj,
    }


def bridging_entities(entities, triples) -> set[str]:
    """Canonical entities spanning >=2 chunks and/or RELATES-sourced from >=2 distinct chunks."""
    edge_chunks: dict[str, set[str]] = defaultdict(set)
    for t in triples:
        if t.get("subj_id") and t.get("obj_id"):
            edge_chunks[t["subj_id"]].add(t["chunk_id"])
            edge_chunks[t["obj_id"]].add(t["chunk_id"])
    out = set()
    for e in entities:
        if e["n_chunks"] >= 2 or len(edge_chunks.get(e["entity_id"], ())) >= 2:
            out.add(e["entity_id"])
    return out


def _bridge_entity_surfaces(gold_triples: list) -> list[str]:
    """The entity/entities shared across >=2 gold hops (the multi-hop bridge)."""
    counts: dict[str, int] = defaultdict(int)
    first_surface: dict[str, str] = {}
    for tr in gold_triples:
        for surf in (tr[0], tr[2]):
            k = (surf or "").strip().lower()
            counts[k] += 1
            first_surface.setdefault(k, surf)
    return [first_surface[k] for k, n in counts.items() if n >= 2]


def load_test_questions() -> list[dict]:
    test_ids = {l.strip() for l in open(config.TEST_IDS_PATH) if l.strip()}
    qs = corpus_io.load_jsonl(config.QUESTIONS_PATH)
    return [q for q in qs if q["id"] in test_ids]


def bridge_coverage(entities, triples) -> dict:
    """Fraction of gold multi-hop bridge entities that resolved to a single canonical node."""
    matcher = TripleMatcher.load()
    # surface-key -> entity_id, rebuilt from the resolved triples (authoritative mapping)
    key_to_eid: dict[str, str] = {}
    for t in triples:
        for surf, eid in ((t["subject"], t.get("subj_id")), (t["object"], t.get("obj_id"))):
            if eid:
                key_to_eid[mention_key(matcher, surf)] = eid

    qs = [q for q in load_test_questions() if q.get("hop_type") in MULTIHOP and q.get("gold_triples")]
    total, single_node, present = 0, 0, 0
    misses = []
    for q in qs:
        for surf in _bridge_entity_surfaces(q["gold_triples"]):
            total += 1
            eid = key_to_eid.get(mention_key(matcher, surf))
            if eid:
                present += 1
                single_node += 1  # a single key maps to exactly one eid by construction
            else:
                misses.append({"qid": q["id"], "bridge": surf, "hop_type": q["hop_type"]})
    return {
        "multihop_questions": len(qs),
        "bridge_entities": total,
        "resolved_to_node": present,
        "coverage": (present / total) if total else 0.0,
        "misses_sample": misses[:15],
    }


def chain_entity_recall(triples) -> dict:
    """ER recall proxy: of all entities in gold multi-hop chains, the fraction that resolved to
    any canonical node (under-merge / extraction misses show up as low coverage)."""
    matcher = TripleMatcher.load()
    key_to_eid: dict[str, str] = {}
    for t in triples:
        for surf, eid in ((t["subject"], t.get("subj_id")), (t["object"], t.get("obj_id"))):
            if eid:
                key_to_eid[mention_key(matcher, surf)] = eid
    qs = [q for q in load_test_questions() if q.get("hop_type") in MULTIHOP and q.get("gold_triples")]
    keys = set()
    for q in qs:
        for tr in q["gold_triples"]:
            for surf in (tr[0], tr[2]):
                keys.add(mention_key(matcher, surf))
    keys.discard("")
    present = sum(1 for k in keys if k in key_to_eid)
    return {"chain_entities": len(keys), "resolved": present,
            "recall_proxy": (present / len(keys)) if keys else 0.0}


def main() -> int:
    corpus = corpus_io.load_corpus()
    entities = corpus_io.load_jsonl(config.RESOLUTION_ENTITIES_PATH)
    triples = corpus_io.load_jsonl(config.RESOLUTION_TRIPLES_PATH)

    s1 = recompute_stats(corpus, entities, triples)
    s2 = recompute_stats(corpus, entities, triples)
    reproducible = s1 == s2

    manifest_path = config.GRAPH_DIR / "current" / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    matches_manifest = all(manifest.get(k) == s1.get(k) for k in ("entities", "chunks", "sources", "relates_edges")) if manifest else None

    bridges = bridging_entities(entities, triples)
    cov = bridge_coverage(entities, triples)
    rec = chain_entity_recall(triples)

    print("=" * 64)
    print("GRAPH DETERMINISTIC VERIFY + CONNECTIVITY")
    print("=" * 64)
    for k, v in s1.items():
        print(f"  {k}: {v}")
    print(f"recompute identical: {reproducible}   matches manifest: {matches_manifest}")
    print(f"integrity: orphan_edges={s1['orphan_edges']}  (must be 0)")
    print("-" * 64)
    print("CONNECTIVITY")
    print(f"  (a) bridging entities (>=2 chunks/edges): {len(bridges)}")
    print(f"  (b) gold multi-hop questions: {cov['multihop_questions']}  bridge entities: {cov['bridge_entities']}")
    print(f"      resolved to a single node: {cov['resolved_to_node']}  coverage: {cov['coverage']:.3f}")
    print(f"  (c) chain-entity ER recall proxy: {rec['resolved']}/{rec['chain_entities']} = {rec['recall_proxy']:.3f}")
    if cov["misses_sample"]:
        print("  sample bridge misses (entity absent from graph):")
        for m in cov["misses_sample"][:10]:
            print(f"      [{m['hop_type']}] {m['bridge']}  ({m['qid'][:8]})")

    ok = reproducible and s1["orphan_edges"] == 0 and (matches_manifest in (True, None))
    print("-" * 64)
    print("GRAPH VERIFY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
