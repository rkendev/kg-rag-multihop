"""Step 3 — bounded multi-hop traversal of the Kùzu graph as a retriever.

From the linked seed entities (Step 2), walk ``RELATES`` edges **both directions**, to
**depth ≤ 2**, under a **frontier budget** that caps how many nodes are expanded per hop and
how many edges are collected in total (prevents combinatorial explosion on hub nodes).

The graph's job is recall the flat baseline lacks: every traversed edge carries a
``source_chunk_id`` (P2 provenance), so collecting those chunk ids surfaces the *bridging*
chunk — the one that contains the second-hop fact whose entity is not in the question, which
BM25/vector miss. The collected chunks, ranked deterministically, form the graph leg that
P3 fuses with the hybrid leg.

Ranking of graph-retrieved chunks (deterministic, so truncation/ties reproduce):
  (min hop distance asc, support count desc, summed edge confidence desc, chunk_id asc).

Pure read access to ``data/graph/current/kuzu``. No LLM.
"""
from __future__ import annotations

from collections import defaultdict

from .. import config
from .relation_norm import RelationNormalizer

# Frontier budget — caps to prevent explosion; hits are logged, never silently dropped.
MAX_FRONTIER_PER_HOP = 64     # entities expanded at each hop (after dedup), by entity_id order
MAX_EDGES = 512               # total RELATES edges collected across the whole traversal
MAX_DEPTH = 2                 # both directions, depth <= 2

_NEIGHBORS = (
    "MATCH (a:Entity)-[e:RELATES]-(b:Entity) WHERE a.entity_id IN $ids "
    "RETURN a.entity_id, b.entity_id, e.relation, e.confidence, e.source_chunk_id"
)


class GraphTraverser:
    def __init__(self) -> None:
        import kuzu

        db_dir = config.GRAPH_DIR / "current" / "kuzu"
        self._db = kuzu.Database(str(db_dir))
        self._conn = kuzu.Connection(self._db)
        self._relnorm = RelationNormalizer()

    def _query_neighbors(self, ids: list[str]) -> list[tuple]:
        res = self._conn.execute(_NEIGHBORS, {"ids": ids})
        rows = []
        while res.has_next():
            rows.append(tuple(res.get_next()))
        return rows

    def traverse(self, seed_entity_ids: list[str]) -> dict:
        """Return ranked graph-retrieved chunks + diagnostics for the given seeds."""
        if not seed_entity_ids:
            return {"ranked_chunk_ids": [], "edges": [], "n_edges": 0,
                    "budget_truncated": False, "visited": 0}

        visited: set[str] = set()
        frontier = sorted(set(seed_entity_ids))
        # per chunk: min hop, support count, summed confidence
        agg: dict[str, dict] = defaultdict(lambda: {"min_hop": 99, "support": 0, "conf": 0.0})
        edges: list[dict] = []
        truncated = False

        for hop in range(1, MAX_DEPTH + 1):
            frontier = [e for e in frontier if e not in visited]
            if not frontier:
                break
            if len(frontier) > MAX_FRONTIER_PER_HOP:
                truncated = True
                frontier = frontier[:MAX_FRONTIER_PER_HOP]
            visited.update(frontier)

            rows = self._query_neighbors(frontier)
            next_frontier: set[str] = set()
            for a_id, b_id, relation, confidence, src_chunk in rows:
                if len(edges) >= MAX_EDGES:
                    truncated = True
                    break
                conf = float(confidence) if confidence is not None else 0.0
                edges.append({
                    "from": a_id, "to": b_id, "hop": hop, "chunk_id": src_chunk,
                    "relation": relation, "relation_norm": self._relnorm.normalize(relation),
                    "off_vocab": self._relnorm.is_off_vocab(relation), "confidence": conf,
                })
                c = agg[src_chunk]
                c["min_hop"] = min(c["min_hop"], hop)
                c["support"] += 1
                c["conf"] += max(conf, 0.0)
                next_frontier.add(b_id)
            if len(edges) >= MAX_EDGES:
                break
            frontier = sorted(next_frontier)

        ranked = sorted(
            agg.items(),
            key=lambda kv: (kv[1]["min_hop"], -kv[1]["support"], -kv[1]["conf"], kv[0]),
        )
        return {
            "ranked_chunk_ids": [cid for cid, _ in ranked],
            "edges": edges,
            "n_edges": len(edges),
            "off_vocab_edges": sum(1 for e in edges if e["off_vocab"]),
            "budget_truncated": truncated,
            "visited": len(visited),
        }
