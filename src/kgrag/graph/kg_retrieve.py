"""Step 4 — hybrid fusion: the ONLY thing that differs from P0.

``KGRetriever`` produces context chunks for the **unchanged** P0 generator. It fuses two
ranked legs with Reciprocal Rank Fusion (same constant ``RRF_K`` = 60 as P0):

  * **Leg A — flat hybrid (P0, untouched):** ``HybridRetriever.search`` over the full
    3,350-chunk corpus (BM25 + bge vector, internally RRF-fused). This is the P0 control.
  * **Leg B — graph traversal:** link the question's entities to graph nodes (Step 2),
    traverse RELATES both directions depth ≤ 2 (Step 3), collect the source chunks.

The two legs are RRF-fused into one ranking. The context handed to the generator is still a
set of corpus chunk passages (same ``GEN_TOP_K`` = 5), so the generation prompt stays
byte-identical to P0 — only the chunk *set* can change.

Fallback: a question whose entities don't link (leg B empty) is fused trivially and the
result equals leg A — i.e. identical to P0 for that question. This is expected for
under-merged bridges and is recorded per-question (``used_graph`` / ``graph_leg_size``).

``KGRetriever`` mirrors ``HybridRetriever``'s interface (``search``, ``by_id``,
``topk_chunks``) so the existing baseline driver shape carries over unchanged.
"""
from __future__ import annotations

from .. import config
from ..baseline.retrieve import HybridRetriever
from .query_link import QueryLinker
from .traverse import GraphTraverser


def _rrf_fuse(rankings: list[list[str]], k: int) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion over 1-based ranks; identical formula to baseline.retrieve."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


class KGRetriever:
    def __init__(self) -> None:
        self.hybrid = HybridRetriever()
        self.by_id = self.hybrid.by_id          # reuse P0's corpus index
        self.linker = QueryLinker()
        self.traverser = GraphTraverser()

    def retrieve(self, query: str) -> dict:
        """Full KG-RAG retrieval with diagnostics. Returns fused ranking + the query plan."""
        flat = self.hybrid.search(query)                 # leg A: [(chunk_id, score)]
        flat_ranking = [cid for cid, _ in flat]

        plan = self.linker.plan(query)                   # Step 2
        trav = self.traverser.traverse(plan["seed_entity_ids"])  # Step 3
        graph_ranking = trav["ranked_chunk_ids"]         # leg B

        if graph_ranking:
            fused = _rrf_fuse([flat_ranking, graph_ranking], config.RRF_K)
        else:
            fused = flat                                 # pure fallback == P0 for this question

        return {
            "fused": fused,
            "plan": plan,
            "traversal": trav,
            "flat_ranking": flat_ranking,
            "graph_ranking": graph_ranking,
            "used_graph": bool(graph_ranking),
        }

    # -- HybridRetriever-compatible surface --------------------------------
    def search(self, query: str, *, pool: int | None = None) -> list[tuple[str, float]]:
        return self.retrieve(query)["fused"]

    def topk_chunks(self, query: str, k: int) -> list[dict]:
        ranked = self.search(query)[:k]
        return [self.by_id[cid] for cid, _ in ranked]
