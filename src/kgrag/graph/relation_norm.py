"""Step 1 — relation-string normalization for graph edges.

P2 stored each RELATES edge's ``relation`` exactly as the extractor emitted it. Some are
off-vocabulary surface strings (e.g. ``directed by`` instead of the canonical ``director``,
``born`` instead of ``date of birth``). Traversal in P3 is **relation-agnostic** — the
graph leg surfaces a chunk regardless of the relation label — so this normalization is for
**evidence quality only**: cleaner relation labels in the traversal log / spot-check, and a
count of how many edges carry truly off-vocab relations (a reported diagnostic, not a gate).

The canonical map is the frozen ``gold/relation_synonyms.json`` already used by the P1/P2
matcher; we reuse :class:`kgrag.eval.triple_matcher.TripleMatcher` so there is a single
source of truth. Pure-Python, deterministic.
"""
from __future__ import annotations

from ..eval.triple_matcher import TripleMatcher


class RelationNormalizer:
    def __init__(self) -> None:
        self._matcher = TripleMatcher.load()
        # canonical targets (right-hand side of the synonym table) = the in-vocab relations
        self._canonical = set(self._matcher.rel_syn.values())

    def normalize(self, relation: str) -> str:
        """Map a surface relation to its canonical form, else return the normalized surface."""
        return self._matcher.resolve_relation(relation)

    def is_off_vocab(self, relation: str) -> bool:
        """True when the relation has no canonical synonym mapping (kept verbatim)."""
        norm = self._matcher.resolve_relation(relation)
        return norm not in self._canonical
