"""Step 2 — typed query understanding + entity linking (fully deterministic, no LLM).

Replaces any free-text ReAct with a deterministic, typed query plan:

  1. **Entity spans** — GLiNER (the P2 NER model, ``urchade/gliner_medium-v2.1``) over the
     question text with the same open label schema used during extraction. Deterministic
     (eval-mode forward pass, no sampling).
  2. **Intent** — a small regex over comparison/superlative cue words. Recorded for
     diagnosis; traversal itself is relation-agnostic so intent is not required for it.
  3. **Linking** — each span is linked to a canonical graph node by:
       (a) alias-normalized **exact** match: the span's ``mention_key`` (the very key the
           P2 resolver used) looked up against every entity's ``_keys`` — confidence 1.0;
       (b) **bge-small embedding** nearest-neighbour over entity ``canonical_name``s, with a
           cosine floor (:data:`LINK_COSINE_FLOOR`) — confidence = cosine.
     A span below the floor with no exact hit is **unlinked**. A question with zero linked
     seeds contributes no graph leg and falls back to the pure hybrid retrieval (== P0 for
     that question) — the expected behaviour for under-merged bridges, recorded not hidden.

The bge embedder is the same pinned model as P0; we embed entity names and query spans
*without* the asymmetric query prefix (this is short-name ↔ short-name matching, not the
query→passage retrieval the prefix is for).
"""
from __future__ import annotations

import re

import numpy as np

from .. import config
from ..baseline import corpus_io, embed
from ..eval.triple_matcher import TripleMatcher
from .resolve_entities import mention_key

# Cosine floor for the embedding link fallback. Conservative: a wrong link injects noise into
# the (additive) graph leg, whereas a miss merely falls back to pure hybrid. Recorded, tunable.
LINK_COSINE_FLOOR = 0.80

# Only named-entity nodes seed traversal. A question's *relation* words ("director",
# "performer") are typed MISC/LITERAL and resolve to generic occupation hubs; seeding from
# them explodes the frontier with off-topic chunks. The bridge fact is reached by traversing
# the RELATES edge *from* the named entity, so we never need the relation word as a seed.
# DATE nodes are also hubs (every "born in <year>" shares one). Keep only the real anchors.
ALLOWED_SEED_TYPES = {"PERSON", "WORK", "ORG", "PLACE", "AWARD"}

# Comparison / superlative intent cues (2WikiMultiHopQA bridge_comparison + comparison).
_INTENT_CUES = re.compile(
    r"\b(younger|older|earlier|later|more recently|recently|first|last|longer|shorter|"
    r"taller|larger|smaller|both|same|which .*(?:older|younger|earlier|later))\b",
    re.IGNORECASE,
)


class QueryLinker:
    def __init__(self) -> None:
        self.matcher = TripleMatcher.load()
        self.entities = corpus_io.load_jsonl(config.RESOLUTION_ENTITIES_PATH)

        # alias-normalized key -> entity_id (authoritative: the resolver's own keys)
        self.key_to_eid: dict[str, str] = {}
        for e in self.entities:
            for k in e.get("_keys", []):
                self.key_to_eid[k] = e["entity_id"]

        # bge-small embedding index over canonical names (deterministic, ~2.3k rows)
        self._eid_order = [e["entity_id"] for e in self.entities]
        self._name_by_eid = {e["entity_id"]: e["canonical_name"] for e in self.entities}
        self._type_by_eid = {e["entity_id"]: e["type"] for e in self.entities}
        names = [self._name_by_eid[eid] for eid in self._eid_order]
        model = embed.get_model()
        self._name_mat = model.encode(
            names, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        ).astype("float32")

        self._ner = None  # lazy: GLiNER is a heavy import, only load when first linking

    # -- query understanding ------------------------------------------------
    def _load_ner(self):
        if self._ner is None:
            from ..eval.extract_triples import load_ner  # same GLiNER loader as extraction
            self._ner = load_ner()
        return self._ner

    def extract_query_entities(self, question: str) -> list[dict]:
        ner = self._load_ner()
        ents = ner.predict_entities(question, config.ENTITY_LABELS, threshold=config.GLINER_THRESHOLD)
        # dedupe by span, keep highest score, stable order by start offset
        best: dict[tuple[int, int], dict] = {}
        for e in ents:
            key = (e["start"], e["end"])
            if key not in best or e["score"] > best[key]["score"]:
                best[key] = e
        return [
            {"text": e["text"], "label": e["label"], "score": round(float(e["score"]), 4)}
            for e in sorted(best.values(), key=lambda e: e["start"])
        ]

    def intent(self, question: str) -> str:
        return "comparison" if _INTENT_CUES.search(question or "") else "lookup"

    # -- linking ------------------------------------------------------------
    def link_span(self, surface: str) -> dict | None:
        """Link one surface span to a canonical node, or None if below the floor."""
        key = mention_key(self.matcher, surface)
        eid = self.key_to_eid.get(key)
        if eid is not None:
            return {"surface": surface, "entity_id": eid,
                    "canonical_name": self._name_by_eid[eid], "type": self._type_by_eid[eid],
                    "confidence": 1.0, "method": "alias_exact"}
        # embedding fallback
        qv = embed.get_model().encode(
            surface, normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        sims = self._name_mat @ qv
        i = int(np.argmax(sims))
        cos = float(sims[i])
        if cos >= LINK_COSINE_FLOOR:
            eid = self._eid_order[i]
            return {"surface": surface, "entity_id": eid,
                    "canonical_name": self._name_by_eid[eid], "type": self._type_by_eid[eid],
                    "confidence": round(cos, 4), "method": "embedding"}
        return None

    def plan(self, question: str) -> dict:
        """Typed query plan: spans, intent, linked seeds, and unlinked spans."""
        spans = self.extract_query_entities(question)
        linked: list[dict] = []          # named-entity links used as traversal seeds
        dropped: list[dict] = []         # linked but non-seed types (relation/role/date hubs)
        unlinked: list[dict] = []        # no node above the floor
        seen_eids: set[str] = set()
        for sp in spans:
            link = self.link_span(sp["text"])
            if link is None:
                unlinked.append(sp)
                continue
            if link["entity_id"] in seen_eids:
                continue
            seen_eids.add(link["entity_id"])
            link["label"] = sp["label"]
            if link["type"] in ALLOWED_SEED_TYPES:
                linked.append(link)
            else:
                dropped.append(link)
        return {
            "intent": self.intent(question),
            "spans": spans,
            "linked": linked,
            "dropped_seeds": dropped,
            "unlinked": unlinked,
            "seed_entity_ids": [l["entity_id"] for l in linked],
        }
