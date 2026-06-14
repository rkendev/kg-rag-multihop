"""Entity-resolution-aware triple matcher (the gate's measuring instrument).

A predicted triple counts as a true positive only if its **subject**, its **canonical
relation**, and its **object** all align with a gold triple after three resolution steps:

1. surface-form normalization (lowercase, strip diacritics/punctuation/parentheticals),
2. a canonical-entity alias table (``gold/alias_table.json``), and
3. a canonical-relation synonym table (``gold/relation_synonyms.json``).

The matcher's own correctness is a precondition for the gate meaning anything — a lenient
matcher passes a bad extractor, a strict one fails a good one — so it is validated against
hand-judged pairs (see ``matcher_validation``) and frozen before any scoring run.

Pure Python, deterministic: scoring stored predictions recomputes instantly with no LLM.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass

from .. import config

# ---------------------------------------------------------------------------
# Frozen resolution tables (committed under gold/ before the scoring run)
# ---------------------------------------------------------------------------
_ALIAS_PATH = config.GOLD / "alias_table.json"
_RELSYN_PATH = config.GOLD / "relation_synonyms.json"


def _load_table(path) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
_PAREN = re.compile(r"\([^)]*\)")
_NONWORD = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")
_POSSESSIVE = re.compile(r"\b(\w+)'s\b")


def normalize(s: str) -> str:
    """Lowercase, strip diacritics, parentheticals, punctuation; collapse whitespace.

    "I.B.M." -> "ibm"; "Sallie( Priddy) Fox" -> "sallie fox"; "Lamač" -> "lamac".
    """
    s = s.strip().lower()
    s = _POSSESSIVE.sub(r"\1", s)            # drop possessive 's before depunctuating
    s = _PAREN.sub(" ", s)                   # remove parenthetical asides
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = _NONWORD.sub("", s)                  # join "i.b.m." -> "ibm", drop commas etc.
    return _WS.sub(" ", s).strip()


# ---------------------------------------------------------------------------
# Date handling — match on (year, month, day) where both sides specify them
# ---------------------------------------------------------------------------
_MONTHS = {
    m: i
    for i, ms in enumerate(
        [
            ("january", "jan"), ("february", "feb"), ("march", "mar"),
            ("april", "apr"), ("may",), ("june", "jun"), ("july", "jul"),
            ("august", "aug"), ("september", "sep", "sept"), ("october", "oct"),
            ("november", "nov"), ("december", "dec"),
        ],
        start=1,
    )
    for m in ms
}
_YEAR = re.compile(r"\b(1\d{3}|20\d{2})\b")
_DAY = re.compile(r"\b([0-3]?\d)\b")


def parse_date(s: str):
    """Return (year, month|None, day|None) if ``s`` looks like a date, else None."""
    low = s.lower()
    ym = _YEAR.search(low)
    if not ym:
        return None
    year = int(ym.group(1))
    month = next((num for name, num in _MONTHS.items() if re.search(rf"\b{name}\b", low)), None)
    day = None
    for m in _DAY.finditer(low):
        d = int(m.group(1))
        if 1 <= d <= 31 and m.group(1) != ym.group(1):
            day = d
            break
    return (year, month, day)


def _date_compatible(a, b) -> bool:
    """Years must agree; month/day must agree only where BOTH sides specify them."""
    if a[0] != b[0]:
        return False
    if a[1] is not None and b[1] is not None and a[1] != b[1]:
        return False
    if a[2] is not None and b[2] is not None and a[2] != b[2]:
        return False
    return True


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------
@dataclass
class TripleMatcher:
    alias: dict[str, str]
    rel_syn: dict[str, str]

    @classmethod
    def load(cls) -> "TripleMatcher":
        alias = {normalize(k): normalize(v) for k, v in _load_table(_ALIAS_PATH).items()}
        rel_syn = {normalize(k): v.strip().lower() for k, v in _load_table(_RELSYN_PATH).items()}
        return cls(alias=alias, rel_syn=rel_syn)

    # -- resolution -------------------------------------------------------
    def resolve_entity(self, s: str) -> str:
        n = normalize(s)
        return self.alias.get(n, n)

    def resolve_relation(self, r: str) -> str:
        n = normalize(r)
        # exact normalized-surface hit, else fall back to the normalized surface itself
        return self.rel_syn.get(n, n)

    # -- field alignment --------------------------------------------------
    def entity_match(self, a: str, b: str) -> bool:
        """Named-entity / place / date alignment (strict enough not to over-merge)."""
        ra, rb = self.resolve_entity(a), self.resolve_entity(b)
        if not ra or not rb:
            return False
        if ra == rb:
            return True
        da, db = parse_date(a), parse_date(b)
        if da is not None and db is not None:
            return _date_compatible(da, db)
        # Place containment: compare the most-specific (first, pre-normalization)
        # comma component — "Hamburg" ~ "Hamburg, Germany", "Gainesville, Texas".
        fa = self.resolve_entity(a.split(",")[0])
        fb = self.resolve_entity(b.split(",")[0])
        if fa and fb and fa == fb:
            return True
        # Multi-token name subset — "Jeremy Camp" ~ "Jeremy Thomas Camp". Require >=2
        # shared tokens so a single shared surname ("Fox" ~ "Frank Fox") does NOT merge.
        ta, tb = set(ra.split()), set(rb.split())
        if ta and tb and (ta <= tb or tb <= ta) and min(len(ta), len(tb)) >= 2:
            return True
        return False

    def literal_match(self, a: str, b: str) -> bool:
        """Free-text literal alignment for descriptive values (occupation, genre,
        cause of death): token-containment either way — "director" ~ "film director"."""
        ta, tb = set(normalize(a).split()), set(normalize(b).split())
        if not ta or not tb:
            return False
        return ta <= tb or tb <= ta

    def relation_match(self, a: str, b: str) -> bool:
        return self.resolve_relation(a) == self.resolve_relation(b)

    # Relations whose object is a descriptive literal rather than a named entity.
    LITERAL_RELATIONS = {"occupation", "genre", "cause of death"}

    def triple_match(self, pred: tuple[str, str, str], gold: tuple[str, str, str]) -> bool:
        if not self.relation_match(pred[1], gold[1]):
            return False
        rc = self.resolve_relation(gold[1])
        obj_ok = (
            self.literal_match(pred[2], gold[2])
            if rc in self.LITERAL_RELATIONS
            else self.entity_match(pred[2], gold[2])
        )
        return obj_ok and self.entity_match(pred[0], gold[0])


def _as_triple(d) -> tuple[str, str, str]:
    if isinstance(d, dict):
        return (d["subject"], d["relation"], d["object"])
    return (d[0], d[1], d[2])


def score(predictions: list, gold: list, matcher: TripleMatcher | None = None) -> dict:
    """Greedy one-to-one matching within each chunk. Returns precision/recall/F1,
    counts, per-relation F1 (keyed by the *gold* canonical relation), and the matched
    (pred, gold) index pairs for hand-checking.
    """
    matcher = matcher or TripleMatcher.load()
    preds = [d if isinstance(d, dict) else {"subject": d[0], "relation": d[1], "object": d[2]} for d in predictions]
    golds = [d if isinstance(d, dict) else {"subject": d[0], "relation": d[1], "object": d[2]} for d in gold]

    gold_used = [False] * len(golds)
    tp = 0
    matched_pairs: list[tuple[int, int]] = []
    # per-relation tallies on the gold side
    rel_total: dict[str, int] = {}
    rel_hit: dict[str, int] = {}
    for gi, g in enumerate(golds):
        rc = matcher.resolve_relation(g["relation"])
        rel_total[rc] = rel_total.get(rc, 0) + 1

    for pi, p in enumerate(preds):
        for gi, g in enumerate(golds):
            if gold_used[gi]:
                continue
            if p.get("chunk_id") and g.get("chunk_id") and p["chunk_id"] != g["chunk_id"]:
                continue
            if matcher.triple_match(_as_triple(p), _as_triple(g)):
                gold_used[gi] = True
                tp += 1
                matched_pairs.append((pi, gi))
                rc = matcher.resolve_relation(g["relation"])
                rel_hit[rc] = rel_hit.get(rc, 0) + 1
                break

    n_pred, n_gold = len(preds), len(golds)
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    per_relation = {}
    for rc, total in sorted(rel_total.items()):
        hit = rel_hit.get(rc, 0)
        # recall on gold side; precision per-relation needs pred-side relation tallies
        per_relation[rc] = {"gold": total, "matched": hit, "recall": hit / total}

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": n_pred - tp,
        "fn": n_gold - tp,
        "n_pred": n_pred,
        "n_gold": n_gold,
        "per_relation": per_relation,
        "matched_pairs": matched_pairs,
    }
