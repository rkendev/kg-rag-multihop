"""Step 2 — corpus-scale entity resolution into canonical nodes.

Over-merging is the unrecoverable, gated failure: fusing two distinct entities poisons every
multi-hop answer that later traverses the node. So resolution is deliberately conservative and
**precision-first** — under-merging (lower recall, reported) is acceptable; over-merging is not.

Pipeline:
  1. Gather every subject/object mention with its GLiNER type and source chunk. The mention KEY
     is the alias-resolved normalized surface plus a disambiguating parenthetical tag when present
     (``Halloween (1978)`` vs ``(2018)``; ``Gangs of New York (film)`` vs ``(book)``), so works
     that the matcher's parenthetical-stripping would otherwise collapse stay distinct.
  2. **Exact-merge** is implicit: identical keys collapse (this reconnects an entity GLiNER
     labelled with different types across chunks).
  3. **Type-blocked conservative merge** for non-identical keys. The earlier spec
     (``token_set_ratio >= 92`` OR ``cosine >= 0.86``) over-merged badly: token_set_ratio scores
     any subset 100, so a single shared token ("Charles") chained every Charles together, and
     embedding cosine fused distinct-but-similar entities (different awards/films). It is replaced
     by structural rules (``merge_ok``): no compound/relational/cross-numeral fusion, honorific-
     and initial-aware name variants, a single added middle-name/epithet for PERSONs only, and a
     tight ``token_sort_ratio`` typo-catch for short names. Type-blocking prevents cross-type fusion.

Output: ``resolution/entities.jsonl`` and ``resolution/triples_resolved.jsonl`` (triples rewritten
to canonical entity_ids). Fully deterministic, pure-Python — re-resolving reproduces exactly.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict

from .. import config
from ..baseline import corpus_io
from ..eval.triple_matcher import TripleMatcher, parse_date

# GLiNER fine label -> coarse block. Same real entity must land in the same block regardless of
# which fine label GLiNER happened to assign, so related labels are grouped.
COARSE = {
    "person": "PERSON",
    "organization": "ORG", "company": "ORG",
    "location": "PLACE", "country": "PLACE", "city": "PLACE",
    "creative work": "WORK", "film": "WORK", "album": "WORK", "song": "WORK", "book": "WORK",
    "date": "DATE",
    "award": "AWARD",
    "role": "MISC", "occupation": "MISC", "nationality": "MISC",
}
# Relations whose object is a descriptive literal, not a named entity (mirrors the matcher).
LITERAL_RELATIONS = {"occupation", "genre", "cause of death"}

# Particles dropped before comparing name content (kept in the surface/canonical display form).
STOPWORDS = {"the", "of", "a", "an", "and", "de", "la", "le", "du", "des", "von", "van",
             "der", "den", "di", "da", "del", "el"}
# Titles/honorifics: their presence/absence must NOT distinguish two entities.
HONORIFICS = {
    "sir", "dame", "lord", "lady", "king", "queen", "prince", "princess", "infante", "infanta",
    "emperor", "empress", "duke", "duchess", "count", "countess", "earl", "viscount", "baron",
    "baroness", "dr", "doctor", "saint", "st", "pope", "cardinal", "archbishop", "bishop",
    "sheikh", "sheikha", "emir", "amir", "his", "her", "highness", "majesty", "mr", "mrs", "ms",
}
# Kinship nouns: a surface like "Isabel's brother" denotes a DIFFERENT entity than "Isabel".
RELATIONAL = {"brother", "sister", "wife", "husband", "son", "daughter", "cousin", "nephew",
              "niece", "uncle", "aunt", "widow", "widower"}
# A trailing keyword that turns a name into a DIFFERENT entity ("Andy Warhol" vs the
# "Andy Warhol Museum"); such an added token must block a name-subset merge.
DISTINCT_SUFFIX = {"museum", "foundation", "award", "prize", "university", "college", "school",
                   "hospital", "airport", "station", "company", "society", "institute", "gallery",
                   "library", "theatre", "theater", "park", "trophy", "medal", "cup", "stadium",
                   "bridge", "memorial", "trust", "fund"}
# Regnal numerals / ordinals / generational suffixes that DISAMBIGUATE same-named people.
_NUMERAL_TOKENS = {
    "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", "xiii", "xiv",
    "xv", "xvi", "xvii", "xviii", "xix", "xx", "xxi", "xxii", "jr", "jnr", "sr", "snr",
}
_DIGITS = re.compile(r"^\d+$")
_ORDINAL = re.compile(r"^\d+(st|nd|rd|th)$")
_PAREN_TAG = re.compile(
    r"\((film|movie|book|novel|play|tv series|miniseries|series|album|song|soundtrack|comics?|"
    r"video game|opera|\d{4})\)", re.IGNORECASE,
)


def _numerals(tokens: list[str]) -> set[str]:
    return {t for t in tokens if t in _NUMERAL_TOKENS or _DIGITS.match(t) or _ORDINAL.match(t)}


def _content(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in STOPWORDS]


def is_compound(surface: str) -> bool:
    """Surface denotes >1 entity (a list/conjunction) — must not pivot a merge."""
    s = surface.strip()
    if " & " in s or re.search(r"\band\b", s, re.IGNORECASE):
        return True
    # "A, B, C" style lists of capitalized names
    if s.count(",") >= 1 and len(re.findall(r"[A-Z][a-z]+", s)) >= 3:
        return True
    return False


def is_relational(surface: str) -> bool:
    """Surface is a relational reference ("Isabel's brother") — a different entity."""
    low = surface.lower()
    if "'s " in low or "’s " in low:
        return True
    return any(re.search(rf"\b{w}\b", low) for w in RELATIONAL)


def mention_key(matcher: TripleMatcher, surface: str) -> str:
    """Alias-resolved normalized key + a disambiguating parenthetical tag when present, so
    ``X (film)``, ``X (book)`` and ``X (1978)`` resolve to distinct entities."""
    base = matcher.resolve_entity(surface)
    m = _PAREN_TAG.search(surface or "")
    return f"{base}|{m.group(1).lower()}" if m else base


def _base(key: str) -> str:
    return key.split("|", 1)[0]


def name_variant(ca: list[str], cb: list[str]) -> bool:
    """Same content tokens modulo order and initial/full-name abbreviation
    ("George A. Romero" ~ "George Andrew Romero"; "Shimizu Takashi" ~ "Takashi Shimizu")."""
    if len(ca) != len(cb):
        return False
    for x, y in zip(sorted(ca), sorted(cb)):
        if x == y:
            continue
        if (len(x) == 1 and y.startswith(x)) or (len(y) == 1 and x.startswith(y)):
            continue
        return False
    return True


def merge_ok(ka: str, kb: str, btype: str) -> bool:
    """Conservative, precision-first decision whether two keys denote the same entity."""
    if ka.split("|")[1:] != kb.split("|")[1:]:    # different disambiguating tags -> distinct
        return False                              # (one tagged vs not, or film vs book vs year)
    a, b = _base(ka), _base(kb)
    if a == b:
        return False
    ta, tb = a.split(), b.split()
    if _numerals(ta) != _numerals(tb):            # different regnal/ordinal/suffix -> distinct
        return False
    ca, cb = _content(ta), _content(tb)
    # Require >=2 content tokens on BOTH sides: a bare single name ("John") must never fuzzy-
    # or subset-merge, or it pivots every same-first-name person into one cluster.
    if len(ca) < 2 or len(cb) < 2:
        return False
    if name_variant(ca, cb):
        return True
    sa, sb = set(ca), set(cb)
    if sa <= sb or sb <= sa:                       # one name is contained in the other
        extra = (sb - sa) if sa <= sb else (sa - sb)
        nonhon = extra - HONORIFICS
        if nonhon & DISTINCT_SUFFIX:              # "X" vs "X Museum/Foundation/..." -> distinct
            return False
        if not nonhon:                            # only honorifics/particles added
            return True
        if btype == "PERSON" and len(nonhon) == 1:  # a single added middle name / epithet
            return True
        return False
    # tight typo / spacing catch for SHORT names only (avoid long-title near-collisions)
    if max(len(ta), len(tb)) <= 4:
        from rapidfuzz import fuzz
        if fuzz.token_sort_ratio(a, b) >= config.RESOLVE_TYPO_RATIO:
            return True
    return False


def coarse_type(fine: str | None, surface: str, *, is_literal: bool) -> str:
    if fine and fine in COARSE:
        return COARSE[fine]
    if parse_date(surface) is not None:
        return "DATE"
    return "LITERAL" if is_literal else "MISC"


def gather_mentions(preds: list[dict], matcher: TripleMatcher):
    """{key: {"surfaces": Counter, "types": Counter, "chunks": set, "no_merge": bool}}.

    ``no_merge`` flags keys whose representative surface is a compound/relational reference; those
    keep their own node and never pivot a merge.
    """
    mentions: dict[str, dict] = {}

    def add(surface: str, fine: str | None, chunk_id: str, *, is_literal: bool):
        surface = (surface or "").strip()
        if not surface:
            return
        key = mention_key(matcher, surface)
        if not _base(key):
            return
        m = mentions.setdefault(
            key, {"surfaces": Counter(), "types": Counter(), "chunks": set(),
                  "compound": 0, "clean": 0})
        m["surfaces"][surface] += 1
        m["types"][coarse_type(fine, surface, is_literal=is_literal)] += 1
        m["chunks"].add(chunk_id)
        if is_compound(surface) or is_relational(surface):
            m["compound"] += 1
        else:
            m["clean"] += 1

    for p in preds:
        rel = matcher.resolve_relation(p["relation"])
        add(p["subject"], p.get("subj_type"), p["chunk_id"], is_literal=False)
        add(p["object"], p.get("obj_type"), p["chunk_id"], is_literal=(rel in LITERAL_RELATIONS))
    for m in mentions.values():
        m["no_merge"] = m["clean"] == 0          # only compound/relational surfaces seen
    return mentions


class _UF:
    def __init__(self, keys):
        self.parent = {k: k for k in keys}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            lo, hi = sorted((ra, rb))            # deterministic root: smaller key wins
            self.parent[hi] = lo


def _block_type(types: Counter) -> str:
    return sorted(types.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def resolve(preds: list[dict]):
    matcher = TripleMatcher.load()
    mentions = gather_mentions(preds, matcher)
    keys = sorted(mentions)
    uf = _UF(keys)

    blocks: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        blocks[_block_type(mentions[k]["types"])].append(k)

    n_merges = 0
    for btype, bkeys in blocks.items():
        if btype == "DATE":
            continue                              # dates only ever exact-merge
        bkeys = sorted(k for k in bkeys if not mentions[k]["no_merge"])
        for i in range(len(bkeys)):
            for j in range(i + 1, len(bkeys)):
                a, b = bkeys[i], bkeys[j]
                if uf.find(a) == uf.find(b):
                    continue
                if merge_ok(a, b, btype):
                    uf.union(a, b); n_merges += 1

    clusters: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        clusters[uf.find(k)].append(k)

    entities = []
    key_to_eid: dict[str, str] = {}
    for idx, (root, members) in enumerate(sorted(clusters.items()), 1):
        eid = f"e{idx:05d}"
        surf_counter: Counter = Counter()
        type_counter: Counter = Counter()
        chunks: set[str] = set()
        aliases = []
        for k in sorted(members):
            m = mentions[k]
            surf_counter.update(m["surfaces"])
            type_counter.update(m["types"])
            chunks |= m["chunks"]
            key_to_eid[k] = eid
            for surf in sorted(m["surfaces"]):
                aliases.append({"surface": surf, "count": m["surfaces"][surf], "chunks": sorted(m["chunks"])})
        canonical = sorted(surf_counter.items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))[0][0]
        etype = sorted(type_counter.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        entities.append({
            "entity_id": eid,
            "canonical_name": canonical,
            "type": etype,
            "n_surface_forms": len({a["surface"] for a in aliases}),
            "n_chunks": len(chunks),
            "aliases": aliases,
            "_keys": sorted(members),
        })

    return matcher, mentions, entities, key_to_eid


def rewrite_triples(preds, matcher, key_to_eid) -> list[dict]:
    out = []
    for p in preds:
        out.append({
            **p,
            "subj_id": key_to_eid.get(mention_key(matcher, p["subject"])),
            "obj_id": key_to_eid.get(mention_key(matcher, p["object"])),
        })
    return out


def main() -> int:
    preds = corpus_io.load_jsonl(config.EXTRACTION_PRED_CORPUS_PATH)
    matcher, mentions, entities, key_to_eid = resolve(preds)
    resolved = rewrite_triples(preds, matcher, key_to_eid)

    config.RESOLUTION_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RESOLUTION_ENTITIES_PATH, "w", encoding="utf-8") as f:
        for e in entities:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(config.RESOLUTION_TRIPLES_PATH, "w", encoding="utf-8") as f:
        for t in resolved:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    merged = [e for e in entities if e["n_surface_forms"] >= 2]
    print(f"[resolve] mentions(unique keys)={len(mentions)}  canonical entities={len(entities)}")
    print(f"[resolve] merged clusters (>=2 surface forms)={len(merged)}")
    print(f"[resolve] wrote {config.RESOLUTION_ENTITIES_PATH} and {config.RESOLUTION_TRIPLES_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
