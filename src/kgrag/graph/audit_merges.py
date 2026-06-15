"""Merge-precision audit aid for the P2 gate (Step 2).

Dumps every merged canonical entity (>=2 distinct surface forms) with, per alias, one source
sentence from the chunk it came from — so a human (or this agent) can judge whether the cluster
fuses two genuinely different real-world entities. Precision = 1 - (wrong clusters / sampled).
The gate requires >= 0.90 (<= 10% wrong); over-merging is the gated failure.

Read-only; prints a numbered list for hand-marking. No LLM.
"""
from __future__ import annotations

import json
import re
import sys

from .. import config
from ..baseline import corpus_io


def _sentence_for(text: str, surface: str) -> str:
    """Return the sentence in ``text`` mentioning ``surface`` (best-effort, trimmed)."""
    i = text.lower().find(surface.lower())
    if i < 0:
        # fall back to first token of the surface
        tok = surface.split()[0] if surface.split() else surface
        i = text.lower().find(tok.lower())
    if i < 0:
        return "(surface not located in chunk text)"
    start = max(text.rfind(".", 0, i), text.rfind("\n", 0, i)) + 1
    end = text.find(".", i)
    end = end + 1 if end >= 0 else len(text)
    return re.sub(r"\s+", " ", text[start:end]).strip()[:240]


def main() -> int:
    ents = corpus_io.load_jsonl(config.RESOLUTION_ENTITIES_PATH)
    corpus = {c["chunk_id"]: c for c in corpus_io.load_corpus()}
    merged = [e for e in ents if e["n_surface_forms"] >= 2]
    merged.sort(key=lambda e: (-e["n_surface_forms"], e["canonical_name"]))

    print(f"MERGED CLUSTERS (>=2 surface forms): {len(merged)}  | total entities: {len(ents)}")
    print("=" * 72)
    for idx, e in enumerate(merged, 1):
        print(f"[{idx}] ({e['type']}) canonical={e['canonical_name']!r}  forms={e['n_surface_forms']} chunks={e['n_chunks']}")
        # one example sentence per distinct surface form
        seen = set()
        for a in e["aliases"]:
            surf = a["surface"]
            if surf in seen:
                continue
            seen.add(surf)
            cid = a["chunks"][0] if a.get("chunks") else None
            sent = _sentence_for(corpus[cid]["text"], surf) if cid in corpus else "(no chunk)"
            print(f"      - {surf!r}  [{cid}]  :: {sent}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
