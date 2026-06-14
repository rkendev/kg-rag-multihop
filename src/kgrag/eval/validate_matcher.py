"""Validate the frozen triple matcher against hand-judged predicted/gold pairs.

The gate is only meaningful if the matcher itself is neither too lenient (passing a bad
extractor) nor too strict (failing a good one). This checks the matcher's verdicts against
``gold/matcher_validation.jsonl`` — pairs hand-authored from priors and labelled
``should_match`` before any extractor output existed — and reports agreement. Pure Python,
deterministic. Exit code 0 iff agreement is 100%.
"""
from __future__ import annotations

import sys

from .. import config
from ..baseline import corpus_io
from .triple_matcher import TripleMatcher

VAL_PATH = config.GOLD / "matcher_validation.jsonl"


def main() -> int:
    pairs = corpus_io.load_jsonl(VAL_PATH)
    matcher = TripleMatcher.load()
    agree = 0
    disagreements = []
    for p in pairs:
        verdict = matcher.triple_match(tuple(p["pred"]), tuple(p["gold"]))
        if verdict == p["should_match"]:
            agree += 1
        else:
            disagreements.append((p, verdict))

    n = len(pairs)
    pct = 100.0 * agree / n if n else 0.0
    print(f"matcher validation: {agree}/{n} agree ({pct:.1f}%)")
    for p, verdict in disagreements:
        print(f"  DISAGREE want={p['should_match']} got={verdict}: {p['pred']} vs {p['gold']} -- {p['note']}")
    ok = agree == n and n > 0
    print("MATCHER VALIDATION:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
