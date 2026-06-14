"""Deterministic reproducibility gate for the P1 extraction result.

The qwen extraction is non-deterministic in wall-clock and was run exactly once; its
predictions are persisted. The gate number (triple F1 from the frozen matcher) is pure
Python over those stored predictions, so it must recompute **exactly** every time. This
check recomputes it twice from the stored file and confirms the two runs are identical and
that the matcher still validates — without ever re-running the LLM.

It also prints 10 predicted true positives for a manual over-merge spot-check.
Exit code 0 iff F1 reproduces and the matcher still passes its validation.
"""
from __future__ import annotations

import sys

from .. import config
from ..baseline import corpus_io
from .triple_matcher import TripleMatcher, score

N_SPOTCHECK = 10


def main() -> int:
    matcher = TripleMatcher.load()
    gold = corpus_io.load_jsonl(config.EXTRACTION_GOLD_PATH)
    preds = corpus_io.load_jsonl(config.EXTRACTION_PRED_PATH)

    r1 = score(preds, gold, matcher)
    r2 = score(preds, gold, matcher)
    reproducible = (
        r1["f1"] == r2["f1"] and r1["precision"] == r2["precision"] and r1["recall"] == r2["recall"]
        and r1["tp"] == r2["tp"]
    )
    print(f"stored predictions: {r1['n_pred']}   gold: {r1['n_gold']}")
    print(f"recompute #1 F1={r1['f1']:.6f}  #2 F1={r2['f1']:.6f}  identical={reproducible}")

    # matcher still validates
    val = corpus_io.load_jsonl(config.GOLD / "matcher_validation.jsonl")
    agree = sum(1 for p in val if matcher.triple_match(tuple(p["pred"]), tuple(p["gold"])) == p["should_match"])
    matcher_ok = agree == len(val)
    print(f"matcher validation: {agree}/{len(val)} agree")

    # spot-check: 10 predicted true positives -> their matched gold triple
    print(f"\n{N_SPOTCHECK} predicted true positives (manual over-merge check):")
    for pi, gi in r1["matched_pairs"][:N_SPOTCHECK]:
        p, g = preds[pi], gold[gi]
        print(f"  PRED ({p['subject']} | {p['relation']} | {p['object']})")
        print(f"    -> GOLD ({g['subject']} | {g['relation']} | {g['object']})  [{g['chunk_id']}]")

    ok = reproducible and matcher_ok and r1["n_pred"] > 0
    print("\nEXTRACTION REPRODUCIBILITY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
