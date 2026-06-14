"""Step-0 re-verification of the FIXED extractor on the frozen 8-paragraph gold.

Scores ``predictions_gold_postfix.jsonl`` (produced once by the fixed extractor) against the
same frozen gold with the same frozen matcher used for the P1 gate — nothing here is re-tuned.
The P1 record (``EXTRACTION_BASELINE.md``) is left untouched; this prints the post-fix numbers
and the Step-0 gate verdict, which is recorded in ``GRAPH_BUILD.md``.

Step-0 gate (no regression from the bug fixes):
  - overall F1 >= 0.613
  - `performer` recall recovered (> 0; was 0.0 in P1)
  - core-relation F1 >= 0.85
"""
from __future__ import annotations

import sys

from .. import config
from ..baseline import corpus_io
from .extraction_eval import CORE_RELATIONS, _aggregate, per_relation_prf
from .triple_matcher import TripleMatcher

P1_F1 = 0.613
CORE_GATE = 0.85


def main() -> int:
    matcher = TripleMatcher.load()
    gold = corpus_io.load_jsonl(config.EXTRACTION_GOLD_PATH)
    preds = corpus_io.load_jsonl(config.EXTRACTION_PRED_POSTFIX_PATH)
    n_paras = len({g["chunk_id"] for g in gold})

    out = per_relation_prf(preds, gold, matcher)
    overall, table = out["overall"], out["per_relation"]
    core = _aggregate(table, CORE_RELATIONS)
    performer = table.get("performer", {"gold": 0, "pred": 0, "tp": 0, "recall": 0.0})

    print("=" * 64)
    print("STEP-0 POST-FIX EXTRACTION RE-VERIFY")
    print("=" * 64)
    print(f"paragraphs: {n_paras}   gold triples: {overall['n_gold']}   predicted: {overall['n_pred']}")
    print(f"OVERALL  P={overall['precision']:.3f}  R={overall['recall']:.3f}  F1={overall['f1']:.3f}  (P1 was {P1_F1})")
    print(f"CORE-REL P={core['precision']:.3f}  R={core['recall']:.3f}  F1={core['f1']:.3f}")
    print(f"performer: gold={performer['gold']} pred={performer['pred']} tp={performer['tp']} recall={performer['recall']:.3f}")
    print("per-relation F1:")
    for rc in sorted(table, key=lambda r: (-table[r]["gold"], r)):
        d = table[rc]
        tag = " *" if rc in CORE_RELATIONS else "  "
        print(f"  {tag}{rc:<24} gold={d['gold']:>2} pred={d['pred']:>2} tp={d['tp']:>2} F1={d['f1']:.2f}")

    ok_overall = overall["f1"] >= P1_F1
    ok_performer = performer["recall"] > 0
    ok_core = core["f1"] >= CORE_GATE
    passed = ok_overall and ok_performer and ok_core
    print("-" * 64)
    print(f"gate: overall F1 >= {P1_F1}: {ok_overall}   performer recovered: {ok_performer}   core F1 >= {CORE_GATE}: {ok_core}")
    print("STEP-0:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
