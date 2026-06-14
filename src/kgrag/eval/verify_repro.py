"""Deterministic reproducibility gate for the recorded baseline.

The gated, recorded targets — answer EM, token-F1, and support recall@k — depend only on
the stored answers and retrievals, so they must recompute *exactly* from the scored run
(``runs/scored_test.jsonl``). This check does that, instantly, with no LLM.

The provisional faithfulness/citation numbers are deliberately **not** re-verified here:
the local judge is non-deterministic on CPU and uncalibrated (a P4 concern), so its
verdicts are advisory only and gating on them adds cost without value.

Exit code 0 iff every deterministic metric reproduces.
"""
from __future__ import annotations

import sys

from .. import config
from ..baseline import corpus_io
from . import metrics

SCORED = config.RUNS_DIR / "scored_test.jsonl"


def main() -> int:
    scored = corpus_io.load_jsonl(SCORED)
    mismatch = 0
    for r in scored:
        if metrics.exact_match(r["answer"], r["gold_answer"]) != r["em"]:
            mismatch += 1
        if abs(metrics.token_f1(r["answer"], r["gold_answer"]) - r["f1"]) > 1e-12:
            mismatch += 1
        for k in config.RECALL_KS:
            rk = metrics.support_recall_at_k(r["retrieved_ids"], r["gold_support_chunk_ids"], k)
            rec = r[f"recall@{k}"]
            if not (rk == rec or (rk != rk and rec != rec)):  # NaN-safe compare
                mismatch += 1

    print(f"records: {len(scored)}  deterministic metric mismatches: {mismatch} (expect 0)")
    ok = mismatch == 0 and len(scored) > 0
    print("DETERMINISTIC REPRODUCIBILITY:", "PASS" if ok else "FAIL")
    print("(faithfulness/citation are advisory only and intentionally not gated)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
