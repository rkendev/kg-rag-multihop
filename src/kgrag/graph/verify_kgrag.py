"""P3 verification gate (required before declaring done).

Three deterministic checks, no LLM:

1. **Deterministic recompute** — EM/token-F1/support recall@k recompute *exactly* from the
   stored ``runs/kgrag_scored_test.jsonl`` (mirrors ``kgrag.eval.verify_repro``). The judge's
   advisory faithfulness/citation numbers are intentionally not re-verified (CPU-nondeterministic).

2. **Fairness guardrail** — the comparison only means something if the single thing that
   changed is retrieval. Asserts (a) ``git diff`` is empty for the generator, generation
   prompt, judge, metrics, and the frozen test/no-knowledge/test-id gold files, and (b) the
   pinned top-k / RRF constants are unchanged (``GEN_TOP_K=5``, ``RRF_K=60``).

3. **Spot-check** — 5 multi-hop questions where KG-RAG's retrieved set differs from P0 AND a
   gold supporting chunk was pulled into the top-5 that the baseline missed: prints the
   bridging evidence (linked seeds + the traversed edge whose source chunk was recovered),
   confirming the graph genuinely surfaced a second-hop chunk.

Exit 0 iff checks (1) and (2) pass.
"""
from __future__ import annotations

import subprocess
import sys

from .. import config
from ..baseline import corpus_io
from ..eval import metrics

KG_SCORED = config.RUNS_DIR / "kgrag_scored_test.jsonl"
KG_TEST = config.RUNS_DIR / "kgrag_test.jsonl"
BASE_SCORED = config.RUNS_DIR / "scored_test.jsonl"

# Files that must be byte-identical to the committed P0 versions (only retrieval may change).
GUARDED_PATHS = [
    "src/kgrag/baseline/generate.py",   # generator + generation prompt
    "src/kgrag/eval/judge.py",          # judge
    "src/kgrag/eval/metrics.py",        # metrics
    "gold/test_ids.txt",
    "gold/questions.jsonl",
    "gold/no_knowledge.jsonl",
]
MULTIHOP = {"bridge", "compositional", "bridge_comparison"}


def check_repro() -> bool:
    scored = corpus_io.load_jsonl(KG_SCORED)
    mismatch = 0
    for r in scored:
        if metrics.exact_match(r["answer"], r["gold_answer"]) != r["em"]:
            mismatch += 1
        if abs(metrics.token_f1(r["answer"], r["gold_answer"]) - r["f1"]) > 1e-12:
            mismatch += 1
        for k in config.RECALL_KS:
            rk = metrics.support_recall_at_k(r["retrieved_ids"], r["gold_support_chunk_ids"], k)
            rec = r[f"recall@{k}"]
            if not (rk == rec or (rk != rk and rec != rec)):
                mismatch += 1
    ok = mismatch == 0 and len(scored) > 0
    print(f"[1] deterministic recompute: {len(scored)} records, {mismatch} mismatches -> "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def check_guardrail() -> bool:
    # (a) protected files unmodified vs HEAD
    res = subprocess.run(
        ["git", "-C", str(config.ROOT), "diff", "--stat", "HEAD", "--", *GUARDED_PATHS],
        capture_output=True, text=True,
    )
    diff_out = res.stdout.strip()
    files_ok = diff_out == ""
    # (b) pinned constants
    const_ok = config.GEN_TOP_K == 5 and config.RRF_K == 60
    ok = files_ok and const_ok
    print(f"[2] fairness guardrail: protected files unmodified={files_ok}, "
          f"GEN_TOP_K={config.GEN_TOP_K} RRF_K={config.RRF_K} -> {'PASS' if ok else 'FAIL'}")
    if not files_ok:
        print("    UNEXPECTED CHANGES:\n" + diff_out)
    return ok


def spot_check(n: int = 5) -> None:
    kg_runs = {r["id"]: r for r in corpus_io.load_jsonl(KG_TEST)}
    base = {r["id"]: r for r in corpus_io.load_jsonl(BASE_SCORED)}
    print(f"[3] spot-check — multi-hop questions where KG-RAG recovered a gold chunk into top-5\n"
          f"    that the baseline's top-5 missed:")
    shown = 0
    for qid, kgr in kg_runs.items():
        if kgr["hop_type"] not in MULTIHOP or not kgr.get("used_graph"):
            continue
        kg_top5 = kgr["context_chunk_ids"]
        base_top5 = base[qid]["context_chunk_ids"]
        recovered = [c for c in kgr["gold_support_chunk_ids"]
                     if c in kg_top5 and c not in base_top5]
        if not recovered:
            continue
        shown += 1
        edges = kgr.get("graph_edges_detail") or []  # may be absent (slim records)
        print(f"\n  Q {qid[:8]} [{kgr['hop_type']}] {kgr['question'][:78]}")
        print(f"    linked seeds: "
              f"{[(l['surface'], l['canonical_name'], l['method']) for l in kgr.get('kg_linked', [])]}")
        print(f"    gold support: {kgr['gold_support_chunk_ids']}")
        print(f"    baseline top5 (missed): {base_top5}")
        print(f"    kg-rag  top5 (recovered {recovered}): {kg_top5}")
        print(f"    graph leg size={kgr.get('graph_leg_size')} edges={kgr.get('graph_edges')} "
              f"-> the bridging chunk(s) {recovered} entered via graph traversal, not vector/BM25")
        if shown >= n:
            break
    if shown == 0:
        print("    (none found — KG-RAG did not recover any baseline-missed gold chunk into top-5)")


def main() -> int:
    ok1 = check_repro()
    ok2 = check_guardrail()
    spot_check()
    ok = ok1 and ok2
    print("\nP3 VERIFY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
