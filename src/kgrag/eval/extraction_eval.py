"""Score stored extraction predictions against the hand gold; decide the P1 gate.

Deterministic: reads ``predictions.jsonl`` (produced once by ``extract_triples``) and the
frozen gold, scores with the frozen matcher, and writes ``EXTRACTION_BASELINE.md`` with
overall precision/recall/F1, per-relation F1, the core-relation aggregate, the matcher
agreement, and the go/no-go decision. No LLM is run here.

Gate decision rule (pre-committed, before any prediction was scored):
  - overall F1 >= 0.65          -> GO (clear pass)
  - overall F1 <  0.55          -> NO-GO (clear fail; pivot to hardened flat-RAG)
  - 0.55 <= overall F1 < 0.65   -> NEAR THE LINE: decide on the core-relation F1
        (director, father, spouse, date of birth, place of birth), because the
        micro-average over a small gold is dominated by high-frequency descriptive
        relations (occupation/performer). GO iff core-relation F1 >= 0.60.
  Per the approved marginal policy, a pass in [0.60, 0.65] is an immediate GO (no extra
  extractor-tuning round). Small-sample caveat is recorded alongside.
"""
from __future__ import annotations

import sys

from .. import config
from ..baseline import corpus_io
from .triple_matcher import TripleMatcher

CORE_RELATIONS = ["director", "father", "spouse", "date of birth", "place of birth"]
BASELINE_MD = config.ROOT / "EXTRACTION_BASELINE.md"


def per_relation_prf(preds, golds, matcher: TripleMatcher) -> dict:
    """Per-canonical-relation precision/recall/F1 (gold-side recall, pred-side precision)."""
    from .triple_matcher import score as score_fn

    res = score_fn(preds, golds, matcher)
    matched_pred = {pi for pi, _ in res["matched_pairs"]}

    gold_total, pred_total, tp = {}, {}, {}
    for g in golds:
        rc = matcher.resolve_relation(g["relation"])
        gold_total[rc] = gold_total.get(rc, 0) + 1
    for pi, p in enumerate(preds):
        rc = matcher.resolve_relation(p["relation"])
        pred_total[rc] = pred_total.get(rc, 0) + 1
        if pi in matched_pred:
            tp[rc] = tp.get(rc, 0) + 1

    rels = sorted(set(gold_total) | set(pred_total))
    table = {}
    for rc in rels:
        g, pr, t = gold_total.get(rc, 0), pred_total.get(rc, 0), tp.get(rc, 0)
        p = t / pr if pr else 0.0
        r = t / g if g else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        table[rc] = {"gold": g, "pred": pr, "tp": t, "precision": p, "recall": r, "f1": f1}
    return {"overall": res, "per_relation": table}


def _aggregate(table: dict, relations: list[str]) -> dict:
    g = sum(table[r]["gold"] for r in relations if r in table)
    pr = sum(table[r]["pred"] for r in relations if r in table)
    t = sum(table[r]["tp"] for r in relations if r in table)
    p = t / pr if pr else 0.0
    r = t / g if g else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"gold": g, "pred": pr, "tp": t, "precision": p, "recall": r, "f1": f1}


def decide(overall_f1: float, core_f1: float) -> tuple[str, str]:
    if overall_f1 >= 0.65:
        return "GO", "Overall F1 >= 0.65 — extraction is clearly good enough to build the graph on."
    if overall_f1 < 0.55:
        return "NO-GO", (
            "Overall F1 < 0.55 — extraction is not good enough; errors would compound through "
            "every multi-hop traversal. Pivot to hardened flat-RAG (the P0 baseline as the "
            "shipped system); record as an honest negative."
        )
    # near the line: decide on core-relation F1
    if core_f1 >= 0.60:
        return "GO", (
            f"Overall F1 in [0.55,0.65) (near the line); core-relation F1 = {core_f1:.3f} >= 0.60, "
            "so the relations that actually carry multi-hop bridges are reliable. Immediate GO per "
            "the approved marginal policy. Small gold (n=97) — treat as provisional and re-measure at corpus scale in P2."
        )
    return "NO-GO", (
        f"Overall F1 in [0.55,0.65) (near the line) but core-relation F1 = {core_f1:.3f} < 0.60 — the "
        "bridge-carrying relations are too weak to build on. Pivot to hardened flat-RAG; honest negative."
    )


def build_report(g, table, core, matcher_agreement, overall, n_paras) -> str:
    decision, rationale = decide(overall["f1"], core["f1"])
    weak = sorted(
        [(rc, d) for rc, d in table.items() if d["gold"] >= 1 and d["f1"] < 0.5],
        key=lambda kv: (kv[1]["f1"], -kv[1]["gold"]),
    )
    lines = []
    lines.append("# EXTRACTION_BASELINE.md — Phase P1 triple-extraction gate")
    lines.append("")
    lines.append(
        "Local triple extraction (GLiNER entities + Ollama `qwen2.5:7b-instruct` relations, "
        "temperature 0) scored against a frozen hand-annotated gold with an "
        "entity-resolution-aware matcher. **This is a gate, not a deliverable:** it decides "
        "whether a graph built from this extractor could beat the P0 flat-RAG baseline."
    )
    lines.append("")
    lines.append("## Setup (all frozen before scoring — see git history)")
    lines.append("")
    lines.append(f"- Gold: **{overall['n_gold']} hand-annotated triples** across **{n_paras} paragraphs** of the frozen P0 corpus, exhaustively labelled per `gold/ANNOTATION_GUIDELINE.md`.")
    lines.append("- Entities: `urchade/gliner_medium-v2.1` (open zero-shot schema, CPU).")
    lines.append("- Relations: Ollama `qwen2.5:7b-instruct`, JSON mode, temperature 0, fixed seed; predictions persisted once to `data/processed/extraction/predictions.jsonl`.")
    lines.append(f"- Matcher: `src/kgrag/eval/triple_matcher.py` + frozen `gold/alias_table.json`, `gold/relation_synonyms.json`. Validated at **{matcher_agreement}** agreement on 20 hand-judged pairs (`gold/matcher_validation.jsonl`).")
    lines.append("")
    lines.append("## Overall (micro-averaged over all triples)")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| predicted triples | {overall['n_pred']} |")
    lines.append(f"| gold triples | {overall['n_gold']} |")
    lines.append(f"| true positives | {overall['tp']} |")
    lines.append(f"| precision | **{overall['precision']:.3f}** |")
    lines.append(f"| recall | **{overall['recall']:.3f}** |")
    lines.append(f"| **F1** | **{overall['f1']:.3f}** |")
    lines.append("")
    lines.append("## Core-relation aggregate (the bridge-carrying relations)")
    lines.append("")
    lines.append("These five relations carry the multi-hop bridges the graph must traverse; near the 0.60 line they are weighted over the micro-average, which is dominated by high-frequency descriptive relations (occupation/performer).")
    lines.append("")
    lines.append(f"- Relations: {', '.join('`'+r+'`' for r in CORE_RELATIONS)}")
    lines.append(f"- Core gold triples: {core['gold']}; TP: {core['tp']}")
    lines.append(f"- Core precision / recall / **F1**: {core['precision']:.3f} / {core['recall']:.3f} / **{core['f1']:.3f}**")
    lines.append("")
    lines.append("## Per-relation breakdown")
    lines.append("")
    lines.append("| relation | gold | pred | TP | precision | recall | F1 |")
    lines.append("|---|---|---|---|---|---|---|")
    for rc in sorted(table, key=lambda r: (-table[r]["gold"], r)):
        d = table[rc]
        star = " ⭐" if rc in CORE_RELATIONS else ""
        lines.append(f"| {rc}{star} | {d['gold']} | {d['pred']} | {d['tp']} | {d['precision']:.2f} | {d['recall']:.2f} | {d['f1']:.2f} |")
    lines.append("")
    lines.append("⭐ = core (bridge-carrying) relation.")
    lines.append("")
    lines.append("## Weak predicates (carry into P2)")
    lines.append("")
    if weak:
        for rc, d in weak:
            lines.append(f"- `{rc}` — F1 {d['f1']:.2f} (gold {d['gold']}, pred {d['pred']}, TP {d['tp']}).")
    else:
        lines.append("- None below F1 0.50.")
    lines.append("")
    lines.append("## Gate decision")
    lines.append("")
    lines.append(f"**{decision}** — {rationale}")
    lines.append("")
    lines.append("### Decision rule (pre-committed, before scoring)")
    lines.append("- F1 ≥ 0.65 → GO; F1 < 0.55 → NO-GO; 0.55 ≤ F1 < 0.65 → decide on core-relation F1 (GO iff ≥ 0.60). A pass in [0.60,0.65] is an immediate GO (no extra tuning round).")
    lines.append("")
    lines.append("### Small-sample caveat")
    lines.append(f"- The gold is {overall['n_gold']} triples over {n_paras} paragraphs. Per-relation numbers with single-digit support are indicative only; several relations have gold support of 1–3. These F1s are a go/no-go signal for whether to *start* P2, not a calibrated corpus-scale accuracy. P2 must re-measure extraction quality at scale before relying on weak predicates.")
    lines.append("")
    lines.append("## Reproduce (deterministic — no LLM re-run)")
    lines.append("```bash")
    lines.append("just extraction-eval     # rescore stored predictions, rewrite this file")
    lines.append("just verify-extraction   # confirm F1 recomputes exactly from stored predictions")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    matcher = TripleMatcher.load()
    gold = corpus_io.load_jsonl(config.EXTRACTION_GOLD_PATH)
    preds = corpus_io.load_jsonl(config.EXTRACTION_PRED_PATH)
    n_paras = len({g["chunk_id"] for g in gold})

    out = per_relation_prf(preds, gold, matcher)
    overall, table = out["overall"], out["per_relation"]
    core = _aggregate(table, CORE_RELATIONS)

    # matcher agreement (recompute from the frozen validation set)
    val = corpus_io.load_jsonl(config.GOLD / "matcher_validation.jsonl")
    agree = sum(1 for p in val if matcher.triple_match(tuple(p["pred"]), tuple(p["gold"])) == p["should_match"])
    matcher_agreement = f"{agree}/{len(val)}"

    report = build_report(gold, table, core, matcher_agreement, overall, n_paras)
    BASELINE_MD.write_text(report, encoding="utf-8")

    decision, rationale = decide(overall["f1"], core["f1"])
    print("=" * 64)
    print("P1 EXTRACTION GATE — SUMMARY")
    print("=" * 64)
    print(f"paragraphs: {n_paras}   gold triples: {overall['n_gold']}   predicted: {overall['n_pred']}")
    print(f"OVERALL  P={overall['precision']:.3f}  R={overall['recall']:.3f}  F1={overall['f1']:.3f}")
    print(f"CORE-REL P={core['precision']:.3f}  R={core['recall']:.3f}  F1={core['f1']:.3f}  ({', '.join(CORE_RELATIONS)})")
    print(f"matcher agreement: {matcher_agreement}")
    print("per-relation F1:")
    for rc in sorted(table, key=lambda r: (-table[r]["gold"], r)):
        d = table[rc]
        tag = " *" if rc in CORE_RELATIONS else "  "
        print(f"  {tag}{rc:<24} gold={d['gold']:>2} pred={d['pred']:>2} tp={d['tp']:>2} F1={d['f1']:.2f}")
    print("-" * 64)
    print(f"DECISION: {decision}")
    print(rationale)
    print(f"wrote {BASELINE_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
