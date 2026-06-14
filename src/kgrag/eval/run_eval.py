"""Score the baseline run and (re)write BASELINE.md.

Reads the per-question records produced by ``kgrag.baseline.run_baseline`` and computes:
  * answer EM + token-F1 vs gold,
  * supporting-chunk recall@k,
  * faithfulness + citation correctness via the local judge (PROVISIONAL),
  * abstention rate on the no-knowledge set (and over-abstention on answerable test),
all broken down by hop_type and hop_count. The multi-hop target the graph must beat is
called out explicitly.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict

from .. import config
from ..baseline import corpus_io
from . import judge, metrics

TEST_RUN = config.RUNS_DIR / "baseline_test.jsonl"
NK_RUN = config.RUNS_DIR / "baseline_no_knowledge.jsonl"
SCORED_PATH = config.RUNS_DIR / "scored_test.jsonl"


def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if x == x]  # drop NaN
    return statistics.fmean(xs) if xs else float("nan")


def pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.1f}"


def score_test(records: list[dict], by_id_corpus: dict, *, run_judge: bool) -> list[dict]:
    for r in records:
        r["em"] = metrics.exact_match(r["answer"], r["gold_answer"])
        r["f1"] = metrics.token_f1(r["answer"], r["gold_answer"])
        for k in config.RECALL_KS:
            r[f"recall@{k}"] = metrics.support_recall_at_k(
                r["retrieved_ids"], r["gold_support_chunk_ids"], k
            )
        # judge only answered (non-abstained) questions
        if run_judge and not r["abstained"]:
            chunks = [by_id_corpus[c] for c in r["context_chunk_ids"]]
            verdict = judge.judge_answer(r["question"], r["answer"], r["citations"], chunks)
            r.update(verdict)
            print(f"judged {r['id']}: faithful={verdict['faithful']} "
                  f"cite={verdict['citations_correct']}", flush=True)
    return records


def aggregate(records: list[dict]) -> dict:
    em = _mean([r["em"] for r in records])
    f1 = _mean([r["f1"] for r in records])
    out = {"n": len(records), "em": em, "f1": f1}
    for k in config.RECALL_KS:
        out[f"recall@{k}"] = _mean([r[f"recall@{k}"] for r in records])
    answered = [r for r in records if not r["abstained"]]
    out["over_abstain"] = _mean([float(r["abstained"]) for r in records])
    judged = [r for r in answered if "faithful" in r]
    out["n_judged"] = len(judged)
    out["faithful"] = _mean([float(r["faithful"]) for r in judged]) if judged else float("nan")
    out["citations_correct"] = (
        _mean([float(r["citations_correct"]) for r in judged]) if judged else float("nan")
    )
    return out


def group_by(records: list[dict], key: str) -> dict:
    groups: dict = defaultdict(list)
    for r in records:
        groups[r[key]].append(r)
    return {k: aggregate(v) for k, v in groups.items()}


def _row(label, a) -> str:
    return (
        f"| {label} | {a['n']} | {pct(a['em'])} | {pct(a['f1'])} | "
        f"{pct(a['recall@1'])} | {pct(a['recall@5'])} | {pct(a['recall@10'])} | "
        f"{pct(a['recall@20'])} |"
    )


def write_baseline_md(stats: dict, overall: dict, by_type: dict, by_count: dict,
                      nk: dict) -> None:
    L: list[str] = []
    A = L.append
    A("# BASELINE.md — Phase P0 flat-RAG baseline\n")
    A("Recorded flat-RAG baseline over the frozen 2WikiMultiHopQA corpus. **No graph.** "
      "This is the number later phases must beat to justify the knowledge graph.\n")

    A("## Corpus (frozen)\n")
    A(f"- Chunks: **{stats['n_chunks']}** (ceiling {stats['chunk_ceiling']}) — "
      f"paragraph-level, deduped by (title, normalized text).")
    A(f"- Paragraphs pooled: {stats['n_paragraphs_pooled']} → {stats['n_chunks']} unique chunks.")
    A(f"- Token estimate: ~{stats['token_estimate']:,} (whitespace words × 1.3; "
      f"{stats['n_words']:,} words).")
    A(f"- Corpus SHA-256: `{stats['corpus_sha256']}`")
    A(f"- Provenance on every chunk: `chunk_id`, `source_title`, `source_para_idx`.")
    A(f"- Embedding model: `{stats['embed_model']}`; seed `{stats['seed']}`.\n")

    A("## Gold sets (frozen)\n")
    A(f"- Answerable questions: {stats['n_answerable_questions']} "
      f"({stats['n_dev']} dev / {stats['n_test']} held-out test, "
      f"{int(stats['test_frac']*100)}% stratified by hop_type — test slice never tuned on).")
    A(f"- No-knowledge (abstention) questions: {stats['n_no_knowledge_questions']} "
      f"— gold supporting titles absent from the corpus, so abstention is the only correct answer.")
    A(f"- hop_type distribution (all answerable): {stats['hop_type_counts']}.")
    A(f"- hop_count distribution: {stats['hop_count_counts']} "
      f"(hop_count = number of distinct gold supporting titles; bridge_comparison = 4).\n")

    A("## Baseline configuration\n")
    A(f"- Retrieval: BGE dense (`{config.EMBED_MODEL}`, FAISS flat IP) + BM25 "
      f"(`rank_bm25`), fused with Reciprocal Rank Fusion (k={config.RRF_K}).")
    A(f"- Generation: Ollama `{config.GEN_MODEL}`, temperature 0, top-{config.GEN_TOP_K} "
      f"context, answers with `[chunk_id]` citations; abstains with `{config.ABSTAIN_TOKEN}`.")
    A(f"- Judge (faithfulness/citation): Ollama `{config.JUDGE_MODEL}`, temperature 0 — "
      f"**PROVISIONAL**, separate from the generator; judge calibration is a P4 concern.\n")

    header = ("| slice | n | EM | token-F1 | recall@1 | recall@5 | recall@10 | recall@20 |\n"
              "|---|---|---|---|---|---|---|---|")

    A("## Results — held-out test slice (all questions are multi-hop)\n")
    A(header)
    A(_row("**overall (multi-hop target)**", overall))
    A("")
    A(f"- Faithfulness (judged, **advisory only**): {pct(overall['faithful'])}% "
      f"over {overall['n_judged']} answered questions.")
    A(f"- Citation correctness (judged, **advisory only**): {pct(overall['citations_correct'])}%.")
    A("  - _These two are PROVISIONAL and **not gated**: the local judge is non-deterministic "
      "on CPU and uncalibrated (P4 concern). Treat as directional signal, not a recorded target._")
    A(f"- Over-abstention on answerable test questions: {pct(overall['over_abstain'])}% "
      f"(lower is better; these questions have supporting evidence in the corpus).\n")

    A("### Breakdown by hop_type\n")
    A(header)
    for t in sorted(by_type):
        A(_row(t, by_type[t]))
    A("")

    A("### Breakdown by hop_count\n")
    A(header)
    for c in sorted(by_count):
        A(_row(f"{c}-hop", by_count[c]))
    A("")

    bc = by_type.get("bridge_comparison")
    A("## The number to beat\n")
    A("2WikiMultiHopQA is wholly multi-hop, so the **headline multi-hop target is the "
      "overall test slice**:\n")
    A(f"- **answer token-F1 = {pct(overall['f1'])}%**, **EM = {pct(overall['em'])}%**, "
      f"**support recall@5 = {pct(overall['recall@5'])}%**.\n")
    if bc:
        A("The hardest subset is **4-hop `bridge_comparison`** — called out separately as the "
          "stress target for the graph:\n")
        A(f"- bridge_comparison: **answer token-F1 = {pct(bc['f1'])}%**, "
          f"**EM = {pct(bc['em'])}%**, **support recall@5 = {pct(bc['recall@5'])}%** "
          f"(n={bc['n']}).\n")

    A("## Abstention (no-knowledge set)\n")
    A(f"- Abstention rate: **{pct(nk['abstain_rate'])}%** over {nk['n']} questions whose "
      f"gold support is absent from the corpus (higher is better; 100% = always correctly abstains).\n")

    A("## Reproduce\n")
    A("```bash\njust build-corpus   # deterministic given the pinned seed\n"
      "just baseline       # hybrid retrieval + generation over the test slice\n"
      "just eval           # rewrites this file\n```\n")
    A("Notes: hop_type maps the dataset's `inference` → `bridge`. The gated, recorded "
      "targets — answer EM/token-F1 and support recall@k — are fully deterministic and "
      "recompute exactly from the stored scored run (`python -m kgrag.eval.verify_repro`). "
      "Generation uses temperature 0 with a fixed seed. Faithfulness/citation are advisory "
      "only: the local judge is provisional and non-deterministic on CPU, so its "
      "reproducibility is intentionally NOT gated (judge calibration is a P4 concern).")

    config.ROOT.joinpath("BASELINE.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    run_judge = "--no-judge" not in sys.argv
    stats = json.loads((config.PROCESSED / "corpus_stats.json").read_text())
    corpus = corpus_io.load_corpus()
    by_id = {c["chunk_id"]: c for c in corpus}

    test = corpus_io.load_jsonl(TEST_RUN)
    test = score_test(test, by_id, run_judge=run_judge)
    with open(SCORED_PATH, "w", encoding="utf-8") as f:
        for r in test:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    overall = aggregate(test)
    by_type = group_by(test, "hop_type")
    by_count = group_by(test, "hop_count")

    nk_records = corpus_io.load_jsonl(NK_RUN)
    nk = {
        "n": len(nk_records),
        "abstain_rate": _mean([float(r["abstained"]) for r in nk_records]),
    }

    write_baseline_md(stats, overall, by_type, by_count, nk)
    print("\n=== overall test ===")
    print(json.dumps({k: (pct(v) if isinstance(v, float) else v)
                      for k, v in overall.items()}, indent=2))
    print(f"no-knowledge abstention: {pct(nk['abstain_rate'])}%  (n={nk['n']})")
    print("BASELINE.md written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
