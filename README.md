# kg-rag-multihop

Knowledge-Graph RAG for multi-hop question answering, evaluated on
[2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop) (Apache-2.0).

This repository is built in phases. **Phase P0 (this milestone) contains no graph code.**
P0 produces the three things a graph must later justify itself against:

1. a **frozen corpus** (paragraph chunks with provenance, under a 9,000-chunk ceiling),
2. a **gold multi-hop question set** (hop-type tags, gold supporting chunks, reasoning
   triples) plus a held-out abstention set, and
3. a **measured flat-RAG baseline** (hybrid BGE + BM25 retrieval fused with Reciprocal
   Rank Fusion, local generation) with metrics broken down by hop type.

The recorded baseline numbers live in [`BASELINE.md`](BASELINE.md). The graph work in
later phases is only worthwhile if it beats the multi-hop target written there.

## Constraints

- **Local-first, $0.** Embeddings, retrieval, and generation all run locally
  (Ollama + `sentence-transformers`). No paid API calls.
- **Reproducible.** uv-managed env with a committed lockfile; pinned embedding model id
  and Ollama model tags; fixed seeds.

## Stack

| Component   | Choice                                        |
|-------------|-----------------------------------------------|
| Embeddings  | `BAAI/bge-small-en-v1.5` (CPU, FAISS flat IP) |
| Lexical     | `rank_bm25` (BM25Okapi)                        |
| Fusion      | Reciprocal Rank Fusion (k=60)                 |
| Generator   | Ollama `qwen2.5:7b-instruct` (temp 0)         |
| Judge       | Ollama `llama3.1:8b-instruct-q4_K_M` (temp 0, provisional) |

## Layout

```
src/kgrag/
  config.py            # pinned ids, seeds, k-values, paths
  ingest/              # download + corpus/gold construction
  baseline/            # embed, index, hybrid retrieve, generate
  eval/                # metrics, local judge, end-to-end eval
data/        raw/ (gitignored)  processed/ (frozen corpus)  LICENSE (dataset)
gold/        questions.jsonl  no_knowledge.jsonl  test_ids.txt
```

## Reproduce

Requires [uv](https://docs.astral.sh/uv/) and a running Ollama with
`qwen2.5:7b-instruct` and `llama3.1:8b-instruct-q4_K_M` pulled.

```bash
just bootstrap       # uv sync + verify python pin and ollama tags
just build-corpus    # download 2Wiki, freeze corpus + gold sets
just baseline        # build hybrid index, run retrieval + generation on the test slice
just eval            # score and write BASELINE.md
just verify          # confirm the gated deterministic metrics recompute exactly
```

## P0 → P1 handoff

**Status:** P0 closed. The corpus and gold sets are frozen, and the flat-RAG baseline is
recorded in [`BASELINE.md`](BASELINE.md). No graph code exists yet — by design.

**The number to beat** (held-out test, n=100, all multi-hop):

- Overall: **answer token-F1 22.1**, EM 21.0, **support recall@5 74.2**.
- Hardest subset — 4-hop `bridge_comparison`: **F1 21.7**, **recall@5 55.4** (n=23).

**Where flat RAG breaks (the opening for the graph):**

- The *second hop* is not co-retrieved — recall@5 falls from 79.9 (2-hop) to 55.4 (4-hop),
  because bridge entities (a film's director, a person's parent) sit in paragraphs the
  question text doesn't lexically or semantically reach.
- The generator over-abstains: 54% of answerable test questions get `INSUFFICIENT`, 53/54
  of them *despite* having gold evidence in context — partial single-hop evidence reads as
  insufficient for a multi-hop chain.

**Ground rules carried into P1:**

- The corpus (`data/processed/corpus.jsonl`, SHA in BASELINE.md), `gold/*`, and the
  `gold/test_ids.txt` test slice are **frozen** — do not regenerate or tune against them.
  P1 is measured on the same slice with the same metrics so numbers are comparable.
- This baseline's hybrid retrieval (`src/kgrag/baseline/`) is written to be **reused as the
  hybrid leg** of the graph system; extend, don't replace.
- Faithfulness/citation are advisory-only (provisional judge); only EM/token-F1 and
  support recall@k are gated targets.

## P1 → P2 handoff

**Status:** P1 closed — **GATE PASSED (GO)**. Local triple extraction was measured against a
frozen hand-annotated gold and an entity-resolution-aware matcher; the result is recorded in
[`EXTRACTION_BASELINE.md`](EXTRACTION_BASELINE.md). No graph was built in P1 — by design; P1
only decides whether one is worth building.

**The extraction gate** (hand gold, 97 triples over 8 frozen-corpus paragraphs):

- Overall triple **F1 = 0.613** (precision 0.640, recall 0.588).
- **Core-relation F1 = 0.903** over the bridge-carrying relations (`director`, `father`,
  `spouse`, `date of birth`, `place of birth`).
- Matcher validated at 20/20 agreement on hand-judged pairs; F1 recomputes exactly from the
  stored predictions (`just verify-extraction`).
- Decision: overall F1 landed in the near-the-line band [0.55, 0.65); per the pre-committed
  rule the core-relation F1 (≥ 0.60) decides it → **GO** to P2.

**Carry into P2 (weak predicates / extractor fixes — see EXTRACTION_BASELINE.md):**

- `performer` is systematically **direction-reversed** (actor as subject); pin subject = the
  work in the P2 extraction schema to recover it.
- `nominated for` / `award received` fold the award into the relation string; constrain the
  relation schema.
- Re-measure extraction quality at corpus scale before relying on any single-digit-support
  relation — the gate gold is small (go/no-go signal, not calibrated accuracy).

**Frozen P1 artifacts:** `gold/ANNOTATION_GUIDELINE.md`, `gold/extraction_gold.jsonl`,
`gold/alias_table.json`, `gold/relation_synonyms.json`, `gold/matcher_validation.jsonl`, and
`data/processed/extraction/predictions.jsonl` (the one-time extractor output).

**P2 scope (not started):** full-corpus entity/relation extraction, corpus-scale entity
resolution, a graph index, and retrieval-traversal that assembles multi-hop evidence —
justified only if it beats the P0 targets above, especially on the 4-hop subset.

## P2 → P3 handoff

**Status:** P2 closed — graph built and verified. Full record in
[`GRAPH_BUILD.md`](GRAPH_BUILD.md). P2 ends at a trustworthy graph; it does **not** test the
thesis (graph vs flat RAG) — that is P3.

**Exit gate — entity-merge precision: PASSED.**

- Post-fix extractor (Step-0, same frozen gold + matcher): overall **F1 0.742** (P1 0.613),
  `performer` recall **0.80** (was 0), core-relation **F1 0.968**. Both P1 bugs fixed
  deterministically (type-based direction orientation; `unfold_award`).
- Corpus extraction: **246/246** test-support chunks, **2,481** triples, full provenance +
  confidence, persisted once (`predictions_corpus.jsonl`).
- **Merge precision = 0.930** (53/57 merged clusters correct; full census, gate ≥ 0.90). The 4
  residuals are same-name-different-person cases separable only by birth years in prose.
- Graph: **2,253** entities / **3,350** chunks / **2,481** RELATES edges (0 orphan, provenance on
  every edge) + FAISS **3,350×384** (`bge-small`, pinned). Deterministic verify reproduces exactly.
- Connectivity (under-merge guard): **254** bridging entities; bridge-entity resolution coverage
  **0.897** on multi-hop test questions; ER recall proxy **0.807** (reported, not gated).

**Scope decision to carry into P3 (operator-approved):** the **graph overlay** is built over the
246 test-question support chunks (the full-corpus qwen run is ~84 h; this bounds it to one
overnight run). The **FAISS/vector index is over the full 3,350-chunk corpus**, so the KG-vs-flat
comparison varies only retrieval and stays fair. The corpus extraction is resumable
(`just extract-corpus`), so the overlay can be grown to dev-support (~1,200 chunks) or the full
corpus before P3 if broader graph coverage is wanted.

**Carry into P3 / known limitations:**

- Resolution is precision-first (conservative): ER recall ~0.81 — some same-entity surfaces stay
  split (bare single-token names, comma-compound surfaces). Loosen thresholds only with a fresh
  merge-precision check, never below the 0.90 gate.
- A minority of edges carry off-vocab relation strings (e.g. "directed by"); they are true and
  provenanced but not canonical — a P3 traversal/normalization concern, not a graph-integrity one.
- 4 known same-name-different-person over-merges (see `GRAPH_BUILD.md`); all low-degree PERSON nodes.

**Frozen P2 artifacts:** `data/processed/extraction/predictions_corpus.jsonl` (one-time extractor
output), `data/processed/extraction/graph_corpus_ids.txt`, `data/processed/resolution/*`, and the
built graph under `data/graph/v1/` (`current` → `v1`; the Kùzu/FAISS binaries are gitignored and
rebuilt via `just build-graph`).

**Do NOT start P3 in P2.** No query planner, question entity-linking, traversal, retrieval, RRF,
generation, abstention, reranker, or P0-baseline comparison — all P3+.

## P3 result — KG-RAG vs flat RAG (the thesis test)

**Status:** P3 closed. **Thesis holds.** Full record in
[`THESIS_RESULT.md`](THESIS_RESULT.md). The single thing changed vs P0 is retrieval: graph
traversal over the P2 Kùzu graph is RRF-fused with the unchanged P0 hybrid retrieval; the
**byte-identical** P0 generator + judge score the frozen 100-question test slice.

| slice | n | token-F1 (P0→KG) | support recall@5 (P0→KG) |
|---|---|---|---|
| bridge | 14 | 7.2 → **19.4** (+12.2) | 75.0 → **89.3** (+14.3) |
| compositional | 42 | 7.5 → **12.4** (+4.9) | 71.4 → **95.2** (+23.8) |
| bridge_comparison (4-hop) | 23 | 21.7 → **43.5** (+21.7) | 55.4 → **71.7** (+16.3) |
| comparison (single-hop guardrail) | 21 | 61.9 → **61.9** (+0.0) | 100.0 → 95.2 (−4.8) |
| **overall** | 100 | 22.1 → **30.9** (+8.8) | 74.2 → **89.0** (+14.7) |

**Headline:** overall multi-hop token-F1 **+8.8 pts** (22.1 → 30.9), recall@5 **+14.7**. Every
multi-hop slice improved and each beat its P0 target; the single-hop guardrail held flat.
Bonus: over-abstention on answerable questions 54.0% → 47.0%, no-knowledge abstention 97.5% →
100%. 94/100 questions used the graph leg; 6 fell back to pure hybrid (the expected
under-merged-bridge fallback). Mechanism confirmed by spot-check: the graph surfaces the
second-hop bridging chunk (the bridge entity isn't named in the question, so vector/BM25
can't rank it) — the recall gap P0 left open.

**Pipeline (all deterministic except the single temp-0 generator pass, which runs once):**
relation normalization → GLiNER entity extraction + alias/bge linking → bounded Kùzu
traversal (both directions, depth ≤ 2, frontier budget) → RRF fusion with the P0 hybrid leg
→ unchanged P0 generation → same judge/metrics.

```bash
just kgrag         # graph-fused retrieval + UNCHANGED P0 generator (resumable, one-time)
just eval-kgrag    # score with the SAME judge/metrics; writes THESIS_RESULT.md
just verify-kgrag  # deterministic recompute + fairness git-diff guardrail + spot-check
```

**Fairness guardrail:** `git diff` over the generator, prompt, judge, metrics, and frozen
gold is empty (last touched in the P0 commit); `GEN_TOP_K=5`, `RRF_K=60` unchanged.
**Caveat:** the graph overlay covers the 246 test-support chunks (P2 operator-approved scope),
so the graph leg can only surface chunks within that overlay; growing it to the full corpus is
future work. See `THESIS_RESULT.md`.

## License

Code: see repository. Dataset (2WikiMultiHopQA) is Apache-2.0; its license is retained
at `data/LICENSE`. The raw dataset is not committed (`data/raw/` is gitignored); the
derived frozen corpus and gold sets are committed under the dataset's permissive terms.
