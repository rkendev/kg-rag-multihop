# DEFENSIBILITY_REGROW.md — re-testing P3 on a graph that is NOT answer-enriched

**Status: COMPLETE (2026-06-17).** Result in `THESIS_RESULT_v2.md`: the multi-hop win largely
collapses on the realistic graph — overall token-F1 +8.8 (v1) → −0.2 (v2); recall@5 +14.7 → +8.2;
`compositional` F1 +4.9 → −5.1. Merge precision recovered 0.821 → 0.903 via 4 surgical fixes
(connectivity held). Fairness git-diff empty; `verify_kgrag` PASS. Nothing committed. This file is the
handoff/state record. It does **not** change P3 generation logic — scope + resolution-bugfixes only.

## Why

The committed P3 win (`THESIS_RESULT.md`, +8.8 overall F1, guardrail held) was measured on a graph
overlay scoped to the **246 gold-support chunks** of the 100 test questions — an *answer-enriched*
graph: the only paragraphs in the graph are the ones that support the gold answers. That biases the
graph leg in its own favour. This regrow re-tests the thesis on a graph scoped to the **full context
pools** of the same 100 test questions (gold support **+** their distractor paragraphs as they appear
in the frozen corpus), so the graph leg must contend with the same distractors as the flat leg. The
FAISS/vector index stays over the full 3,350-chunk corpus throughout (unchanged) — only the graph
overlay scope changed.

## Scope: 246 → 851 (done, reproducible)

- Generator: `src/kgrag/graph/make_graph_scope.py` (`just graph-scope full`). Maps each test
  question's context paragraphs to `chunk_id` via the **same** `(source_title, para_text)` key the
  corpus build used — byte-identical mapping.
- Result: **851 chunks** (1000 context paragraphs pooled, 0 unmapped). Of these, **246 are
  gold-support** — exactly the old overlay, and a verified **subset** of the 851, so the existing
  extraction checkpoint stays valid (no re-extraction of the 246).
- `data/processed/extraction/graph_corpus_ids.txt` now holds the 851 ids; the old 246 set is
  preserved at `graph_corpus_ids_v1_goldsupport.txt`.

## Step 2 — extraction (resumable, per-chunk fault-isolated)

- Command: `just extract-corpus` (single background pass; **no** auto-restart wrapper).
- Keeps already-extracted chunks. Same FIXED qwen2.5:7b extractor as P2.
- **Resilience fix (root cause of the earlier stall):** a single Ollama call can exceed its 600s
  read timeout and previously crashed the whole multi-hour run; an auto-restart loop then reloaded
  GLiNER each cycle and churned memory into swap. Fixed in `extract_corpus.py`: each chunk's
  extraction is wrapped in try/except — on any exception the chunk is recorded as
  **attempted-with-0-triples**, its id + error appended to `timed_out_chunk_ids.txt`, and the loop
  continues. The validated extractor in `extract_triples.py` is **unchanged** (the catch is in the
  driver only). One stalled chunk no longer kills the run, so no restart wrapper is needed.
- **Resume bookkeeping:** done-set = chunk_ids in `predictions_corpus.jsonl` ∪ chunk_ids in the
  `attempted_chunk_ids.txt` sidecar. The sidecar records every attempted chunk regardless of triple
  count, so legitimately-0-triple chunks (e.g. `c00403`, `c00778`, `c01726`) and timed-out chunks
  are skipped on resume instead of re-run every restart.
- Check progress: `tail data/processed/extract_corpus_v2_pass2.log`. Resume: re-run `just extract-corpus`.
- Pathological/timed-out chunks skipped (documented): see `timed_out_chunk_ids.txt`
  (≤ a handful out of 851; acceptable per operator).

## ⚠ KEY FINDING — merge-precision gate FAILS on the full-context graph

Extraction completed (851/851; one chunk, `c03234`, timed out and was skipped — see
`timed_out_chunk_ids.txt`). Resolution with the **SAME** unchanged rules produced 5,425 entities /
140 merged clusters. The merge-precision audit (full census in
`data/processed/resolution/merge_audit_v2.txt`) finds:

- **25 clear over-merges / 140 → precision ≈ 0.821** (v1 was 53/57 = 0.930). **GATE FAILS (<0.90).**
- **Root cause (the actual defensibility result):** the 2WikiMultiHopQA *distractor* paragraphs —
  absent from the answer-enriched v1 overlay — are deliberately full of same-name / same-family /
  same-dynasty entities (Württemberg/Anhalt-Dessau/Nassau/Bourbon-Two-Sicilies royals, Al Thani
  patronymics, 8 different "Robert Gordon"s, the St Leger family, …). The unchanged string-only
  resolution over-merges them via the "single added trailing token = middle name, merge for PERSON"
  subset branch and a relational-guard leak ("granddaughter of X" fuses with X). **The gold-only v1
  graph masked this entirely** — its chunks happened to hold mostly distinct entities.

Over-merging is the gate's *unrecoverable* failure (a fused node poisons every traversal through
it), so building + testing the thesis on this graph would confound the distractor effect with
resolution error.

**Operator chose: surgically tighten the two named culprits + re-audit + connectivity re-check**
(see memory `regrow-resolution-decision`). Implemented in `resolve_entities.py`:
(a) the single-added-token PERSON subset merge now requires an INTERIOR token (leading/trailing
blocked); (b) the relational guard gained mother/father/parent/grand* terms. Re-resolve → 110
clusters / 5,478 entities.

**Result (full census in `merge_audit_v2.txt`):**
- Merge precision **0.821 → 0.855** (16/110 clear over-merges). **Still < 0.90 — gate still fails.**
  The residual over-merges come from mechanisms OUTSIDE the two authorized culprits: honorific
  equivalence + patronymic set-collapse (Al Thani, Sir/Duke/Countess royals), interior single
  tokens that are actually distinguishing (Methuen/Drusus/Fitzwilliam/Pauline), and stopword-
  article collapse (The/A works). Clearing 0.90 needs tightening these too, which is broader than
  the two-culprit scope.
- **Connectivity HELD** (the under-merge risk did not materialize): bridge-entity coverage
  **0.907** (v1 0.897), chain-entity recall **0.828** (v1 0.807). Precision, not recall, is the blocker.

**RESOLVED — operator chose (A): two more micro-fixes (final round).**
(c) patronymic/honorific set-collapse → `merge_ok` now requires the same leading (non-honorific)
content token; (d) WORK-article → `the`/`a`/`an` removed from STOPWORDS. Re-resolve → 103 clusters /
5,493 entities. **Merge precision 0.821 → 0.855 → 0.903 (10/103 clear over-merges) — GATE PASSES.**
Connectivity HELD throughout (bridge coverage 0.907 vs v1 0.897; chain recall 0.828 vs 0.807). The
residual 10 are the irreducible **string-only ceiling** (interior middle-names like "Paul Cobb
Methuen"; same-name-different-person honorific cases) — a proper fix needs embedding/attribute or LLM
disambiguation, out of v1 scope. Full census: `merge_audit_v2.txt`.

Graph rebuilt (`data/graph/v1`, 5,493 entities / 6,292 RELATES / FAISS 3,350; `verify-graph` PASS,
0 orphans). Old 246-overlay KG-RAG runs moved to `runs/kgrag_*_v1_goldsupport.jsonl`. `run_kgrag`
(v2 graph, byte-identical generator) is the multi-hour pass now in flight → then `eval_kgrag` →
restore original `THESIS_RESULT.md` (git) → author `THESIS_RESULT_v2.md` → `verify_kgrag`.

## Remaining steps (AFTER the gate is satisfied) — no code changes needed for build/eval

All downstream scripts read the predictions/corpus generically; the larger input needs no edits.

1. `just resolve` — same conservative, precision-first rules (`resolve_entities.py`, unchanged).
2. `just build-graph` — rebuilds Kùzu + FAISS under `data/graph/v1` (overwrites; the original is
   recoverable from git commit `77a23f9`). FAISS still reuses the full-corpus P0 embeddings → 3,350.
3. `just verify-graph` — deterministic recompute + connectivity report (must show 0 orphans).
4. **Merge-precision gate (≥0.90):** `uv run python -m kgrag.graph.audit_merges` dumps every merged
   cluster (more than the v1 57, since the graph is larger). Hand-judge each for over-merges
   (fusing genuinely different real-world entities); precision = 1 − wrong/total must be **≥0.90**.
   Record the census in `data/processed/resolution/merge_audit.txt` (v1 was 53/57 = 0.930).

5. **⚠ TRAP — re-running the KG-RAG eval.** `run_kgrag.py` resumes by **skipping `id`s already in
   `kgrag_test.jsonl` / `kgrag_no_knowledge.jsonl`**. Those files still hold the **246-overlay** run.
   If you re-run without clearing them, it silently reuses the OLD retrievals. **Before re-running,
   move them aside:**
   ```bash
   cd data/processed/runs
   mv kgrag_test.jsonl          kgrag_test_v1_goldsupport.jsonl
   mv kgrag_no_knowledge.jsonl  kgrag_no_knowledge_v1_goldsupport.jsonl
   mv kgrag_scored_test.jsonl   kgrag_scored_test_v1_goldsupport.jsonl
   ```
   Then: `just kgrag` (graph-fused retrieval + **byte-identical** P0 generator; multi-hour, resumable)
   → `just eval-kgrag` → `just verify-kgrag` (deterministic recompute + git-diff fairness guardrail +
   spot-check). Keep `GEN_TOP_K=5`, `RRF_K=60` (verify_kgrag asserts these).

6. **⚠ TRAP — `eval_kgrag.py` overwrites `THESIS_RESULT.md`.** The original (the 246-overlay P3
   result we compare against) is committed at `77a23f9` — restore it after with
   `git checkout THESIS_RESULT.md`, then write **`THESIS_RESULT_v2.md`** comparing the new KG-RAG
   numbers to BOTH:
   - **P0 flat-RAG** — recomputed from `runs/scored_test.jsonl` (overall F1 22.1, EM 21.0, r@5 74.2;
     comparison guardrail F1 61.9).
   - **246-overlay P3** — from the original `THESIS_RESULT.md` (overall F1 30.9 / +8.8; bridge F1
     +12.2; compositional +4.9; bridge_comparison +21.7; comparison 61.9 held).
   Report honestly whether the multi-hop win **holds, shrinks, or grows** once the graph includes
   distractors. Stay within scope — **no P4 generation changes.**

## Fairness guardrail (unchanged, must stay green)

`git diff` over `generate.py`, `judge.py`, `metrics.py`, `gold/test_ids.txt`, `gold/questions.jsonl`,
`gold/no_knowledge.jsonl` must remain empty. `verify_kgrag.py` asserts this. The only thing that
changed between v1 and v2 is the **graph overlay scope** (246 → 851).
