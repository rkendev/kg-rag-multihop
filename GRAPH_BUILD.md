# GRAPH_BUILD.md — Phase P2 knowledge-graph build

P2 builds the graph the thesis will later be tested on (P3): it fixes the two P1 extractor bugs,
extracts triples from the corpus with the fixed extractor, resolves entities into canonical nodes,
and populates an embedded Kùzu graph + a FAISS vector index with provenance on every edge. **P2 does
not test the thesis** (graph vs flat RAG) — that is P3. P2's exit gate is **entity-merge precision**,
because over-merging is the unrecoverable failure: a wrongly fused entity poisons every multi-hop
traversal downstream and nothing in P3+ recovers it.

All numbers below recompute deterministically from stored artifacts (`just verify-graph`); no model
is re-run to "prove" anything.

## Step 0 — extractor bug fixes, re-verified on the frozen gold

Re-ran the **fixed** extractor on the frozen 8-paragraph / 97-triple gold and re-scored with the
**unchanged** matcher (`just extraction-eval-postfix`). The frozen P1 `predictions.jsonl` /
`EXTRACTION_BASELINE.md` were left intact; post-fix predictions are in
`data/processed/extraction/predictions_gold_postfix.jsonl`.

| metric | P1 | P2 post-fix | Step-0 gate |
|---|---|---|---|
| overall triple **F1** | 0.613 | **0.742** | ≥ 0.613 ✓ |
| `performer` recall | 0.00 | **0.80** | recovered ✓ |
| core-relation **F1** | 0.903 | **0.968** | ≥ 0.85 ✓ |

Both bugs were fixed **deterministically in post-processing**, not via prompt rules — a 7B model's
compliance measurably degrades as the prompt grows (adding rule/example blocks dropped F1 to 0.54 and
broke previously-correct relations such as `father`/`director`). The prompt was reverted to the exact
validated P1 form; `extract_triples.py` then applies:

- **`performer` direction** — type-based orientation: a work↔person/org relation (`performer`,
  `director`, `producer`, `composer`, `screenwriter`, `production company`) is flipped so the work is
  the subject; all other relations are left untouched.
- **award folding** — `unfold_award` splits an award baked into the relation string into a bare
  relation (`nominated for` / `award received`) plus the award as the object.

## Step 1 — corpus extraction (resumable, checkpointed)

The full 3,350-chunk run measured at ~90–145 s/chunk (qwen2.5:7b on CPU, 6 cores shared) → ~84 h,
which is prohibitive for an overnight build. **Decision (operator-approved):** scope the graph corpus
to the **246 unique support chunks of the 100 held-out test questions** (`graph_corpus_ids.txt`), keep
the validated qwen2.5:7b extractor (no model swap), and keep the **FAISS/vector index over the full
3,350-chunk corpus** so the P3 KG-vs-flat comparison stays fair (only the graph overlay is scoped).

- Chunks extracted: **246/246** (one, `c01726`, yielded 0 extractable triples). Wall-clock **9.1 h**.
- Triples persisted: **2,481** (1 flagged low-confidence) → `data/processed/extraction/predictions_corpus.jsonl`,
  each with `source_chunk_id`, char spans, GLiNER entity scores + types, LLM confidence, model tags.
- This LLM pass ran **once**; all downstream stats recompute from the stored file.

**Limitation (documented):** the graph overlay covers the test-support subset, not the full corpus, so
it cannot contribute to questions whose support lies outside those 246 chunks. The corpus run is
resumable/checkpointed (`just extract-corpus`), so the graph can be grown to the dev-support set
(~1,200 chunks) or the full corpus later without redoing completed chunks.

## Step 2 — entity resolution + the merge-precision gate

Mentions are keyed by the alias-resolved normalized surface **plus a disambiguating parenthetical tag**
(`Halloween (1978)` vs `(2018)`; `Gangs of New York (film)` vs `(book)`). Identical keys exact-merge;
non-identical keys merge only under conservative, **precision-first** structural rules.

> **Deviation from the P2 spec, and why.** The spec merge rule was `rapidfuzz token_set_ratio ≥ 92` OR
> `bge-small cosine ≥ 0.86`. On the real corpus this over-merged badly (~71 % precision): `token_set_ratio`
> scores any subset 100, so a single shared token ("Charles") chained every Charles into one node, and
> embedding cosine fused distinct-but-similar entities (different awards/films). Per the gate's own
> escalation rule ("a failing merge-precision gate means tightening resolution thresholds and
> re-resolving"), the similarity test was replaced with structural rules (`merge_ok`): no
> compound/relational/cross-numeral fusion; honorific- and initial-aware name variants; a single added
> middle-name/epithet for PERSONs only; a distinct-entity-suffix guard (`X` vs `X Museum`); and a tight
> `token_sort_ratio ≥ 93` typo-catch for short names. Type-blocking prevents cross-type fusion. This is
> precision-first by design: under-merging (reported as recall) is acceptable; over-merging is not.

- Unique mention keys: **2,292** → canonical entities: **2,253**.
- Merged clusters (≥ 2 surface forms): **57**.

**Merge-precision gate — full census of all 57 merged clusters** (hand-judged, recorded in
`data/processed/resolution/merge_audit.txt`):

- Over-merges (fuse genuinely different real-world entities): **4**
  1. `Paul Methuen` — fuses three different politicians (b. 1752 / 1723 / 1672).
  2. `Drusus Caesar` + `Drusus Julius Caesar` — Germanicus's son vs Tiberius's son.
  3. `Edward Fitzwilliam` + `Edward Francis Fitzwilliam` — father (b. 1788) vs son (b. 1824).
  4. `Elisabeth of Leuchtenberg` — two different women (b. 1568 vs 1537).
- **Merge precision = 53/57 = 0.930 ≥ 0.90 → GATE PASSED.**

All four residuals are the same hard class: distinct people sharing a name, separable only by birth
years embedded in prose — beyond what string-only resolution can decide. They are PERSON nodes; none
is a high-degree hub, so traversal blast-radius is small.

## Step 3 — Kùzu graph + FAISS index

Embedded Kùzu (`data/graph/v1/`, `current` → `v1` rollback symlink). FAISS reuses the **pinned
`BAAI/bge-small-en-v1.5`** P0 corpus embeddings (embedder not re-run), keyed by `chunk_id`.

| node / edge | count |
|---|---|
| `Entity{entity_id, canonical_name, type, aliases[]}` | 2,253 |
| `Chunk{chunk_id, doc_id, text, source_title}` | 3,350 |
| `Source{doc_id}` | 3,340 |
| `(Entity)-[:RELATES {relation, confidence, source_chunk_id}]->(Entity)` | 2,481 |
| `(Entity)-[:MENTIONED_IN]->(Chunk)` | 2,760 |
| `(Chunk)-[:FROM_SOURCE]->(Source)` | 3,350 |
| aliases recorded | 2,315 |
| FAISS vectors × dim | 3,350 × 384 |

Provenance (`source_chunk_id`, `confidence`) is on **every** RELATES edge. Integrity: **0 orphan
edges**, 0 unresolved/dropped triples — every RELATES edge resolves to a real chunk and two real
canonical entities. (Cypher reserves `FROM`, so the chunk→source edge table is named `FROM_SOURCE`;
semantics unchanged.)

## Step 4 — deterministic verify + connectivity

`just verify-graph` recomputes all stats from the stored resolution output + frozen corpus (pure
Python, no LLM), twice, and checks them against the build manifest.

- **Deterministic verify: PASS** — recompute identical across runs, matches manifest, **0 orphan edges**.
- **Connectivity report** (guards the *under-merge* failure that high precision can hide):
  - (a) **Bridging entities** (appear in ≥ 2 chunks and/or RELATES-sourced from ≥ 2 chunks): **254**.
  - (b) **Bridge-entity resolution coverage** — of the gold bridge/compositional/bridge_comparison
    **test** questions, the fraction whose bridge entity resolved to a single connecting node:
    **0.897** (87/97 bridge entities across 79 multi-hop test questions).
  - (c) **Entity-resolution recall proxy** (gold multi-hop chain entities present as a node):
    **0.807** (239/296) — reported, not gated; the precision-first resolution trades some recall, and
    most misses are bare single-token names or comma-compound surfaces (e.g. "Charles, Duke of Durazzo").

## Verification performed before sign-off

- Re-ran `just verify-graph` from the built state: stats reproduce exactly; 0 orphans.
- Hand-inspected all 57 merged clusters (not just a 30–50 sample): precision 0.930.
- Spot-checked 10 RELATES edges → each traces to a source chunk that states the relation (2 carry an
  off-vocab relation string, e.g. "directed by" — a known extraction-quality limitation; the edges are
  true and provenanced).
- Confirmed via `git status` that the fairness-guardrail files are untouched: `gold/questions.jsonl`,
  `gold/no_knowledge.jsonl`, `gold/test_ids.txt`, `src/kgrag/baseline/generate.py`,
  `src/kgrag/eval/judge.py`, plus the frozen P1 `predictions.jsonl` / `EXTRACTION_BASELINE.md` and the
  corpus / extraction gold.

## One-screen summary

```
post-fix extractor : F1 0.742 (P1 0.613) | performer recall 0.80 | core-rel F1 0.968   [Step-0 PASS]
corpus extraction  : 246/246 test-support chunks, 2,481 triples, 9.1h, full provenance
entity resolution  : 2,292 keys -> 2,253 canonical entities, 57 merged clusters
merge precision    : 53/57 = 0.930  (>= 0.90 GATE PASSED; 4 same-name-different-person residuals)
connectivity       : 254 bridging entities | bridge coverage 0.897 | ER recall proxy 0.807
graph + index      : 2,253 entities / 3,350 chunks / 2,481 RELATES (0 orphan) / FAISS 3,350x384
deterministic verify: PASS (recompute identical, matches manifest, 0 orphans)
embedding model     : BAAI/bge-small-en-v1.5 (pinned) | extractor qwen2.5:7b-instruct (pinned)
```

## Reproduce (deterministic — no LLM/embedder re-run)

```bash
just resolve          # rebuild canonical entities + resolved triples from stored predictions
just build-graph      # rebuild Kùzu graph + FAISS index under data/graph/v1 (+ current symlink)
just verify-graph     # recompute stats + connectivity; confirm identical + 0 orphans
```
