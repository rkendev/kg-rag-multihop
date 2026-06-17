# THESIS_RESULT_v2.md — Defensibility re-test: KG-RAG on a graph that is NOT answer-enriched

This is the **defensibility regrow** of the P3 result. The committed P3 win (`THESIS_RESULT.md`:
overall token-F1 +8.8, multi-hop slices all up, guardrail held) was measured on a graph overlay
scoped to the **246 gold-support chunks** of the 100 test questions — an *answer-enriched* graph:
the only paragraphs in it are the ones that support the gold answers. That is favourable to the
graph leg by construction. This re-test rebuilds the graph over the **full context pool** of the
same 100 test questions — gold support **plus their distractor paragraphs** as they appear in the
frozen corpus (851 chunks) — keeping FAISS over the full 3,350-chunk corpus. **Only the graph
overlay scope changed; the generator, judge, metrics, and frozen gold are byte-identical to P0.**

There are **two** headline findings — one about the graph, one about the thesis.

---

## Headline 1 — the answer-enriched slice hid a resolution-precision failure

`v1`'s entity-resolution merge precision was reported at **0.930**. That number was an **artifact of
the answer-enriched corpus**: the 246 gold-support chunks happen to contain mostly *distinct*
entities. The realistic graph — which includes the same distractor paragraphs the flat retriever
must contend with — is full of the same-name / same-family / same-dynasty entities that
2WikiMultiHopQA deliberately uses as distractors (Württemberg & Anhalt-Dessau & Nassau &
Bourbon-Two-Sicilies & Al Thani royals; eight different "Robert Gordon"s; the St Leger family; …).
On that graph the **same unchanged string-only resolution rules collapse to 0.821 precision** — the
merge-precision gate (≥0.90) **fails**. String-only resolution breaks across *multiple* mechanisms at
once, not one:

| mechanism | example over-merge |
|---|---|
| trailing/leading added token read as a middle name | "Robert Gordon" fused with "Robert Gordon **Pearson**" |
| relational-guard leak | "**granddaughter of** Adib Kheir" fused with Adib Kheir |
| patronymic / honorific set-collapse (word order lost) | "Hamad bin Khalifa" fused inside "**Khalifa bin Hamad** …" |
| article collapse | "**The** Lump of Coal" (story) fused with "**A** Lump of Coal" (album) |
| interior token that is actually distinguishing | "Paul **Cobb** Methuen" (1752) fused with "Paul Methuen" (1723) |

Four **surgical, precision-first bug-fixes** (not tuning) were applied to `resolve_entities.py` —
edge-token vs interior-token discrimination; an extended kinship guard; a same-leading-name
requirement that restores word order to the set-based subset test; and treating articles as
significant. Each is a real correctness fix. Precision recovered in two rounds:

```
merge precision:  0.821 (same v1 rules)  ->  0.855 (+edge-token & relational fixes)
                                          ->  0.903 (+leading-name & article fixes)   GATE PASSES
```

**Crucially, this did NOT come at the cost of recall.** Fixing over-merges risks swinging into
under-merging that *splits* bridge entities and breaks multi-hop a different way, so connectivity was
re-checked every round:

| connectivity metric | v1 (gold-only) | v2 (full-context, post-fix) |
|---|---|---|
| bridge-entity coverage (gold multi-hop) | 0.897 | **0.907** |
| chain-entity ER recall proxy | 0.807 | **0.828** |

**Residual string-only ceiling (documented, out of v1 scope to fix).** 10 of 103 merged clusters
remain wrong — the *irreducible* limit of string matching: interior middle-names that are genuine in
"John **Marcellus** Huston" but distinguishing in "Paul **Cobb** Methuen" / "Drusus **Julius** Caesar"
/ "Edward **Francis** Fitzwilliam" / "Pauline **Therese** of Württemberg"; and same-first-and-last-name
different people separated only by birth-year or title ("Sir" Colin Campbell, "Duke" Ernest I,
"Countess" Elisabeth of Leuchtenberg). String matching **fundamentally cannot** disambiguate these —
a correct fix needs embedding/attribute (birth-year) or LLM disambiguation, which is out of v1 scope.
Full census: `data/processed/resolution/merge_audit_v2.txt`.

---

## Headline 2 — the multi-hop *win* largely COLLAPSES once the graph includes distractors

On the realistic 851-chunk graph, the headline F1 win **does not hold up** — it shrinks to flat.
Three-way comparison, all recomputed from the stored runs against the same P0 control:

| metric | P0 flat | v1 246-overlay (answer-enriched) | **v2 851 (realistic, +distractors)** |
|---|---|---|---|
| overall token-F1 | 22.1 | 30.9  (**+8.8**) | **22.0  (−0.2)** |
| overall EM | 21.0 | 28.0  (+7.0) | **20.0  (−1.0)** |
| overall support recall@5 | 74.2 | 89.0  (+14.7) | **82.5  (+8.2)** |
| comparison F1 (single-hop guardrail) | 61.9 | 61.9  (+0.0) | **57.1  (−4.8)** |

Per multi-hop slice (token-F1, Δ vs P0):

| slice | n | P0 | v1 Δ | **v2 Δ** | recall@5 v1 Δ | **recall@5 v2 Δ** |
|---|---|---|---|---|---|---|
| bridge | 14 | 7.2 | +12.2 | **+5.1** | +14.3 | +14.3 |
| compositional | 42 | 7.5 | +4.9 | **−5.1** | +23.8 | +10.7 |
| bridge_comparison | 23 | 21.7 | +21.7 | **+9.8** | +16.3 | +12.0 |

**What survived, what didn't.**
- **The overall answer-F1 win is gone:** +8.8 → **−0.2** (flat). Essentially *all* of the apparent
  overall F1 gain in v1 was an artifact of the answer-only overlay.
- **Retrieval recall@5 still improves (+8.2)** — about half of v1's +14.7. The graph genuinely still
  surfaces second-hop bridging chunks the flat retriever misses (the spot-check below recovers the
  *same* gold chunks as v1, e.g. `c00018`, `c00065`, `c00392`). The recall mechanism is real.
- **But the recall gain no longer converts to answer quality.** Every multi-hop F1 delta shrank
  (bridge +12.2→+5.1; bridge_comparison +21.7→+9.8), and the **largest slice, `compositional`,
  flipped negative (+4.9 → −5.1)**. On the realistic graph, traversal also pulls *distractor* chunks
  into the top-5 (the distractors are now in the graph too), which compete with — and dilute — the
  recovered bridging chunk in the fixed 5-chunk context the generator sees.
- **The single-hop guardrail slipped** 61.9 → 57.1 (−4.8). Within the automated ±5-pt tolerance
  (`verdict()` still prints "THESIS HOLDS"), but it is no longer cleanly preserved.

**Honest verdict.** *Does the multi-hop win hold, shrink, or grow?* It **shrinks — to near-nothing on
answer quality.** What remains is a *retrieval-recall* advantage (recall@5 +8.2, ~half of v1) that,
on a distractor-inclusive graph, largely fails to translate into EM/F1 gains, with one multi-hop
slice actively regressing. The automated gate calls it a (marginal) hold; the honest reading is that
**the headline P3 multi-hop F1 win was substantially an answer-enrichment artifact and does not
survive a realistic graph.** v1 was optimistic on *both* resolution precision *and* the size of the win.

---

## Diagnosis — why recall improves but F1 does not

The graph leg does its job: it links the query entities and traverses `RELATES` to reach the
second-hop chunk whose bridging entity the question never names. `verify_kgrag` confirms 5 multi-hop
questions where a gold chunk the baseline top-5 *missed* was recovered into the v2 top-5 purely via
traversal (`c00018`, `c00065`, `c00392`, `c00409`, `c00431` — the same recoveries as v1). So recall@5
rises (+8.2). The failure is **downstream of retrieval**: with distractors in the graph, traversal
*also* surfaces plausible-but-wrong neighbour chunks, the RRF fusion promotes them into the top-5
alongside the true bridge, and the generator — seeing a 5-chunk context that is now noisier than the
flat baseline's — does no better (and on `compositional`, worse). The over-merge residual ceiling
compounds this: a fused node (e.g. a wrong "Paul Methuen") routes traversal to the wrong person's
chunks. The lesson: **on an answer-enriched graph, "graph recall" and "answer quality" move together;
on a realistic graph they decouple, and only the recall half survives.**

---

## Fairness guardrail (the only thing that changed is the graph overlay scope)

`verify_kgrag.py` (deterministic, no LLM) asserts all of the below:

```
[1] deterministic recompute: 100 records, 0 mismatches -> PASS
[2] fairness guardrail: protected files unmodified=True, GEN_TOP_K=5 RRF_K=60 -> PASS
[3] spot-check: 5 multi-hop gold chunks recovered into top-5 via graph traversal (PASS)
```

`git diff --stat HEAD` over the generator, generation prompt, judge, metrics, and the frozen
test / no-knowledge / test-id gold is **empty** — those are byte-identical to P0. Between v1 and v2
the changes are confined to graph construction: the overlay scope (246 → 851 chunks,
`make_graph_scope.py`), the per-chunk fault-isolation in extraction, and the four resolution
bug-fixes (`resolve_entities.py`). Pinned constants unchanged (`GEN_TOP_K=5`, `RRF_K=60`).

## Caveats / scope

- **Still test-question-scoped.** The graph overlay is the 851-chunk *full context pool of the 100
  test questions*, not the full 3,350 corpus. It is no longer answer-enriched (distractors included),
  which removes the v1 bias, but a full-corpus graph is still future work.
- **Judge advisory only**, not gated (identical to P0).
- **One generation pass, temp 0.** EM/F1/recall@k recompute exactly from the stored run
  (`python -m kgrag.graph.verify_kgrag`).
- **Resolution residual.** ~10% of merged clusters remain over-merged at the string-only ceiling;
  a stronger ER (embedding/LLM disambiguation) is the next lever and is out of v1 scope.

## Reproduce

```bash
just graph-scope full        # 851-chunk test-context union (make_graph_scope.py)
just extract-corpus          # resumable, per-chunk fault-isolated (851 chunks)
just resolve                 # 4 surgical precision fixes; 103 merged clusters, precision 0.903
uv run python -m kgrag.graph.audit_merges   # hand-census (gate >= 0.90)
just build-graph && just verify-graph        # Kuzu + FAISS(3350); 0 orphans; bridge cov 0.907
# move runs/kgrag_*.jsonl aside, then:
just kgrag && just eval-kgrag && just verify-kgrag
```

## One-screen summary

```
graph scope      : 246 gold-only (v1)  ->  851 full test-context incl. distractors (v2)
merge precision  : 0.930 (v1, artifact) ; 0.821 same-rules on v2 -> 0.903 after 4 surgical fixes
connectivity     : bridge coverage 0.907 (v1 0.897) | chain recall 0.828 (v1 0.807)  HELD
thesis (overall) : token-F1  +8.8 (v1)  ->  -0.2 (v2)   |  recall@5  +14.7 -> +8.2
multi-hop F1 Δ   : bridge +12.2->+5.1 | compositional +4.9->-5.1 | bridge_comp +21.7->+9.8
guardrail F1     : 61.9 held (v1)  ->  57.1 (-4.8, within tolerance) (v2)
verdict          : the multi-hop WIN SHRINKS to a (halved) recall-only effect; the F1 win was
                   substantially an answer-enrichment artifact and does not survive distractors.
fairness         : generator/judge/metrics/gold byte-identical to P0 (git diff empty); verify PASS
```
