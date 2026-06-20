# THESIS_RESULT.md — Phase P3: KG-RAG vs flat RAG on multi-hop QA

The P3 thesis test. The **only** difference from the P0 flat-RAG baseline is retrieval: graph traversal over the P2 Kùzu knowledge graph is RRF-fused with the unchanged P0 hybrid retrieval, then the **byte-identical** P0 generator and judge score the frozen 100-question multi-hop test slice. Baseline numbers below are recomputed from `runs/scored_test.jsonl` (the P0 control), so every delta is apples-to-apples.

## Headline

- **THESIS HOLDS** on this corpus + judge.
- Overall test token-F1: 22.1% → 30.9% (Δ +8.8 pts), EM 21.0% → 28.0% (Δ +7.0), support recall@5 74.2% → 89.0% (Δ +14.7).
- Single-hop guardrail (`comparison` F1): 61.9% → 61.9% — preserved.

## Delta by hop_type (the headline breakdown)

| slice | n | EM (base→kg, Δ) | token-F1 (base→kg, Δ) | recall@5 (base→kg, Δ) |
|---|---|---|---|---|
| bridge | 14 | 0.0→7.1 (+7.1) | 7.2→19.4 (+12.2) | 75.0→89.3 (+14.3) |
| compositional | 42 | 7.1→9.5 (+2.4) | 7.5→12.4 (+4.9) | 71.4→95.2 (+23.8) |
| bridge_comparison | 23 | 21.7→43.4 (+21.7) | 21.7→43.4 (+21.7) | 55.4→71.7 (+16.3) |
| comparison | 21 | 61.9→61.9 (+0.0) | 61.9→61.9 (+0.0) | 100.0→95.2 (-4.8) |

Overall + by hop_count:
| slice | n | EM (base→kg, Δ) | token-F1 (base→kg, Δ) | recall@5 (base→kg, Δ) |
|---|---|---|---|---|
| overall | 100 | 21.0→28.0 (+7.0) | 22.1→30.9 (+8.8) | 74.2→89.0 (+14.7) |

## Multi-hop subset (the thesis target)

The multi-hop subset is `bridge` + `compositional` + `bridge_comparison` (the slices whose answer requires a second-hop chunk the query entity does not name). Per-slice improvement (Δ in points):

- `bridge`: F1 +12.2, recall@5 +14.3 ✓ improved
- `bridge_comparison`: F1 +21.7, recall@5 +16.3 ✓ improved
- `compositional`: F1 +4.9, recall@5 +23.8 ✓ improved

## Abstention

- No-knowledge set (correct answer = abstain): baseline 97.5% → KG-RAG 100.0% abstention.
- Over-abstention on answerable test (lower is better): baseline 54.0% → KG-RAG 47.0%.

## Verdict + diagnosis

KG-RAG improves the multi-hop subset without collapsing the single-hop guardrail: the graph leg surfaces second-hop bridging chunks the flat retriever misses, which is exactly the recall gap P0 left open. See the spot-check below for concrete cases where the retrieved set changed and a bridging chunk was recovered.

## Caveats (honest scope)

- **Graph overlay scope.** Per the P2 operator-approved decision, the graph was extracted over the 246 test-support chunks (a full-corpus qwen extraction was ~84h). The flat leg retrieves over the full 3,350-chunk corpus, but the graph leg can only surface chunks within that 246-chunk overlay. This is favourable to the graph on exactly the test questions and is the main threat to external validity; growing the overlay to the full corpus is future work, not P3.
- **Judge advisory only.** Faithfulness/citation come from a provisional CPU judge and are not gated, identical to P0.
- **Determinism.** EM/F1/recall@k recompute exactly from the stored run (`python -m kgrag.graph.verify_kgrag`); the generator ran once at temp 0.

## Spot-check — the graph genuinely surfaced second-hop chunks

Five multi-hop questions where KG-RAG pulled a gold supporting chunk into the top-5 that the
baseline's top-5 *missed*. In each case the recovered chunk entered via graph traversal from
the linked query entity, not via vector/BM25 (reproduce: `python -m kgrag.graph.verify_kgrag`):

| qid | hop_type | linked seed(s) | recovered gold chunk(s) | baseline top-5 had them? |
|---|---|---|---|---|
| 02b62fce | bridge_comparison | Phalitamsha, Gladiators Seven | c00015, c00018 | no |
| 50e96f4e | compositional | The Idiot Returns | c00065 | no |
| 39befe64 | compositional | Constance of York | c00392 | no |
| c04d7e50 | bridge_comparison | Mokey, When Worlds Collide | c00409 | no |
| e1f928d8 | bridge_comparison | There's Always Vanilla, L'ultimo amante | c00431 | no |

These are the second-hop bridging chunks the flat baseline lacked: the bridge entity (the
director, the father) is *not* named in the question, so vector/BM25 cannot rank its chunk —
but traversing the RELATES edge from the named film/person reaches it. This is the recall
gap P0 left open, closed.

## Guardrail proof (fairness — the only thing that changed is retrieval)

The generator, generation prompt, judge, metrics, and frozen test/no-knowledge/test-id gold
are reused byte-for-byte from P0. `git diff --stat HEAD` over those paths is **empty**, and
they were last modified in the P0 baseline commit (never in P2 or P3):

```
$ git diff --stat HEAD -- src/kgrag/baseline/generate.py src/kgrag/eval/judge.py \
      src/kgrag/eval/metrics.py gold/test_ids.txt gold/questions.jsonl gold/no_knowledge.jsonl
(no output — all six paths unmodified)

$ git log --oneline -1 -- src/kgrag/baseline/generate.py src/kgrag/eval/judge.py src/kgrag/eval/metrics.py
97a73e5 Add flat-RAG baseline and eval
```

Pinned constants unchanged: `GEN_TOP_K = 5`, `RRF_K = 60`. The generation prompt is
byte-identical because the context is still a set of chunk passages — only the chunk *set*
changed. `verify_kgrag.py` asserts all of the above and the deterministic metric recompute
(100 records, 0 mismatches).

