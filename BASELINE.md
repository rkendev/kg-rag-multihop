# BASELINE.md — Phase P0 flat-RAG baseline

Recorded flat-RAG baseline over the frozen 2WikiMultiHopQA corpus. **No graph.** This is the number later phases must beat to justify the knowledge graph.

## Corpus (frozen)

- Chunks: **3350** (ceiling 9000) — paragraph-level, deduped by (title, normalized text).
- Paragraphs pooled: 5000 → 3350 unique chunks.
- Token estimate: ~310,671 (whitespace words × 1.3; 238,978 words).
- Corpus SHA-256: `e08a154ed32a49b3d47634450d3628de564d2c1bc9aa2f8a28e17440dbaec666`
- Provenance on every chunk: `chunk_id`, `source_title`, `source_para_idx`.
- Embedding model: `BAAI/bge-small-en-v1.5`; seed `20260613`.

## Gold sets (frozen)

- Answerable questions: 500 (400 dev / 100 held-out test, 20% stratified by hop_type — test slice never tuned on).
- No-knowledge (abstention) questions: 40 — gold supporting titles absent from the corpus, so abstention is the only correct answer.
- hop_type distribution (all answerable): {'bridge': 70, 'bridge_comparison': 113, 'compositional': 212, 'comparison': 105}.
- hop_count distribution: {'2': 387, '4': 113} (hop_count = number of distinct gold supporting titles; bridge_comparison = 4).

## Baseline configuration

- Retrieval: BGE dense (`BAAI/bge-small-en-v1.5`, FAISS flat IP) + BM25 (`rank_bm25`), fused with Reciprocal Rank Fusion (k=60).
- Generation: Ollama `qwen2.5:7b-instruct`, temperature 0, top-5 context, answers with `[chunk_id]` citations; abstains with `INSUFFICIENT`.
- Judge (faithfulness/citation): Ollama `llama3.1:8b-instruct-q4_K_M`, temperature 0 — **PROVISIONAL**, separate from the generator; judge calibration is a P4 concern.

## Results — held-out test slice (all questions are multi-hop)

| slice | n | EM | token-F1 | recall@1 | recall@5 | recall@10 | recall@20 |
|---|---|---|---|---|---|---|---|
| **overall (multi-hop target)** | 100 | 21.0 | 22.1 | 37.2 | 74.2 | 79.5 | 81.5 |

- Faithfulness (judged, **advisory only**): 15.2% over 46 answered questions.
- Citation correctness (judged, **advisory only**): 15.2%.
  - _These two are PROVISIONAL and **not gated**: the local judge is non-deterministic on CPU and uncalibrated (P4 concern). Treat as directional signal, not a recorded target._
- Over-abstention on answerable test questions: 54.0% (lower is better; these questions have supporting evidence in the corpus).

### Breakdown by hop_type

| slice | n | EM | token-F1 | recall@1 | recall@5 | recall@10 | recall@20 |
|---|---|---|---|---|---|---|---|
| bridge | 14 | 0.0 | 7.2 | 35.7 | 75.0 | 78.6 | 82.1 |
| bridge_comparison | 23 | 21.7 | 21.7 | 22.8 | 55.4 | 60.9 | 63.0 |
| comparison | 21 | 61.9 | 61.9 | 47.6 | 100.0 | 100.0 | 100.0 |
| compositional | 42 | 7.1 | 7.5 | 40.5 | 71.4 | 79.8 | 82.1 |

### Breakdown by hop_count

| slice | n | EM | token-F1 | recall@1 | recall@5 | recall@10 | recall@20 |
|---|---|---|---|---|---|---|---|
| 2-hop | 77 | 20.8 | 22.3 | 41.6 | 79.9 | 85.1 | 87.0 |
| 4-hop | 23 | 21.7 | 21.7 | 22.8 | 55.4 | 60.9 | 63.0 |

## The number to beat

2WikiMultiHopQA is wholly multi-hop, so the **headline multi-hop target is the overall test slice**:

- **answer token-F1 = 22.1%**, **EM = 21.0%**, **support recall@5 = 74.2%**.

The hardest subset is **4-hop `bridge_comparison`** — called out separately as the stress target for the graph:

- bridge_comparison: **answer token-F1 = 21.7%**, **EM = 21.7%**, **support recall@5 = 55.4%** (n=23).

## Abstention (no-knowledge set)

- Abstention rate: **97.5%** over 40 questions whose gold support is absent from the corpus (higher is better; 100% = always correctly abstains).

## Reproduce

```bash
just build-corpus   # deterministic given the pinned seed
just baseline       # hybrid retrieval + generation over the test slice
just eval           # rewrites this file
```

Notes: hop_type maps the dataset's `inference` → `bridge`. The gated, recorded targets — answer EM/token-F1 and support recall@k — are fully deterministic and recompute exactly from the stored scored run (`python -m kgrag.eval.verify_repro`). Generation uses temperature 0 with a fixed seed. Faithfulness/citation are advisory only: the local judge is provisional and non-deterministic on CPU, so its reproducibility is intentionally NOT gated (judge calibration is a P4 concern).
