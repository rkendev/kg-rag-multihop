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

**P1 scope (not started):** entity/relation extraction over the corpus, entity resolution,
a graph index, and retrieval-traversal that assembles multi-hop evidence — justified only
if it beats the targets above, especially on the 4-hop subset.

## License

Code: see repository. Dataset (2WikiMultiHopQA) is Apache-2.0; its license is retained
at `data/LICENSE`. The raw dataset is not committed (`data/raw/` is gitignored); the
derived frozen corpus and gold sets are committed under the dataset's permissive terms.
