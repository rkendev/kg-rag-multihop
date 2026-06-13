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
```

## License

Code: see repository. Dataset (2WikiMultiHopQA) is Apache-2.0; its license is retained
at `data/LICENSE`. The raw dataset is not committed (`data/raw/` is gitignored); the
derived frozen corpus and gold sets are committed under the dataset's permissive terms.
