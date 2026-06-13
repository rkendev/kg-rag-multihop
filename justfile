# kg-rag-multihop — Phase P0 tasks
# Requires: uv, and a running Ollama with the pinned model tags pulled.

# Sync env, confirm python pin and required ollama model tags are present.
bootstrap:
    uv sync
    uv run python -c "import faiss, rank_bm25, sentence_transformers, datasets; print('deps ok')"
    uv run python -m kgrag.ingest.check_ollama

# Download 2WikiMultiHopQA (Apache-2.0), then freeze corpus + gold sets.
build-corpus:
    uv run python -m kgrag.ingest.download
    uv run python -m kgrag.ingest.build_corpus

# Build the hybrid index and run retrieval + generation over the test slice.
baseline:
    uv run python -m kgrag.baseline.index
    uv run python -m kgrag.baseline.run_baseline

# Score the baseline run and (re)write BASELINE.md.
eval:
    uv run python -m kgrag.eval.run_eval

# Remove regenerable artifacts (keeps frozen corpus + gold sets).
clean-runs:
    rm -rf data/processed/runs data/processed/embeddings.npy data/processed/faiss.index data/processed/bm25.pkl
