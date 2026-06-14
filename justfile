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

# Confirm the gated, deterministic metrics recompute exactly from the scored run.
verify:
    uv run python -m kgrag.eval.verify_repro

# P1 gate: validate the frozen matcher against hand-judged pairs.
validate-matcher:
    uv run python -m kgrag.eval.validate_matcher

# P1 gate: run local triple extraction ONCE and persist predictions (non-deterministic; do not re-run to score).
extract:
    PYTHONUNBUFFERED=1 uv run python -u -m kgrag.eval.extract_triples

# P1 gate: score stored predictions against the hand gold and (re)write EXTRACTION_BASELINE.md.
extraction-eval:
    uv run python -m kgrag.eval.extraction_eval

# P1 gate: confirm triple F1 recomputes exactly from stored predictions (no LLM re-run).
verify-extraction:
    uv run python -m kgrag.eval.verify_extraction

# Remove regenerable artifacts (keeps frozen corpus + gold sets).
clean-runs:
    rm -rf data/processed/runs data/processed/embeddings.npy data/processed/faiss.index data/processed/bm25.pkl
