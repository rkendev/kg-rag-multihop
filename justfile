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

# P1 gate: score stored predictions against the hand gold and (re)write EXTRACTION_BASELINE.md.
extraction-eval:
    uv run python -m kgrag.eval.extraction_eval

# P1 gate: confirm triple F1 recomputes exactly from stored predictions (no LLM re-run).
verify-extraction:
    uv run python -m kgrag.eval.verify_extraction

# --- P2: build the knowledge graph ---

# Step 0: re-run the FIXED extractor on the frozen 8-paragraph gold (post-fix predictions).
extract-postfix:
    PYTHONUNBUFFERED=1 uv run python -u -m kgrag.eval.extract_triples

# Step 0: score the post-fix gold predictions and print the no-regression gate verdict.
extraction-eval-postfix:
    uv run python -m kgrag.eval.extraction_eval_postfix

# Step 1a: regenerate the graph-corpus scope (gold=246 answer-only overlay | full=851 test-context union).
graph-scope scope="full":
    uv run python -m kgrag.graph.make_graph_scope --scope {{scope}} --write

# Step 1: full-corpus extraction ONCE — resumable/checkpointed. Run as a quiet background job.
extract-corpus:
    PYTHONUNBUFFERED=1 uv run python -u -m kgrag.graph.extract_corpus

# Step 2: resolve surface forms into canonical entities (type-blocked, conservative thresholds).
resolve:
    uv run python -m kgrag.graph.resolve_entities

# Step 3: build the Kùzu graph + FAISS index from stored predictions + resolution output.
build-graph:
    uv run python -m kgrag.graph.build_graph

# Step 4: deterministic graph-stats verify + connectivity report (pure Python, no LLM).
verify-graph:
    uv run python -m kgrag.graph.verify_graph

# --- P3: KG-RAG multi-hop retrieval (the thesis test) ---

# Run KG-RAG end to end (graph-fused retrieval + UNCHANGED P0 generator) over the test slice
# + no-knowledge set. Resumable/checkpointed; the generation pass runs ONCE. Background job.
kgrag:
    PYTHONUNBUFFERED=1 uv run python -u -m kgrag.graph.run_kgrag

# Score the KG-RAG run with the SAME judge/metrics as P0 and write THESIS_RESULT.md.
eval-kgrag:
    uv run python -m kgrag.graph.eval_kgrag

# P3 gate: deterministic metric recompute + fairness git-diff guardrail + 5-question spot-check.
verify-kgrag:
    uv run python -m kgrag.graph.verify_kgrag

# Remove regenerable artifacts (keeps frozen corpus + gold sets).
clean-runs:
    rm -rf data/processed/runs data/processed/embeddings.npy data/processed/faiss.index data/processed/bm25.pkl
