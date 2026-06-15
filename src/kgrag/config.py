"""Central, pinned configuration for Phase P0.

Everything that affects reproducibility lives here: model ids/tags, seeds, sample
sizes, retrieval k-values, and on-disk paths. No tuning happens against the held-out
test slice; only the values below (and the dev split) may be adjusted.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
GOLD = ROOT / "gold"

CORPUS_PATH = PROCESSED / "corpus.jsonl"
EMB_PATH = PROCESSED / "embeddings.npy"
FAISS_PATH = PROCESSED / "faiss.index"
BM25_PATH = PROCESSED / "bm25.pkl"
RUNS_DIR = PROCESSED / "runs"

QUESTIONS_PATH = GOLD / "questions.jsonl"
NO_KNOWLEDGE_PATH = GOLD / "no_knowledge.jsonl"
TEST_IDS_PATH = GOLD / "test_ids.txt"

# --- P1 extraction gate ---
EXTRACTION_GOLD_PATH = GOLD / "extraction_gold.jsonl"
EXTRACTION_DIR = PROCESSED / "extraction"
EXTRACTION_PRED_PATH = EXTRACTION_DIR / "predictions.jsonl"  # FROZEN P1 gate artifact — never overwrite

# --- P2 graph build ---
# Post-fix re-verification of the extractor on the frozen 8-paragraph gold (Step 0).
EXTRACTION_PRED_POSTFIX_PATH = EXTRACTION_DIR / "predictions_gold_postfix.jsonl"
# Graph-corpus extraction output (Step 1), written checkpointed/resumable per chunk.
EXTRACTION_PRED_CORPUS_PATH = EXTRACTION_DIR / "predictions_corpus.jsonl"
# Optional scope: if this file exists, the graph corpus is restricted to these chunk_ids (one
# per line). v1 scopes to the 100 test questions' support chunks (full qwen run is ~84h; this
# bounds it to an overnight run). The FAISS/vector index stays over the full corpus regardless.
GRAPH_CORPUS_IDS_PATH = EXTRACTION_DIR / "graph_corpus_ids.txt"
RESOLUTION_DIR = PROCESSED / "resolution"
RESOLUTION_ENTITIES_PATH = RESOLUTION_DIR / "entities.jsonl"
RESOLUTION_TRIPLES_PATH = RESOLUTION_DIR / "triples_resolved.jsonl"
GRAPH_DIR = DATA / "graph"            # versioned graph + index live under here (v1/, current symlink)

# Entity-resolution thresholds (conservative; over-merging is the gated failure). On top of
# these, a structural guard (>=2 shared tokens, both multi-token, comparable length, no
# conflicting regnal numerals) gates every fuzzy/embedding merge — raw token_set_ratio alone
# over-merges badly (it scores any subset 100, so "Charles" pivots all the Charleses together).
RESOLVE_FUZZ_RATIO = 92               # rapidfuzz floor (legacy; superseded by structural rules)
RESOLVE_COSINE = 0.90                 # bge-small cosine floor (legacy; embedding path removed)
RESOLVE_TYPO_RATIO = 93               # token_sort_ratio floor for the short-name typo catch
# Low-confidence triple flag (a threshold + log line, NOT a quarantine pipeline).
LOW_CONFIDENCE = 0.5

DEV_JSON = RAW / "dev.json"  # extracted from the official data.zip

# ---------------------------------------------------------------------------
# Data source (official, Apache-2.0)
# ---------------------------------------------------------------------------
# Alab-NII/2wikimultihop — full dataset zip (dropbox direct-download form).
DATA_ZIP_URL = "https://www.dropbox.com/s/npidmtadreo6df2/data.zip?dl=1"
LICENSE_URL = "https://raw.githubusercontent.com/Alab-NII/2wikimultihop/main/LICENSE"

# ---------------------------------------------------------------------------
# Sampling / corpus
# ---------------------------------------------------------------------------
SEED = 20260613
N_ANSWERABLE = 500          # sampled dev questions for the answerable gold set
N_NO_KNOWLEDGE = 40         # disjoint reserved questions for the abstention set
CHUNK_CEILING = 9000        # hard ceiling on frozen corpus chunk count
TEST_FRAC = 0.20            # held-out, stratified by hop_type, never tuned on

# dataset `type` -> our hop_type taxonomy
HOP_TYPE_MAP = {
    "inference": "bridge",
    "comparison": "comparison",
    "compositional": "compositional",
    "bridge_comparison": "bridge_comparison",
}

# ---------------------------------------------------------------------------
# Models (pinned)
# ---------------------------------------------------------------------------
EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # 384-d, CPU
# bge-*-v1.5 retrieval uses an asymmetric query instruction; passages get no prefix.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

OLLAMA_HOST = "http://localhost:11434"
GEN_MODEL = "qwen2.5:7b-instruct"            # generator
JUDGE_MODEL = "llama3.1:8b-instruct-q4_K_M"  # separate judge (no self-grading)
GEN_NUM_PREDICT = 256
GEN_TEMPERATURE = 0.0
JUDGE_TEMPERATURE = 0.0

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
RRF_K = 60                       # reciprocal-rank-fusion constant
RECALL_KS = [1, 5, 10, 20]       # support recall@k reported at these k
GEN_TOP_K = 5                    # chunks passed to the generator
RETRIEVE_POOL = 50               # candidates pulled from each leg before fusion

# Sentinel the generator must emit when context cannot support an answer.
ABSTAIN_TOKEN = "INSUFFICIENT"

# ---------------------------------------------------------------------------
# P1 — local triple extraction (GLiNER entities + Ollama qwen relations)
# ---------------------------------------------------------------------------
GLINER_MODEL = "urchade/gliner_medium-v2.1"  # open-schema NER, CPU, cached after first pull
GLINER_THRESHOLD = 0.4                        # span confidence floor (recorded per entity)
# Open entity schema (GLiNER zero-shot labels). Kept broad so relation extraction has
# spans to connect; the relation set is what the gate actually measures.
ENTITY_LABELS = [
    "person", "organization", "company", "location", "country", "city",
    "creative work", "film", "album", "song", "book",
    "date", "role", "occupation", "award", "nationality",
]
EXTRACT_MODEL = GEN_MODEL  # qwen2.5:7b-instruct, JSON mode, temp 0 for relations
