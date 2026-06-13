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
