"""BGE embeddings (CPU) via sentence-transformers.

bge-*-v1.5 retrieval is asymmetric: the *query* gets an instruction prefix, the
*passage* does not. Embeddings are L2-normalized so a FAISS inner-product index gives
cosine similarity.
"""
from __future__ import annotations

import numpy as np

from .. import config

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(config.EMBED_MODEL, device="cpu")
    return _model


def embed_passages(texts: list[str], *, batch_size: int = 64) -> np.ndarray:
    model = get_model()
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")


def embed_query(query: str) -> np.ndarray:
    model = get_model()
    vec = model.encode(
        config.BGE_QUERY_PREFIX + query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vec.astype("float32")
