"""Hybrid retrieval: dense (FAISS/BGE) + lexical (BM25) fused with Reciprocal Rank Fusion.

RRF score for a chunk = sum over legs of 1 / (k + rank), rank 1-based, k = RRF_K.
This baseline is reused later as the hybrid leg of the graph system, so it is kept
self-contained: load once, query many.
"""
from __future__ import annotations

import pickle

import numpy as np

from .. import config
from . import corpus_io, embed


class HybridRetriever:
    def __init__(self) -> None:
        import faiss

        self.corpus = corpus_io.load_corpus()
        self.chunk_ids = [c["chunk_id"] for c in self.corpus]
        self.by_id = {c["chunk_id"]: c for c in self.corpus}
        self.index = faiss.read_index(str(config.FAISS_PATH))
        with open(config.BM25_PATH, "rb") as f:
            payload = pickle.load(f)
        self.bm25 = payload["bm25"]
        assert payload["chunk_ids"] == self.chunk_ids, "bm25/corpus order mismatch"

    # -- individual legs -------------------------------------------------
    def _dense_ranking(self, query: str, pool: int) -> list[str]:
        qv = embed.embed_query(query).reshape(1, -1)
        _, idxs = self.index.search(qv, pool)
        return [self.chunk_ids[i] for i in idxs[0] if i != -1]

    def _bm25_ranking(self, query: str, pool: int) -> list[str]:
        scores = self.bm25.get_scores(corpus_io.tokenize(query))
        top = np.argsort(scores)[::-1][:pool]
        return [self.chunk_ids[i] for i in top]

    # -- fusion ----------------------------------------------------------
    def search(self, query: str, *, pool: int | None = None) -> list[tuple[str, float]]:
        """Return chunk_ids fused by RRF, highest score first."""
        pool = pool or config.RETRIEVE_POOL
        dense = self._dense_ranking(query, pool)
        lexical = self._bm25_ranking(query, pool)
        k = config.RRF_K
        fused: dict[str, float] = {}
        for ranking in (dense, lexical):
            for rank, cid in enumerate(ranking, start=1):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
        return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

    def topk_chunks(self, query: str, k: int) -> list[dict]:
        ranked = self.search(query)[:k]
        return [self.by_id[cid] for cid, _ in ranked]
