"""Build and persist the hybrid retrieval index over the frozen corpus.

Two legs:
* dense  — BGE passage embeddings in a FAISS flat inner-product index (cosine).
* lexical — rank_bm25 BM25Okapi over the same chunks.

Artifacts (all regenerable, gitignored): embeddings.npy, faiss.index, bm25.pkl.
The chunk order is the corpus order, so FAISS row i and BM25 doc i both map to
``corpus[i]["chunk_id"]``.
"""
from __future__ import annotations

import pickle
import sys

import numpy as np

from .. import config
from . import corpus_io, embed


def main() -> int:
    corpus = corpus_io.load_corpus()
    texts = [c["text"] for c in corpus]
    print(f"corpus: {len(corpus)} chunks")

    # dense leg
    print(f"embedding with {config.EMBED_MODEL} (CPU)...")
    vecs = embed.embed_passages(texts)
    np.save(config.EMB_PATH, vecs)

    import faiss

    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, str(config.FAISS_PATH))
    print(f"faiss flat IP index: {index.ntotal} vectors, dim {vecs.shape[1]}")

    # lexical leg
    from rank_bm25 import BM25Okapi

    tokenized = [corpus_io.tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    with open(config.BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": [c["chunk_id"] for c in corpus]}, f)
    print(f"bm25 index: {len(tokenized)} docs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
