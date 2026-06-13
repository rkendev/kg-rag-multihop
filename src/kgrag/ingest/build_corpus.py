"""Freeze the corpus and gold sets from the 2WikiMultiHopQA dev split.

Design
------
* Sample ``N_ANSWERABLE`` dev questions (seeded). The frozen corpus is the pool of
  their context paragraphs — gold supporting + distractors — de-duplicated by
  (title, normalized text). Every answerable question is therefore answerable: its
  gold supporting paragraphs are guaranteed to be in the corpus.
* chunk == one context paragraph. Normalisation is Unicode NFKC + whitespace collapse;
  paragraph boundaries are preserved as chunk boundaries. Provenance on every chunk:
  ``chunk_id``, ``source_title``, ``source_para_idx`` (which distinct paragraph variant
  under that title, 0-based by first encounter).
* ``gold/questions.jsonl`` carries hop_type, hop_count (= # distinct gold supporting
  titles), gold_support_chunk_ids (mapped via exact (title, text)), and gold_triples
  (the dataset's ``evidences``). A 20% test slice, stratified by hop_type, is frozen to
  ``gold/test_ids.txt`` and is never tuned on.
* ``gold/no_knowledge.jsonl`` holds a disjoint set of questions whose gold supporting
  *titles* are entirely absent from the frozen corpus, so the only correct behaviour is
  to abstain. No corpus mutation is needed — the single frozen corpus serves both evals.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
import unicodedata
from collections import Counter, defaultdict

from .. import config

_WS = None  # lazy compiled regex


def normalize(text: str) -> str:
    """NFKC normalise and collapse all whitespace runs to single spaces."""
    global _WS
    if _WS is None:
        import re
        _WS = re.compile(r"\s+")
    text = unicodedata.normalize("NFKC", text)
    return _WS.sub(" ", text).strip()


def para_text(sentences: list[str]) -> str:
    return normalize(" ".join(sentences))


def load_dev() -> list[dict]:
    if not config.DEV_JSON.exists():
        raise SystemExit(f"missing {config.DEV_JSON}; run `python -m kgrag.ingest.download` first")
    with open(config.DEV_JSON, encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    rng = random.Random(config.SEED)
    dev = load_dev()
    order = list(range(len(dev)))
    rng.shuffle(order)

    answerable_idx = order[: config.N_ANSWERABLE]
    answerable = [dev[i] for i in answerable_idx]
    remaining = [dev[i] for i in order[config.N_ANSWERABLE :]]

    # ------------------------------------------------------------------ corpus
    # (title, normalized_text) -> chunk_id ; assigned in stable insertion order.
    key_to_chunk: dict[tuple[str, str], str] = {}
    title_variants: dict[str, list[str]] = defaultdict(list)  # title -> [texts]
    chunks: list[dict] = []
    n_paragraphs = 0

    for ex in answerable:
        for title, sents in ex["context"]:
            n_paragraphs += 1
            text = para_text(sents)
            if not text:
                continue
            key = (title, text)
            if key in key_to_chunk:
                continue
            if text not in title_variants[title]:
                title_variants[title].append(text)
            para_idx = title_variants[title].index(text)
            chunk_id = f"c{len(chunks):05d}"
            key_to_chunk[key] = chunk_id
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source_title": title,
                    "source_para_idx": para_idx,
                    "text": text,
                }
            )

    if len(chunks) > config.CHUNK_CEILING:
        raise SystemExit(
            f"corpus has {len(chunks)} chunks > ceiling {config.CHUNK_CEILING}; "
            f"reduce N_ANSWERABLE in config.py and rebuild."
        )

    corpus_titles = set(title_variants.keys())

    # ---------------------------------------------------------------- gold qs
    def map_support(ex: dict) -> list[str]:
        support_titles = {t for t, _ in ex["supporting_facts"]}
        ids: list[str] = []
        for title, sents in ex["context"]:
            if title in support_titles:
                cid = key_to_chunk.get((title, para_text(sents)))
                if cid is not None and cid not in ids:
                    ids.append(cid)
        return ids

    gold: list[dict] = []
    for ex in answerable:
        hop_type = config.HOP_TYPE_MAP[ex["type"]]
        support_ids = map_support(ex)
        hop_count = len({t for t, _ in ex["supporting_facts"]})
        gold.append(
            {
                "id": ex["_id"],
                "question": ex["question"],
                "answer": ex["answer"],
                "hop_type": hop_type,
                "hop_count": hop_count,
                "gold_support_chunk_ids": support_ids,
                "gold_triples": ex["evidences"],
            }
        )

    # Sanity: every answerable question must have its gold support present.
    missing = [g["id"] for g in gold if not g["gold_support_chunk_ids"]]
    if missing:
        raise SystemExit(f"{len(missing)} answerable questions have no mapped gold support; abort")

    # ----------------------------------------------------------- test slice
    # 20% held-out, stratified by hop_type, seeded.
    by_type: dict[str, list[str]] = defaultdict(list)
    for g in gold:
        by_type[g["hop_type"]].append(g["id"])
    test_ids: set[str] = set()
    split_rng = random.Random(config.SEED + 1)
    for htype, ids in sorted(by_type.items()):
        ids_sorted = sorted(ids)
        split_rng.shuffle(ids_sorted)
        n_test = round(len(ids_sorted) * config.TEST_FRAC)
        test_ids.update(ids_sorted[:n_test])
    for g in gold:
        g["split"] = "test" if g["id"] in test_ids else "dev"

    # -------------------------------------------------------- no-knowledge
    # Disjoint questions whose gold supporting titles are all absent from corpus.
    no_knowledge: list[dict] = []
    for ex in remaining:
        support_titles = {t for t, _ in ex["supporting_facts"]}
        if support_titles & corpus_titles:
            continue  # some evidence about these entities exists in corpus -> skip
        no_knowledge.append(
            {
                "id": ex["_id"],
                "question": ex["question"],
                "answer": ex["answer"],
                "hop_type": config.HOP_TYPE_MAP[ex["type"]],
                "hop_count": len(support_titles),
                "absent_support_titles": sorted(support_titles),
                "gold_triples": ex["evidences"],
            }
        )
        if len(no_knowledge) >= config.N_NO_KNOWLEDGE:
            break

    # --------------------------------------------------------------- write
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    config.GOLD.mkdir(parents=True, exist_ok=True)

    with open(config.CORPUS_PATH, "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")
    corpus_hash = hashlib.sha256(config.CORPUS_PATH.read_bytes()).hexdigest()

    with open(config.QUESTIONS_PATH, "w", encoding="utf-8") as f:
        for g in gold:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    with open(config.NO_KNOWLEDGE_PATH, "w", encoding="utf-8") as f:
        for q in no_knowledge:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    with open(config.TEST_IDS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(test_ids)) + "\n")

    # token estimate: whitespace words * 1.3
    n_words = sum(len(ch["text"].split()) for ch in chunks)
    token_est = int(n_words * 1.3)

    stats = {
        "n_answerable_questions": len(gold),
        "n_no_knowledge_questions": len(no_knowledge),
        "n_paragraphs_pooled": n_paragraphs,
        "n_chunks": len(chunks),
        "chunk_ceiling": config.CHUNK_CEILING,
        "n_words": n_words,
        "token_estimate": token_est,
        "corpus_sha256": corpus_hash,
        "seed": config.SEED,
        "embed_model": config.EMBED_MODEL,
        "test_frac": config.TEST_FRAC,
        "n_test": len(test_ids),
        "n_dev": len(gold) - len(test_ids),
        "hop_type_counts": dict(Counter(g["hop_type"] for g in gold)),
        "hop_type_counts_test": dict(
            Counter(g["hop_type"] for g in gold if g["split"] == "test")
        ),
        "hop_count_counts": dict(Counter(g["hop_count"] for g in gold)),
    }
    with open(config.PROCESSED / "corpus_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
