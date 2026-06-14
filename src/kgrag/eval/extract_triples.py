"""P1 local triple extraction — runs ONCE, persists predictions, never re-run to score.

Pipeline per gold paragraph:
  1. GLiNER (open zero-shot schema) extracts entity spans + confidence.
  2. Ollama qwen2.5:7b-instruct (JSON mode, temperature 0) emits (subject, relation,
     object, confidence) triples linking those entity spans, steered to the frozen
     canonical-relation vocabulary.

Every predicted triple is stored with full provenance and confidence (source chunk id,
subject/object char spans, GLiNER span scores, LLM relation confidence, model tags) so
accuracy can be debugged later from metadata captured on the first run. Output:
``data/processed/extraction/predictions.jsonl`` (one triple per line).

The qwen step is non-deterministic in wall-clock but pinned to temp 0 + fixed seed; it is
run a single time. Scoring/reproducibility recompute from the stored file, never re-running
the model (see ``extraction_eval`` / ``verify_extraction``).
"""
from __future__ import annotations

import json
import sys

from .. import config
from ..baseline import corpus_io
from ..ollama_client import generate as ollama_generate

# Canonical relation vocabulary handed to the extractor as its target schema (the same
# closed+open set the gold was annotated against — frozen before this run).
RELATION_VOCAB = [
    "director", "performer", "composer", "publisher", "producer", "screenwriter",
    "based on", "publication date", "country of origin", "production company", "genre",
    "date of birth", "date of death", "place of birth", "place of death",
    "place of burial", "cause of death", "father", "mother", "spouse", "sibling", "child",
    "country of citizenship", "educated at", "employer", "award received", "nominated for",
    "founded by", "inception", "country", "member of", "acquired by", "birth name",
    "occupation",
]

SYSTEM = (
    "You are an information-extraction system. From a single paragraph you output factual "
    "triples (subject, relation, object) that are explicitly stated or unambiguously "
    "entailed by the paragraph text. Use only the provided entity mentions as subjects and "
    "objects. Use only relations from the allowed list. Do not invent facts or use outside "
    "knowledge. Respond with JSON only."
)

PROMPT_TEMPLATE = """Paragraph:
{text}

Entities detected in this paragraph (use these as subjects/objects; dates/values may also be used verbatim from the text):
{entities}

Allowed relations (use the closest one; subject is the entity the fact is about):
{relations}

Extract every true triple. For each, give a confidence in [0,1].
Respond with JSON exactly of the form:
{{"triples": [{{"subject": "...", "relation": "...", "object": "...", "confidence": 0.0}}]}}"""


def load_gold_chunk_ids() -> list[str]:
    gold = corpus_io.load_jsonl(config.EXTRACTION_GOLD_PATH)
    seen = []
    for g in gold:
        if g["chunk_id"] not in seen:
            seen.append(g["chunk_id"])
    return seen


def extract_entities(model, text: str) -> list[dict]:
    ents = model.predict_entities(text, config.ENTITY_LABELS, threshold=config.GLINER_THRESHOLD)
    # dedupe identical surface+span, keep highest score
    out = {}
    for e in ents:
        key = (e["start"], e["end"])
        if key not in out or e["score"] > out[key]["score"]:
            out[key] = e
    return sorted(out.values(), key=lambda e: e["start"])


def _entity_provenance(surface: str, ents: list[dict], text: str):
    """Best-effort char span + GLiNER score for a triple element."""
    n = surface.strip().lower()
    for e in ents:
        if e["text"].strip().lower() == n:
            return [e["start"], e["end"]], round(float(e["score"]), 4)
    for e in ents:  # containment fallback (e.g. "Genoa" within "Genoa, Italy")
        if n and (n in e["text"].lower() or e["text"].lower() in n):
            return [e["start"], e["end"]], round(float(e["score"]), 4)
    i = text.lower().find(n)
    return ([i, i + len(surface)] if i >= 0 else None), None


def extract_relations(text: str, ents: list[dict]) -> list[dict]:
    ent_lines = "\n".join(f"- \"{e['text']}\" ({e['label']})" for e in ents) or "- (none)"
    prompt = PROMPT_TEMPLATE.format(
        text=text, entities=ent_lines, relations=", ".join(RELATION_VOCAB)
    )
    raw = ollama_generate(
        config.EXTRACT_MODEL,
        prompt,
        system=SYSTEM,
        temperature=config.GEN_TEMPERATURE,
        num_predict=1500,
        format="json",
    )
    try:
        data = json.loads(raw)
        triples = data.get("triples", []) if isinstance(data, dict) else []
    except json.JSONDecodeError:
        triples = []
    return [t for t in triples if isinstance(t, dict) and t.get("subject") and t.get("relation") and t.get("object")]


def main() -> int:
    from gliner import GLiNER  # heavy import; only when actually extracting

    chunk_ids = load_gold_chunk_ids()
    corpus = {c["chunk_id"]: c for c in corpus_io.load_corpus()}
    print(f"[extract] loading GLiNER {config.GLINER_MODEL} ...", flush=True)
    ner = GLiNER.from_pretrained(config.GLINER_MODEL)
    print(f"[extract] GLiNER ready; extracting from {len(chunk_ids)} gold paragraphs", flush=True)

    config.EXTRACTION_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for i, cid in enumerate(chunk_ids, 1):
        text = corpus[cid]["text"]
        ents = extract_entities(ner, text)
        triples = extract_relations(text, ents)
        print(f"[extract] {i}/{len(chunk_ids)} {cid}: {len(ents)} entities -> {len(triples)} triples", flush=True)
        for t in triples:
            subj, rel, obj = str(t["subject"]), str(t["relation"]), str(t["object"])
            ss, sscore = _entity_provenance(subj, ents, text)
            os_, oscore = _entity_provenance(obj, ents, text)
            try:
                conf = float(t.get("confidence"))
            except (TypeError, ValueError):
                conf = None
            records.append({
                "chunk_id": cid,
                "subject": subj,
                "relation": rel,
                "object": obj,
                "subj_span": ss,
                "obj_span": os_,
                "subj_gliner_score": sscore,
                "obj_gliner_score": oscore,
                "llm_confidence": conf,
                "models": {"ner": config.GLINER_MODEL, "relations": config.EXTRACT_MODEL},
            })

    with open(config.EXTRACTION_PRED_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[extract] wrote {len(records)} predicted triples -> {config.EXTRACTION_PRED_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
