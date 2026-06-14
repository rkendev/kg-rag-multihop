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
import re
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

# The extractor prompt is the EXACT P1 prompt (verified F1 0.613, with father/mother/director/
# award-received already in the correct direction). The two P1 bugs — `performer` reversed and
# `nominated for` folding the award into the relation string — are fixed DETERMINISTICALLY in
# ``canonicalize_triples`` below, not via prompt rules: a 7B model degrades when the prompt grows
# (longer rule/example blocks measurably worsened compliance and broke previously-correct
# relations), whereas the post-process is reliable and reproducible.
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

# --- Deterministic schema fixes for the two P1 bugs (applied after the LLM emits triples) ---

# Relations whose canonical direction is (work, relation, person/org). GLiNER types decide when a
# triple is reversed; only person/org<->work flips are touched, so correctly-oriented triples and
# unrelated relations (father, spouse, ...) are never altered.
_WORK_SUBJECT_RELATIONS = {
    "performer", "director", "producer", "composer", "screenwriter", "production company",
}
_WORK_TYPES = {"creative work", "film", "album", "song", "book"}
_PERSONORG_TYPES = {"person", "organization", "company"}
# "nominated for Best Actress...", "award nominated for X", "won the X Award", "received X"
_AWARD_PREFIX = re.compile(
    r"^(?:award\s+)?(nominated for|award received|won|received|awarded)\b[\s:]*",
    re.IGNORECASE,
)
_NOMINATED = re.compile(r"\bnominated\b", re.IGNORECASE)


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
    """Best-effort char span + GLiNER score + entity label/type for a triple element.

    The label feeds type-blocked entity resolution in P2; ``None`` when the surface is a
    literal/date taken verbatim from the text rather than a detected GLiNER span.
    """
    n = surface.strip().lower()
    for e in ents:
        if e["text"].strip().lower() == n:
            return [e["start"], e["end"]], round(float(e["score"]), 4), e["label"]
    for e in ents:  # containment fallback (e.g. "Genoa" within "Genoa, Italy")
        if n and (n in e["text"].lower() or e["text"].lower() in n):
            return [e["start"], e["end"]], round(float(e["score"]), 4), e["label"]
    i = text.lower().find(n)
    return ([i, i + len(surface)] if i >= 0 else None), None, None


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


def unfold_award(relation: str, obj: str) -> tuple[str, str]:
    """Bug-2 fix: split an award folded into the relation string into a bare relation + award.

    "nominated for Best Actress in a Leading Role" / obj="Deborah Kerr"
        -> ("nominated for", "Best Actress in a Leading Role")
    "award nominated for" / obj="Grammy Award"   -> ("nominated for", "Grammy Award")   [unchanged obj]
    "award received" / obj="GMA Dove Award"       -> ("award received", "GMA Dove Award") [unchanged]
    A non-award relation is returned untouched.
    """
    m = _AWARD_PREFIX.match(relation.strip())
    if not m:
        return relation, obj
    canonical = "nominated for" if _NOMINATED.search(relation) else "award received"
    trailing = relation.strip()[m.end():].strip(" :,-")
    # The folded award lives in the relation tail; if present it is the true object.
    return canonical, (trailing if trailing else obj)


def load_ner():
    """Load the GLiNER model once (heavy import; only when actually extracting)."""
    from gliner import GLiNER

    print(f"[extract] loading GLiNER {config.GLINER_MODEL} ...", flush=True)
    ner = GLiNER.from_pretrained(config.GLINER_MODEL)
    print("[extract] GLiNER ready", flush=True)
    return ner


def extract_chunk_records(ner, cid: str, text: str) -> list[dict]:
    """Extract all predicted triples for one chunk, with full provenance + confidence.

    Reused by the gold re-verify (Step 0) and the resumable full-corpus batch (Step 1) so
    both runs use the identical extractor. ``subj_type``/``obj_type`` carry the GLiNER label
    for type-blocked entity resolution; ``low_confidence`` flags triples below the threshold.
    """
    ents = extract_entities(ner, text)
    triples = extract_relations(text, ents)
    records = []
    for t in triples:
        subj, rel, obj = str(t["subject"]), str(t["relation"]), str(t["object"])
        # Bug-2: unfold an award baked into the relation string into a bare relation + award object.
        rel, obj = unfold_award(rel, obj)
        ss, sscore, stype = _entity_provenance(subj, ents, text)
        os_, oscore, otype = _entity_provenance(obj, ents, text)
        # Bug-1: orient work<->person/org relations so the WORK is the subject. Only flips a
        # person/org-subject + work-object triple; correctly-oriented and unrelated triples
        # (father, spouse, place of birth, ...) are left exactly as the model emitted them.
        rel_norm = rel.strip().lower()
        if rel_norm in _WORK_SUBJECT_RELATIONS and stype in _PERSONORG_TYPES and otype in _WORK_TYPES:
            subj, obj = obj, subj
            ss, os_ = os_, ss
            sscore, oscore = oscore, sscore
            stype, otype = otype, stype
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
            "subj_type": stype,
            "obj_type": otype,
            "llm_confidence": conf,
            "low_confidence": (conf is not None and conf < config.LOW_CONFIDENCE),
            "models": {"ner": config.GLINER_MODEL, "relations": config.EXTRACT_MODEL},
        })
    return records, len(ents)


def run_extraction(chunk_ids: list[str], out_path, *, ner=None) -> int:
    """Extract from the given chunks and write all records to ``out_path`` (one JSON/line).

    Used for the gold re-verify; the full-corpus batch streams/checkpoints separately (see
    ``kgrag.graph.extract_corpus``).
    """
    ner = ner or load_ner()
    corpus = {c["chunk_id"]: c for c in corpus_io.load_corpus()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for i, cid in enumerate(chunk_ids, 1):
        recs, n_ents = extract_chunk_records(ner, cid, corpus[cid]["text"])
        print(f"[extract] {i}/{len(chunk_ids)} {cid}: {n_ents} entities -> {len(recs)} triples", flush=True)
        records.extend(recs)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[extract] wrote {len(records)} predicted triples -> {out_path}", flush=True)
    return 0


def main() -> int:
    # Re-verify the FIXED extractor on the frozen 8-paragraph gold (Step 0). Writes to the
    # post-fix path; the frozen P1 predictions.jsonl is left untouched.
    return run_extraction(load_gold_chunk_ids(), config.EXTRACTION_PRED_POSTFIX_PATH)


if __name__ == "__main__":
    sys.exit(main())
