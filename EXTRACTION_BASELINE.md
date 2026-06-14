# EXTRACTION_BASELINE.md — Phase P1 triple-extraction gate

Local triple extraction (GLiNER entities + Ollama `qwen2.5:7b-instruct` relations, temperature 0) scored against a frozen hand-annotated gold with an entity-resolution-aware matcher. **This is a gate, not a deliverable:** it decides whether a graph built from this extractor could beat the P0 flat-RAG baseline.

## Setup (all frozen before scoring — see git history)

- Gold: **97 hand-annotated triples** across **8 paragraphs** of the frozen P0 corpus, exhaustively labelled per `gold/ANNOTATION_GUIDELINE.md`.
- Entities: `urchade/gliner_medium-v2.1` (open zero-shot schema, CPU).
- Relations: Ollama `qwen2.5:7b-instruct`, JSON mode, temperature 0, fixed seed; predictions persisted once to `data/processed/extraction/predictions.jsonl`.
- Matcher: `src/kgrag/eval/triple_matcher.py` + frozen `gold/alias_table.json`, `gold/relation_synonyms.json`. Validated at **20/20** agreement on 20 hand-judged pairs (`gold/matcher_validation.jsonl`).

## Overall (micro-averaged over all triples)

| metric | value |
|---|---|
| predicted triples | 89 |
| gold triples | 97 |
| true positives | 57 |
| precision | **0.640** |
| recall | **0.588** |
| **F1** | **0.613** |

## Core-relation aggregate (the bridge-carrying relations)

These five relations carry the multi-hop bridges the graph must traverse; near the 0.60 line they are weighted over the micro-average, which is dominated by high-frequency descriptive relations (occupation/performer).

- Relations: `director`, `father`, `spouse`, `date of birth`, `place of birth`
- Core gold triples: 15; TP: 14
- Core precision / recall / **F1**: 0.875 / 0.933 / **0.903**

## Per-relation breakdown

| relation | gold | pred | TP | precision | recall | F1 |
|---|---|---|---|---|---|---|
| occupation | 21 | 18 | 15 | 0.83 | 0.71 | 0.77 |
| performer | 10 | 8 | 0 | 0.00 | 0.00 | 0.00 |
| nominated for | 7 | 0 | 0 | 0.00 | 0.00 | 0.00 |
| country of citizenship | 5 | 3 | 3 | 1.00 | 0.60 | 0.75 |
| date of birth ⭐ | 5 | 5 | 5 | 1.00 | 1.00 | 1.00 |
| date of death | 5 | 5 | 5 | 1.00 | 1.00 | 1.00 |
| place of birth ⭐ | 5 | 5 | 5 | 1.00 | 1.00 | 1.00 |
| cause of death | 3 | 2 | 2 | 1.00 | 0.67 | 0.80 |
| founded by | 3 | 1 | 1 | 1.00 | 0.33 | 0.50 |
| place of death | 3 | 3 | 3 | 1.00 | 1.00 | 1.00 |
| publication date | 3 | 5 | 3 | 0.60 | 1.00 | 0.75 |
| based on | 2 | 2 | 2 | 1.00 | 1.00 | 1.00 |
| birth name | 2 | 1 | 1 | 1.00 | 0.50 | 0.67 |
| director ⭐ | 2 | 2 | 1 | 0.50 | 0.50 | 0.50 |
| genre | 2 | 1 | 1 | 1.00 | 0.50 | 0.67 |
| inception | 2 | 1 | 1 | 1.00 | 0.50 | 0.67 |
| screenwriter | 2 | 2 | 1 | 0.50 | 0.50 | 0.50 |
| sibling | 2 | 0 | 0 | 0.00 | 0.00 | 0.00 |
| spouse ⭐ | 2 | 3 | 2 | 0.67 | 1.00 | 0.80 |
| acquired by | 1 | 1 | 1 | 1.00 | 1.00 | 1.00 |
| award received | 1 | 2 | 0 | 0.00 | 0.00 | 0.00 |
| country | 1 | 0 | 0 | 0.00 | 0.00 | 0.00 |
| country of origin | 1 | 2 | 0 | 0.00 | 0.00 | 0.00 |
| employer | 1 | 0 | 0 | 0.00 | 0.00 | 0.00 |
| father ⭐ | 1 | 1 | 1 | 1.00 | 1.00 | 1.00 |
| member of | 1 | 1 | 1 | 1.00 | 1.00 | 1.00 |
| mother | 1 | 1 | 1 | 1.00 | 1.00 | 1.00 |
| place of burial | 1 | 1 | 1 | 1.00 | 1.00 | 1.00 |
| producer | 1 | 0 | 0 | 0.00 | 0.00 | 0.00 |
| production company | 1 | 4 | 1 | 0.25 | 1.00 | 0.40 |
| appeared in films between | 0 | 1 | 0 | 0.00 | 0.00 | 0.00 |
| award nominated for | 0 | 1 | 0 | 0.00 | 0.00 | 0.00 |
| cowrote | 0 | 1 | 0 | 0.00 | 0.00 | 0.00 |
| directed films between | 0 | 1 | 0 | 0.00 | 0.00 | 0.00 |
| directed films starring | 0 | 1 | 0 | 0.00 | 0.00 | 0.00 |
| nominated for best actress in a leading role | 0 | 1 | 0 | 0.00 | 0.00 | 0.00 |
| nominated for best actress in a supporting role | 0 | 1 | 0 | 0.00 | 0.00 | 0.00 |
| nominated for best director | 0 | 1 | 0 | 0.00 | 0.00 | 0.00 |
| worked with | 0 | 1 | 0 | 0.00 | 0.00 | 0.00 |

⭐ = core (bridge-carrying) relation.

## Weak predicates (carry into P2)

- `performer` — F1 0.00 (gold 10, pred 8, TP 0).
- `nominated for` — F1 0.00 (gold 7, pred 0, TP 0).
- `sibling` — F1 0.00 (gold 2, pred 0, TP 0).
- `award received` — F1 0.00 (gold 1, pred 2, TP 0).
- `country` — F1 0.00 (gold 1, pred 0, TP 0).
- `country of origin` — F1 0.00 (gold 1, pred 2, TP 0).
- `employer` — F1 0.00 (gold 1, pred 0, TP 0).
- `producer` — F1 0.00 (gold 1, pred 0, TP 0).
- `production company` — F1 0.40 (gold 1, pred 4, TP 1).

## Observed error modes (carry into P2)

Inspecting the predictions surfaced two systematic extractor errors that the matcher correctly scored as misses (direction and schema are enforced — gold-vs-gold matches at F1 1.0):

- **`performer` direction reversal.** The extractor emitted `(Deborah Kerr, performer, The Sundowners)` instead of the gold direction `(The Sundowners, performer, actor)`. All 10 gold `performer` triples were missed for this reason alone — a prompt/schema fix (pin subject = the work) should recover them in P2.
- **Award/nomination folding.** Nominations came back with the award baked into the relation string (relation `"nominated for Best Actress in a Leading Role"`, object = the nominee) rather than relation `nominated for`, object = the award. `nominated for` and `award received` therefore score ~0; both need a constrained relation schema in P2.

## Gate decision

**GO** — Overall F1 in [0.55,0.65) (near the line); core-relation F1 = 0.903 >= 0.60, so the relations that actually carry multi-hop bridges are reliable. Immediate GO per the approved marginal policy. Small gold (n=97) — treat as provisional and re-measure at corpus scale in P2.

### Decision rule (pre-committed, before scoring)
- F1 ≥ 0.65 → GO; F1 < 0.55 → NO-GO; 0.55 ≤ F1 < 0.65 → decide on core-relation F1 (GO iff ≥ 0.60). A pass in [0.60,0.65] is an immediate GO (no extra tuning round).

### Small-sample caveat
- The gold is 97 triples over 8 paragraphs. Per-relation numbers with single-digit support are indicative only; several relations have gold support of 1–3. These F1s are a go/no-go signal for whether to *start* P2, not a calibrated corpus-scale accuracy. P2 must re-measure extraction quality at scale before relying on weak predicates.

## Reproduce (deterministic — no LLM re-run)
```bash
just extraction-eval     # rescore stored predictions, rewrite this file
just verify-extraction   # confirm F1 recomputes exactly from stored predictions
```
