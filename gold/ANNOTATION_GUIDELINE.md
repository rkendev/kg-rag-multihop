# Extraction gold — annotation guideline (frozen before labeling)

This guideline defines how the hand-annotated triple gold (`gold/extraction_gold.jsonl`)
is produced. It is committed **before** any paragraph is labeled and **before** any
extractor output is seen, so the gold reflects the language, not the model's behavior.

## Unit of annotation

A **triple** is one atomic fact `(subject, relation, object)` that is **stated or
unambiguously entailed by the text of a single paragraph**. Annotate **every** true
triple in the paragraph, not only the answer-relevant ones — precision can only be
measured against an exhaustive gold.

- **Subject** = the entity the fact is *about* (usually the paragraph's topic entity, or
  another named entity the sentence is predicating over).
- **Relation** = a **canonical relation** from the vocabulary below.
- **Object** = the value: a named entity, a date, a place, or a short literal span.

Both subject and object are recorded as the **surface string from the text** (after
resolving the coreference that points at them — see below). Normalization for matching is
the matcher's job, not the annotator's; the gold keeps the human-readable surface form.

### What is NOT a triple
- Vague, hedged, or non-factual prose ("considered one of the fathers of spaghetti
  westerns", "his work was well above the usual standards"). Subjective/qualitative
  statements are skipped.
- Facts requiring **cross-paragraph** inference. Only within-paragraph facts count.
- Pure existence/aggregate counts with no entity object ("directed 102 films") are skipped
  unless the count is itself the asked-about value; prefer the `count of works` open
  relation only when a concrete number is stated and clearly factual. (Used sparingly.)

## One fact = one triple (n-ary decomposition)

N-ary facts are split into binary triples sharing the subject:

- "born 11 October 1926 in Genoa" → `(X, date of birth, 11 October 1926)` **and**
  `(X, place of birth, Genoa)`.
- "an Italian director, screenwriter and actor" → three `occupation` triples
  `(X, occupation, director)`, `(X, occupation, screenwriter)`, `(X, occupation, actor)`,
  plus `(X, country of citizenship, Italy)` (adjectival nationality → citizenship).
- A cast list "stars A, B and C" → one `performer` triple per actor.

Each emitted triple stands alone and is scored independently.

## Coreference

Resolve pronouns and definite descriptions to the named entity they refer to **within the
paragraph**, and record that named entity as the subject/object. "He died in Rome" in the
Tessari paragraph → subject `Duccio Tessari`. Do not invent entities the paragraph never
names.

## Direction convention

Triples are directional and subject is the topic entity:
- `(film, director, person)`, `(film, performer, person)`, `(film, based on, work)`,
  `(film, producer, person)`, `(film, publication date, date)`.
- `(person, father, person)`, `(person, mother, person)`, `(person, spouse, person)`,
  `(person, date of birth, date)`, `(person, place of birth, place)`.
- `(org, founded by, person)`, `(org, inception, date)`.

Symmetric relations (`spouse`, `sibling`) are annotated **once**, in the direction the
sentence states (topic entity as subject). The matcher does not assume symmetry.

## Canonical relation vocabulary

**Closed core set** — reuse the 2WikiMultiHopQA / Wikidata property names already present
in `gold/questions.jsonl`. Phrase the relation as exactly one of:

```
director, performer, composer, publisher, producer, screenwriter, based on,
publication date, country of origin, has part,
date of birth, date of death, place of birth, place of death, place of burial,
cause of death, father, mother, spouse, sibling, child,
country of citizenship, educated at, employer, award received,
founded by, inception, country, presenter
```

**Open extension set** — for true facts whose predicate is not in the closed set above.
Keep this list small and reuse a label rather than inventing a near-duplicate:

```
occupation        # "was an Italian director" -> occupation=director
nationality       # only when distinct from citizenship and explicitly named
birth name        # "His birth name was Jay John Fox"
genre             # "contemporary Christian music"
member of         # "lead singer of The Benjamin Gate"
acquired by       # "GlobalPost was acquired by WGBH"
production company # film made by a studio/company
nominated for     # award nomination (distinct from award received)
```

The five **core relations** for the small-sample weighting in the gate
(`director, father, spouse, date of birth, place of birth`) are all in the closed set.

## Object formatting

- **Dates** keep the surface form ("11 October 1926", "1935", "January 12, 1978"). The
  matcher normalizes date formats; the annotator does not.
- **Places** keep the most specific named place stated ("Genoa"); if "city, state/country"
  is given as the value, record the full "Gainesville, Texas" string — the matcher handles
  partial-place alignment.
- **Entities** keep the surface name; aliases/abbreviations are reconciled by the matcher's
  alias table, not pre-merged here.

## Record format (`gold/extraction_gold.jsonl`)

One JSON object per triple:

```json
{"chunk_id": "c00725", "subject": "Duccio Tessari", "relation": "date of birth",
 "object": "11 October 1926", "subj_span": [0, 14], "obj_span": [16, 31]}
```

`subj_span` / `obj_span` are `[start, end)` char offsets into the chunk text for
provenance; best-effort for the first surface mention of the resolved entity.
