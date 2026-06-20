# kg-rag-multihop

[![CI](https://github.com/rkendev/kg-rag-multihop/actions/workflows/ci.yml/badge.svg)](https://github.com/rkendev/kg-rag-multihop/actions/workflows/ci.yml)

**A study of whether knowledge graphs help AI answer multi-step questions, and an honest finding that overturned its own positive result.**

▶ **Try it (no setup, opens in your browser):** [Live results explorer](https://rkendev.github.io/kg-rag-multihop/) — browse 100 real test questions and see, side by side, where the approach helped and where it did not.

---

## In plain English

Some questions can only be answered by connecting two or more separate facts. For example: *"Who is the maternal grandfather of person X?"* requires first finding X's mother, then finding her father. Standard AI document-search is good at fetching one fact but often misses the link between them.

This project tests a popular idea: if you first build a **knowledge graph** (a map of how facts connect) and let the AI walk that map, does it answer these multi-step questions better? I built the system end to end and compared it head to head against standard search, on a public 100-question benchmark.

**The result.** At first the knowledge-graph approach looked like a clear win, a large jump in answer quality. But I suspected the test had been unintentionally rigged in the graph's favor (the graph was built only from the paragraphs that contained the answers). So I rebuilt it under realistic conditions and ran the comparison again. **The win largely disappeared.** The graph genuinely finds better evidence, but a small AI model cannot turn that better evidence into better answers. Along the way I also found a second hidden flaw, in how the system tells apart different people with the same name, that the easy test had completely masked.

Every number in this repository is reproducible, and the comparison was kept strictly fair (only the search step changed; the AI model and the scoring were identical and verified unchanged).

---

## What this project demonstrates

- **Building a non-trivial AI system end to end** (document processing, retrieval, knowledge-graph construction, evaluation), entirely on local hardware at zero cost.
- **Designing a fair experiment** and proving it stayed fair, the kind of controlled comparison that makes a result mean something.
- **Distrusting my own positive result.** The win was real and fair, and still misleading as a headline. Catching that before a reviewer does is the judgment that separates senior engineers from people who just ship.
- **Diagnosing *why* something breaks,** not just whether it scores: the bottleneck turned out to be the AI model, not the search.
- **Reproducibility and honest reporting,** including documenting the limitations and the failed assumptions rather than hiding them.

---

## The finding, in one look

Answer quality (higher is better), measured three ways:

| | Standard search | Graph, easy test | Graph, realistic test |
|---|---:|---:|---:|
| Overall answer score | 22.1 | **30.9** (big jump) | **22.0** (no real change) |
| Found the right evidence | 74.2 | 89.0 | 82.5 (still better) |

*Deltas are computed from unrounded scores.*

The graph **keeps finding better evidence** even on the realistic test. The catch is that the small AI model does not convert that better evidence into better answers. On the realistic test, the graph changed the final answer for only **15 of 100 questions** (it improved 7, hurt 8, left 85 unchanged).

*Want the detail?* The [live explorer](https://rkendev.github.io/kg-rag-multihop/) lets you click any of the 100 questions and watch the graph fetch the right evidence while the model still fails to use it. Full numbers and methodology are in [`THESIS_RESULT_v2.md`](THESIS_RESULT_v2.md).

---

## A few terms, briefly

- **RAG (Retrieval-Augmented Generation):** the AI looks up relevant documents before answering, instead of answering from memory.
- **Multi-hop question:** one that needs two or more facts chained together.
- **Knowledge graph:** a network of entities (people, films, places) and how they relate, built automatically from the documents.

---

## How it was built (for the technically curious)

Standard hybrid retrieval (semantic + keyword) as the baseline; a knowledge graph (entity extraction, resolution, a graph store) layered on top so the search can *traverse* from the question to a second-hop fact the question never names. The graph's results are fused with the baseline, and the **same** AI model and scoring judge the answers in both cases, so any difference comes only from the search step. Everything runs locally (no paid APIs), is deterministic, and is built in gated phases, each phase had to pass an explicit quality check before the next began.

<details>
<summary><b>Full phase-by-phase build log and gate results</b> (click to expand)</summary>

Each phase has its own detailed report; together they are the audit trail.

| Phase | What it produced | Report |
|---|---|---|
| P0 | Frozen corpus, gold multi-hop question set, measured baseline | [`BASELINE.md`](BASELINE.md) |
| P1 | Triple-extraction quality gate (passed: core-relation F1 0.90) | [`EXTRACTION_BASELINE.md`](EXTRACTION_BASELINE.md) |
| P2 | Knowledge graph, gated on entity-merge precision (0.93) | [`GRAPH_BUILD.md`](GRAPH_BUILD.md) |
| P3 | The thesis test, the apparent win (+8.8 F1) | [`THESIS_RESULT.md`](THESIS_RESULT.md) |
| Regrow | Defensibility re-test on a realistic graph, the overturn | [`THESIS_RESULT_v2.md`](THESIS_RESULT_v2.md), [`DEFENSIBILITY_REGROW.md`](DEFENSIBILITY_REGROW.md) |

**Why the numbers are trustworthy:** the baseline was frozen before any graph existed; only retrieval changed across runs (generator, prompt, judge, metrics, and test set are byte-identical, proven by an empty `git diff`); all metrics recompute deterministically from stored outputs; each phase passed an explicit gate, and a failed gate triggered a documented pivot, not a workaround.

Two findings from the regrow:
1. The apparent +8.8 win was substantially an artifact of testing on an *answer-enriched* graph. On a realistic graph (with distractor passages), overall answer score returns to baseline and the largest question type regresses. Retrieval still improves; answers do not, the small generator is the bottleneck.
2. The graph's clean-looking entity resolution (precision 0.93) was *also* an artifact. On realistic data, full of same-name people the benchmark uses as distractors, the same rules collapsed to 0.82. Four surgical, documented fixes restored it to 0.90 without losing connectivity; the residual errors are the irreducible limit of string-only matching.

</details>

---

## Stack

| Component | Choice |
|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` (CPU, FAISS) |
| Keyword | `rank_bm25` |
| Fusion | Reciprocal Rank Fusion |
| Entity linking | GLiNER + alias/embedding match |
| Graph store | Kùzu (embedded) |
| Generator | Ollama `qwen2.5:7b-instruct` (local) |
| Judge | Ollama `llama3.1:8b-instruct` (local, advisory) |

## Run it yourself

Requires [uv](https://docs.astral.sh/uv/) and a local Ollama with the two models above. Note: full reproduction runs the models on CPU and takes hours, the hosted explorer above is the fast way to see the results.

```bash
just bootstrap        # set up the environment
just build-corpus     # download the benchmark, freeze corpus + questions
just baseline         # run the standard-search baseline
just eval             # score it
just extract-corpus   # build the knowledge graph (extract, resolve, store)
just build-graph
just kgrag            # graph-augmented retrieval + the unchanged generator
just eval-kgrag       # score it with the same judge
just verify-kgrag     # deterministic recompute + fairness check
```

## License

Code: MIT (see [LICENSE](LICENSE)). Dataset ([2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop)) is Apache-2.0; its license is retained under `data/`. The derived corpus and question sets are committed under the dataset's permissive terms; the raw dataset is not.
