# Evaluation analysis

Interpretation of the generated [results.md](results.md) (which is
overwritten on every `uv run python -m src.eval.run_eval`; this file is not).
Numbers below from the 2026-06-11 run, 57-item golden set, open corpus
(`cross_lingual=True` for all items — the hardest setting, where every query
competes against all 3,710 chunks in three languages).

## Headline finding: the English cross-encoder is a net negative outside English

`cross-encoder/ms-marco-MiniLM-L-6-v2` (pinned by the plan) is trained on
English MS MARCO. The ablation makes its behavior precise:

- **English queries:** reranking is the best configuration — unit hit@5
  0.826, MRR 0.699 (vs 0.783/0.523 dense-only). This is the precision boost
  reranking is supposed to buy.
- **Cross-lingual items:** unit hit@5 drops to **0.0**. The reranker
  systematically demotes correct foreign-language passages that dense
  retrieval had surfaced — it cannot score a Ukrainian question against an
  English statute, so correct candidates lose to same-language noise.
- **Polish/Ukrainian queries overall:** hit@5 falls from 0.667/0.727 (dense)
  to 0.524/0.545 (reranked).

The production default keeps the plan's pipeline, but the right fix is a
multilingual cross-encoder (e.g. `bge-reranker-v2-m3`) — first item under
"what I would do with more time".

## Second finding: naive RRF fusion dilutes cross-lingual retrieval

Hybrid (RRF) improves document-level recall (0.830 vs 0.813) and helps
exact-token queries, but *lowers* unit hit@5 (0.673 vs 0.727): on
cross-lingual items the BM25 leg can only contribute same-language (wrong)
chunks, and fusing them pushes correct dense hits out of the top 5. ADR-3's
"complementary failure modes" holds monolingually; cross-lingually BM25 is
not complementary — it is noise. A language-aware fusion weight would be the
refinement.

## Where reranking unambiguously helps

Multi-hop questions: hit@5 doubles (0.4 → 0.8, MRR 0.4 → 0.567). When the
top-20 candidate pool spans two documents, the cross-encoder is good at
pulling both relevant articles into the top 5 — for English-language pairs.

## Latency

Retrieval-side p50/p95 (M-series laptop, MPS embeddings):

| stage | p50 | p95 |
|---|---|---|
| dense | 60 ms | 107 ms |
| hybrid (RRF) | 54 ms | 64 ms |
| hybrid + rerank | 353 ms | 683 ms |

Reranking dominates retrieval latency but stays well inside the < 3 s
end-to-end target; time-to-first-token depends on the generation model and
needs an API key to measure (the harness records generation token counts and
cost when run with `--ragas`).

## What the numbers do *not* show yet

ragas generation metrics (context precision/recall, faithfulness, answer
relevancy) and the HyDE configuration require an LLM key; the harness
supports both (`--ragas`) and the results table has a placeholder section
until that run happens.
