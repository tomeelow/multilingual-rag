# Evaluation analysis

Interpretation of the generated [results.md](results.md) (which is
overwritten on every `uv run python -m src.eval.run_eval`; this file is not).
Numbers below from the 2026-06-11 run, 57-item golden set, open corpus
(`cross_lingual=True` for all items — the hardest setting, where every query
competes against all 3,710 chunks in three languages).

## Headline finding: a multilingual reranker fixes the cross-language failure

The original `cross-encoder/ms-marco-MiniLM-L-6-v2` is trained on English
MS MARCO. It drove cross-lingual unit hit@5 to **0.0** by demoting correct
foreign-language passages that multilingual E5 had retrieved.

The production pipeline now uses `BAAI/bge-reranker-v2-m3`. On the same
57-item golden set:

- **Overall:** unit hit@5 improves from 0.655 to **0.782**, and MRR from
  0.526 to **0.675**.
- **Cross-lingual items:** unit hit@5 improves from 0.0 to **0.364**.
- **Polish queries:** hit@5 improves from 0.524 to **0.714**.
- **Ukrainian queries:** MRR improves from 0.379 to **0.727**.
- **English queries:** hit@5 also improves, from 0.826 to **0.870**.

For the concrete Ukrainian query asking which AI systems are prohibited in
the EU, dense retrieval and RRF both ranked AI Act Article 5 first. The old
reranker removed every AI Act chunk from its top 10; the multilingual model
ranks Article 5 first.

## Second finding: naive RRF fusion dilutes cross-lingual retrieval

Hybrid (RRF) improves document-level recall (0.830 vs 0.813) and helps
exact-token queries, but *lowers* unit hit@5 (0.673 vs 0.727): on
cross-lingual items the BM25 leg can only contribute same-language (wrong)
chunks, and fusing them pushes correct dense hits out of the top 5. ADR-3's
"complementary failure modes" holds monolingually; cross-lingually BM25 is
not complementary — it is noise. A language-aware fusion weight would be the
refinement.

## Where reranking helps

Multi-hop questions: hit@5 doubles (0.4 → 0.8, MRR 0.4 → 0.7). When the
top-20 candidate pool spans two documents, the cross-encoder is good at
pulling relevant articles into the top 5.

## Latency

Retrieval-side p50/p95 (M-series laptop, MPS embeddings):

| stage | p50 | p95 |
|---|---|---|
| dense | 91 ms | 132 ms |
| hybrid (RRF) | 91 ms | 101 ms |
| hybrid + multilingual rerank | 6,503 ms | 8,480 ms |

The multilingual reranker fixes relevance but is now the dominant local
latency cost and exceeds the original <3 s target on CPU. A distilled
multilingual reranker or GPU-backed serving is the next performance task.
Time-to-first-token still depends on the generation model and needs an API
key to measure.

## What the numbers do *not* show yet

ragas generation metrics (context precision/recall, faithfulness, answer
relevancy) and the HyDE configuration require an LLM key; the harness supports
both (`--ragas`) and the results table has a placeholder section until that run
happens.

The ragas path is verified working — a single-item run scores all four metrics
— but a full table has not been produced, and the obstacle is API quota rather
than wiring. The metrics cost ~11 judge calls per item per config, so the
four-config table needs ~2,700 requests against a free tier capped at 20/day.
Measured budget and the reasoning in
[../phases/ragas_generation_eval.md](../phases/ragas_generation_eval.md).

The retrieval numbers above also predate the retrieval quality fixes
(`../phases/retrieval_quality_fixes.md`). Two of those reach the ablation
through the reranker — rights/duties intent weighting and legal-unit dedup —
so the `hybrid_rerank` row in particular is stale and should be re-measured.
The article-range and comparative fixes sit above `retriever.hybrid()` and do
not affect these configs by design. The ablation is deterministic and needs no
API key, so re-running it costs nothing but time.
