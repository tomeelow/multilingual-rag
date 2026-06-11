# Evaluation results

Golden set: 57 items · all retrieval open-corpus (cross_lingual=True) · run 2026-06-11

## Retrieval ablation (deterministic, no LLM judge)

| Configuration | unit hit@5 | unit recall@5 | MRR | doc recall@5 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| dense | 0.727 | 0.682 | 0.544 | 0.813 | 60.4 | 106.7 |
| hybrid | 0.673 | 0.645 | 0.487 | 0.83 | 54.1 | 64.4 |
| hybrid_rerank | 0.655 | 0.606 | 0.526 | 0.76 | 352.6 | 682.9 |

## By difficulty (unit hit@5 / MRR)

| Configuration | simple | cross-lingual | multi-hop |
|---|---|---|---|
| dense | 0.897 / 0.658 | 0.273 / 0.205 | 0.4 / 0.4 |
| hybrid | 0.846 / 0.601 | 0.182 / 0.182 | 0.4 / 0.267 |
| hybrid_rerank | 0.821 / 0.669 | 0.0 / 0.0 | 0.8 / 0.567 |

## By question language (unit hit@5 / MRR)

| Configuration | en | pl | uk |
|---|---|---|---|
| dense | 0.783 / 0.523 | 0.667 / 0.537 | 0.727 / 0.598 |
| hybrid | 0.783 / 0.476 | 0.571 / 0.421 | 0.636 / 0.636 |
| hybrid_rerank | 0.826 / 0.699 | 0.524 / 0.413 | 0.545 / 0.379 |

## Generation quality (ragas, LLM-judged)

_Not run: requires an LLM API key (`OPENAI_API_KEY` in `.env`). Run `uv run python -m src.eval.run_eval --ragas` to fill this table._
