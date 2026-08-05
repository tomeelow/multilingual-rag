# ragas generation eval — verified working, blocked on API quota

Date: 2026-08-05. Follows `retrieval_quality_fixes.md`. Test suite: 73 passed.

The ragas path is wired, imports resolve, and it produces real scores. It has
never produced a **full** table, and the reason is quota, not code. This note
records the measured request budget so the next attempt is a billing decision
rather than another exploratory run.

## Verification: the path works

A single-item run against the production config (`hybrid_rerank`) returned:

```
context_precision 0.7 · context_recall 1.0 · faithfulness 1.0 · answer_relevancy 0.944
```

Confirmed working: `ragas` 0.2.15 + `langchain` 0.3.30 + `langchain-google-genai`
2.1.12, the 57-item golden set, embedded Qdrant (3,710 chunks — **no Docker
needed**, `QDRANT_URL` unset selects embedded mode), and the local e5 model
backing ragas embeddings so no embedding API is involved.

## The blocker: 20 requests per day

The free tier returns both limits in the 429 body:

| quota | value |
|---|---|
| `GenerateRequestsPerDayPerProjectPerModel-FreeTier` | **20 / day** |
| `GenerateRequestsPerMinutePerProjectPerModel-FreeTier` | **5 / minute** |

Measured consumption: **11 judge calls + 1 generation per item, per config.**
The four metrics are not one call each — `context_precision` costs one call per
context, `faithfulness` two (statement extraction, then NLI), `answer_relevancy`
three.

| scope | requests | at 20/day |
|---|---|---|
| 3 items, 1 config | 36 | already over quota |
| 57 items, 1 config | ~684 | 34 days |
| 57 items, 4 configs | **~2,736** | **137 days** |

A 3-item proof run took 10m27s and still failed: one judge job exhausted its
retries and `context_precision` came back `nan`. That NaN is what motivated
fix 3 below.

Paid Tier 1 lifts this to ~1,000 RPM, which makes the full four-config run a
~15–30 minute job. Estimated **$3–5**, from measured token usage (~1,768 in /
486 out per generation; judge calls carry the retrieved contexts, so they
dominate). Note this is *not* what the results table reports — see below.

## Three robustness fixes

1. **Incremental writes** (`main`). Results were written only after every config
   finished, so a judge dying on config 3 of 4 discarded the retrieval work for
   configs 1–2 — minutes of reranking each. `write_results` is now called after
   each config's retrieval stage and again after its judge stage.

2. **Provider-aware judge** (`judge_llm`). The judge was hardwired to
   `ChatGoogleGenerativeAI` regardless of `LLM_PROVIDER`, so switching provider
   to escape the Gemini quota would have kept calling Gemini — generation on the
   new provider, judging on the old one. It now follows `LLM_PROVIDER`.

3. **NaN guard** (`check_all_items_scored`). pandas `.mean()` skips NaN, so a
   judge job that timed out on *some* rows would publish a mean over a smaller
   sample than the header claims — indistinguishable from a real score. The run
   now aborts naming the metric and the failure count. Safe to abort because of
   fix 1: earlier configs are already on disk.

## Judgment calls

- **No `langchain-anthropic` dependency.** `LLM_PROVIDER=anthropic` raises a
  message naming the exact fix instead. Adding a dependency is an ask-first
  action per `CLAUDE.md`, and openai + gemini are already pinned.
- **The NaN guard aborts rather than degrades.** For an eval whose numbers go
  in a README, a quietly shrunk denominator is worse than a failed run.
- **`generation_cost_usd` still counts generation only**, not judge calls —
  which are ~11× the volume. The column understates true eval cost by roughly
  10×. Left as-is (it is documented at the constant) rather than silently
  changing what a committed metric means, but the label is misleading.
- **Retrieval numbers in `docs/eval/results.md` are stale** (dated 2026-06-12).
  Rights/duties intent weighting and legal-unit dedup both landed in the
  reranker afterwards, so the `hybrid_rerank` row no longer reflects the code.
  The range and comparative fixes sit above `retriever.hybrid()` and do not
  touch the ablation. Refreshing needs no API key at all:
  `uv run python -m src.eval.run_eval --configs dense hybrid hybrid_rerank`.
- **Fixed a pre-existing red test** unrelated to ragas: `test_health` hardcoded
  `generation_provider == "openai"`, which broke when `.env` moved to gemini.
  It now pins `LLM_PROVIDER` via monkeypatch and asserts health reflects it.
