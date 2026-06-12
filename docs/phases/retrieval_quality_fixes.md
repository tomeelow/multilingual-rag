# Retrieval quality fixes — three confirmed failure modes

Date: 2026-06-12. Follows `phase7_status.md`. Test suite: 66 passed.

Three failure modes confirmed by manual query testing, reproduced against
the local indexes, fixed, and covered by regression tests. Branches are
**stacked** (each builds on the previous — they share
`src/pipeline/query_analysis.py`):

```
main
 └─ fix/retrieval-ranking-rights-duties
     └─ fix/article-range-filter
         └─ fix/comparative-source-diversification   <- contains all fixes
```

## Metadata audit (step before any fix)

Chunk payloads already carry everything needed: `article_number` (raw, with
superscripts: `11¹`, `18³a`), `section_title` (nearest chapter heading —
"Rozdział II — Podstawowe zasady prawa pracy" vs "Rozdział II — Obowiązki
pracownika" are distinct), `source_id`, `language`, `jurisdiction`, `kind`.
Both GDPR language versions are indexed (`gdpr_en`, `gdpr_pl`), embeddings
are `multilingual-e5-large`. **No re-index was needed**; all fixes live in
the retrieval layer.

## Fix 1 — rights query topped by duties article

`Jakie są podstawowe prawa pracownika…` ranked Art. 100 KP (duties) first;
the Chapter II candidates were present but reranked below it. Polarity is
invisible to topical similarity. Now: rights/duties intent is detected from
the query (en/pl/uk stems, neutral when mixed), cross-encoder scores are
sigmoid-normalized and weighted — opposing chapter heading ×0.35, aligned
heading ×1.25, obligation/rights language in text ×0.7 as fallback.
Result: top-5 = Arts. 13, 14, 1, 18, 11¹; Art. 100 gone.

## Fix 2 — explicit article range ignored

`art. 10–18 Kodeksu pracy` returned Arts. 100 and 210; Arts. 11¹–11³, 15,
17 never reached the RRF candidate pool, so a post-filter alone could not
fix it. Now: `parse_article_range` (art./artykuł/article/ст., en/pl/uk,
–/—/- dashes) detects the range; in-range child chunks are injected from
metadata into the rerank pool (sources already retrieved first, capped at
40); out-of-range candidates only fill leftover slots. Range membership is
by base number (`11¹` ∈ [10, 18]) and `kind == "article"` (recital 14 is
not article 14). Result: top-5 = Arts. 10, 18, 14, 13, 18¹ — all in range,
all Labour Code.

## Fix 3 — comparative query returns one side only

`difference between consent under GDPR and Ukrainian law` returned a
single side (UA-only in earlier manual testing, GDPR-only when reproduced
here — same failure either way). Three changes:

1. **Dedup** (first commit, cheapest): reranked results collapse to one
   chunk per `(source_id, kind, article_number)` (parent_id fallback for
   unnumbered units) — Art. 6 no longer appears twice.
2. **Per-side retrieval**: comparative phrasing (en/pl/uk) + ≥2 named
   documents (alias table → source_id groups) triggers one source-filtered
   hybrid retrieval per side (Qdrant `MatchAny` on `source_id`, BM25
   post-filter with 10× headroom), each reranked separately, interleaved in
   mention order. Result: GDPR Art. 7 / Recitals 32, 40 interleaved with UA
   law Стаття 2 and Стаття 6.
3. **Diversification fallback**: comparative phrasing without recognizable
   document names — if top-5 is single-source, the best other-source
   candidate takes the last slot.

## Judgment calls

- **Stacked branches**, not independent: all three fixes extend the same
  new module; independent branches would conflict on merge.
- **Inside a comparative source group there is no language filter**, even
  with `cross_lingual=False`: naming a foreign law in the query is treated
  as an explicit request to cross the language boundary (the UA law exists
  only in Ukrainian; honoring the filter would silently re-create the bug).
- **Range beats comparative** when both are detected in one query — the
  explicit article range is the harder constraint.
- **Regression tests instead of golden-set items**: the three queries live
  in `tests/test_rag_integration.py` (runs only when local indexes exist),
  not `src/eval/golden_set.yaml`, so committed eval artifacts in
  `docs/eval/` stay comparable; the eval ablation configs in `run_eval.py`
  deliberately keep calling the base retriever+reranker (they measure the
  pipeline stages, and get dedup + intent weighting through the reranker).
- **Heuristics, not NLP**: per the task brief, range/intent/comparative
  detection are regex/keyword stems in `src/pipeline/query_analysis.py` —
  pure functions, no models, unit-tested in `tests/test_query_analysis.py`.
- The document alias table is keyed to the current corpus; extend it when
  documents are added.
