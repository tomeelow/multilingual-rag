# Status after Phase 4 — core RAG pipeline

Date: 2026-06-11. Covers Phases 0–4; Phases 5–10 not started.

## Done

### Phase 0 — setup (`phase-0: project setup`)
- uv project, Python 3.12 pinned, 212 packages locked (`uv.lock`).
- Repo structure per plan; `docs/ADR.md` with the three required decisions;
  `.env.example`; `configs/pipeline.yaml` for tunable parameters;
  pre-commit with ruff; docker-compose skeleton (Qdrant only until Phase 9).

### Phase 1 — ingestion (`phase-1: document ingestion`)
- All 9 corpus PDFs parsed into `data/processed/{source_id}.json` with full
  per-unit citation metadata. Unit counts verified against the official
  texts — exact for every structured document (GDPR 173 recitals + 99
  articles in EN and PL, DSA 156+93, AI Act 180+113+13 annexes, PL data
  protection act 176 articles, Labour Code 480 articles ending at Art. 305,
  UA law 30 articles).
- `src/ingestion/validate_corpus.py`: per-language counts, declared-vs-
  detected language (lingua), chunk length bounds. All checks pass.
- Layout traps found and handled are documented in `docs/ingestion_notes.md`
  (single-column CELEX PDFs, 8.5pt recital numbers in the 2024 OJ format,
  Labour Code superscript article numbers `18³a` vs `183`, footnote-separator
  detection, repealed-range gaps, EDPB paragraph 113 under a link underline).

### Phase 2 — chunking (`phase-2: hierarchical parent-child chunking`)
- Parent-child chunking on structural boundaries: children 256 tokens /
  32 overlap (retrieval), parents ≤ 1024 tokens (LLM context), tiktoken
  `cl100k_base` counting. 1986 parents / 3710 children; only 5.4% of units
  split across parents. Repealed stubs ("Art. 4. (uchylony)") dropped.
- Full provenance on every chunk: source_id, doc_title, language,
  jurisdiction, doc_type, official flag, url, ingestion_date, unit kind,
  article_number, display ref, section, pages.
- `notebooks/chunking_analysis.ipynb` (executed, committed with outputs):
  en 5.14 / pl 2.71 / uk 1.72 chars per token — token-based budgets verified
  as necessary.

### Phase 3 — indexes (`phase-3: embedding and vector store`)
- `intfloat/multilingual-e5-large` wrapper (`src/retrieval/embedding.py`) —
  the only module touching the model; `query:`/`passage:` prefixes enforced
  there and covered by a unit test.
- Qdrant collection `legal_docs_v1` (1024-dim cosine, full chunk payload),
  3710 child vectors. Server mode via `QDRANT_URL`, embedded local mode
  otherwise. Collection name carries the index version from config.
- BM25 (`bm25s`) over the same chunk_ids; lowercase unicode tokenizer, no
  stemmer (rationale in module docstring). `indexes/index_meta.json` records
  model, chunking config, version, counts, build date.
- `notebooks/retrieval_smoke_test.ipynb` (executed, committed): 10 queries
  across en/pl/uk incl. cross-lingual. Both legs return the right articles
  monolingually; the dense leg correctly bridges a Ukrainian question to the
  English AI Act Article 5 while BM25 stays same-language — the
  complementary failure modes ADR-3 predicts.

### Phase 4 — core pipeline (`phase-4: core RAG pipeline`)
- `src/pipeline/retriever.py`: hybrid retrieval, pure-function RRF
  (`rrf_fuse`, rank-based by design), parent lookup for context.
- `src/pipeline/rerank.py`: `cross-encoder/ms-marco-MiniLM-L-6-v2`, top 20 → 5.
- `src/pipeline/cache.py` + `src/pipeline/llm.py`: SQLite cache keyed by
  sha256 of (provider, model, messages, params). Every completion AND stream
  goes cache-first; streams replay from cache on a hit and fill it on a miss.
  Covered by tests (`_CountingLLM` proves no second network call).
- `src/pipeline/prompts.py`: language-aware system prompt, citations
  mandatory, refuse-without-context instruction, unofficial sources labeled
  `UNOFFICIAL TRANSLATION` in context headers.
- `src/pipeline/rag_chain.py`: `answer(query, filter_language, cross_lingual)
  -> RAGResponse` (pydantic: text, language, sources, chunk_ids, latency_ms +
  retrieval/rerank/generation breakdown, cached flag, token usage) and
  `stream_answer()` yielding token events + a final metadata event for SSE.
- 32 tests pass (extraction, chunking, retrieval scoring, RRF, cache, prompt,
  end-to-end with fake LLM against the real local indexes).

## Deviations and judgment calls

1. **Plan file name/location.** The prompt references
   `docs/implementation_plan.md`; the actual file is
   `docs/multilingual_rag_plan.md`. Followed the latter.
2. **No live LLM call has been made.** No `OPENAI_API_KEY`/`ANTHROPIC_API_KEY`
   is present in this environment, so generation is verified through the
   transport seam with fake LLMs (cache contract, prompt assembly, streaming
   protocol). First run with a real key needs `.env` (see `.env.example`).
3. **Cross-lingual retrieval is opt-in** (`cross_lingual=False` default) to
   honor CLAUDE.md ("matching-language sources by default; cross-language
   fallback only when explicitly enabled"), although the plan's headline
   scenarios are cross-lingual. The API/frontend (Phases 6–7) will expose the
   flag explicitly.
4. **Known limitation found empirically:** the plan-pinned ms-marco reranker
   is English-trained; for a Ukrainian query it can demote the English AI Act
   candidates that dense retrieval correctly surfaced. Kept per plan; its
   cost will be quantified by the Phase 8 ablation table (a multilingual
   cross-encoder would be the "more time" improvement).
5. **Dependency trimming.** The plan's `uv add` line lists alternatives that
   its own text rejects (`rank-bm25`, `langdetect`, `pypdf`, full
   `langchain`/`langchain-community`); only the chosen tools are declared
   (`bm25s`, `lingua-language-detector`, `pdfplumber`+`pymupdf`,
   `langchain-text-splitters`). `black` replaced by `ruff format`
   (CLAUDE.md explicitly allows). spaCy sentence models skipped — chunk
   boundaries are structural (articles/recitals), and the splitter's
   sentence-level separators cover the rest; nothing in the implemented
   pipeline needed sentence tokenization.
6. **Labour Code sub-chunking granularity:** the plan suggests article
   numbers like "Art. 22 ust. 1" as `article_number`; implemented metadata
   stops at the article level (paragraph §/ust. stays inside chunk text).
   Article-level citation matches how the corpus PDFs are structured and
   keeps unit IDs stable.
7. **Phase-1 deliverable timing:** `validate_corpus.py`'s 50–2000-char chunk
   checks (a Phase-1 deliverable that depends on Phase-2 chunks) activate
   when `chunks.json` exists; bounds apply to child chunks (parents are
   ≤ 1024 tokens by design and exceed 2000 chars legitimately).
8. **EDPB opinion: 132 of 135 paragraphs.** Three paragraph numbers don't
   survive text extraction; their text is glued to the preceding paragraph
   (content preserved, citation granularity slightly coarser for those three).
9. **`index_meta.json` lives in `indexes/`** (gitignored build artifact);
   the in-repo source of truth for index configuration is
   `configs/pipeline.yaml` — CLAUDE.md's "record index metadata" is satisfied
   by config + the generated metadata file next to the index itself.

## How to run what exists

```bash
uv sync
uv run python -m src.ingestion.run_ingestion      # PDFs -> data/processed/
uv run python -m src.ingestion.chunker            # -> chunks.json
uv run python -m src.ingestion.validate_corpus
uv run python -m src.retrieval.build_indexes      # Qdrant (embedded) + BM25
uv run pytest -q                                  # 32 tests
# with an API key in .env:
uv run python -c "from src.pipeline.rag_chain import answer; print(answer('Jakie kary przewiduje RODO?').text)"
```

## Next (not started)

Phase 5 (HyDE), Phase 6 (FastAPI/SSE), Phase 7 (Streamlit), Phase 8 (eval +
golden set + ragas + 4-config table), Phase 9 (full docker compose),
Phase 10 (README/blog/demo).
