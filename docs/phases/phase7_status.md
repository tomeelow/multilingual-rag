# Status after Phase 7 — query intelligence, API, frontend

Date: 2026-06-11. Covers Phases 5–7 (see `phase4_status.md` for 0–4).
Phases 8–10 not started. Test suite: 41 passed.

## Done

### Phase 5 — query intelligence (`phase-5: query intelligence`)
- `src/pipeline/hyde.py`: HyDE — the LLM writes a hypothetical statutory
  excerpt (same language as the question, formal style), embedded with the
  e5 **passage** prefix and used as the dense retrieval vector.
- Wired into the pipeline as an opt-in flag: `answer(..., use_hyde=True)` and
  `stream_answer(..., use_hyde=True)`. Per the plan's pitfall note, HyDE is
  an A/B option, never the only path: BM25 keeps the raw query and the
  cross-encoder reranks against the real query, so a hallucinated hypothesis
  cannot poison the lexical leg or the final ranking.
- Hypothesis generation goes through the cache-first LLM client — repeated
  HyDE retrievals for the same query cost one API call total (tested).
- Query decomposition **not** implemented — the plan explicitly defers it
  ("Only implement HyDE in Phase 5; decomposition is a stretch goal").

### Phase 6 — FastAPI backend (`phase-6: fastapi backend`)
- `src/api/main.py` (app factory + lifespan that warms the pipeline at
  startup) and `src/api/schemas.py`:
  - `POST /api/chat` — SSE stream of `data: {"token": …}` events followed by
    one `data: {"done": true, sources, chunk_ids, latency breakdown}` event.
    Request: `{query, session_id?, filter_language?, cross_lingual?,
    use_hyde?}`. Pipeline exceptions become a final
    `{"error": …, "done": true}` event instead of a broken stream.
  - `GET /api/sources` — all 9 indexed documents with full metadata
    (language, jurisdiction, official flag, unit counts, URLs).
  - `GET /api/health` — `{status, indexed_chunks, languages, collection,
    embedding_model}` from the recorded index metadata.
- Structured loguru logging per chat request: detected language, filters,
  HyDE flag, chunk/source counts, retrieval/rerank/generation/total
  latency. **Privacy (CLAUDE.md): the query text is never logged** — only
  a sha256 prefix, character length, and language.
- Verified over real HTTP: pipeline warm-up, health, sources, and the chat
  endpoint running full retrieval then degrading to a clean SSE error event
  when no LLM key is configured.

### Phase 7 — Streamlit frontend (`phase-7: streamlit frontend`)
All five plan features, in the plan's priority order:
1. Chat interface with true token streaming (`st.write_stream` over the SSE
   client in `frontend/api_client.py`).
2. Sources panel per answer: language flag emoji + colour-coded badge
   (🇬🇧 blue / 🇵🇱 red / 🇺🇦 gold), article/recital ref, document title
   linked to the official source, page range. Unofficial translations carry
   an `UNOFFICIAL` badge suffix.
3. Language selector: auto-detect (lingua) or manual en/pl/uk override.
4. Example queries in all three languages as one-click buttons, including a
   deliberately cross-lingual one (🇺🇦→🇬🇧 Ukrainian question answered from
   the English AI Act).
5. Session history capped at the last 5 exchanges.
- Plus: cross-lingual toggle (on by default in the UI — see deviations),
  HyDE toggle (off by default), corpus inventory expander, latency caption
  per answer, friendly error when the API is down.
- `frontend/api_client.py` keeps HTTP/SSE parsing free of Streamlit imports;
  the protocol seam is unit-tested. App execution verified with Streamlit's
  `AppTest` against a live API (renders cleanly, all controls present) and
  by booting `streamlit run` headless next to uvicorn.

## Deviations and judgment calls

1. **`POST /api/ingest` not implemented.** The plan itself marks it
   "(stretch goal for demo)"; skipped to keep scope per the no-gold-plating
   rule. The ingestion path exists as CLI modules.
2. **Extra request fields beyond the plan's API sketch:** `cross_lingual`
   and `use_hyde` on `/api/chat`. Both are required to exercise features the
   plan demands elsewhere (HyDE as an A/B option; cross-lingual retrieval).
3. **Cross-lingual default differs by layer, deliberately.** Core pipeline
   default stays `False` (CLAUDE.md: same-language retrieval by default,
   cross-language only when explicitly enabled); the Streamlit UI ships with
   the toggle **on** — the frontend explicitly enables it, it is user-visible,
   and the cross-lingual demo is the project's headline. Reviewers see the
   flag, not hidden behavior.
4. **`session_id` is accepted and logged but not used for conversation
   memory** — the plan's pipeline is single-turn (no multi-turn prompting is
   specified anywhere); the field exists so the API contract matches the
   plan and the frontend sends a stable per-session UUID.
5. **SSE via plain `StreamingResponse`** exactly as in the plan's code
   sample; the `sse-starlette` dependency declared in Phase 0 turned out
   unnecessary and was removed from `pyproject.toml`.
6. **Streaming latency caveat:** `stream_answer` (and therefore the SSE
   endpoint) reports retrieval/rerank/generation breakdown, where
   "generation" covers the whole streamed window. Time-to-first-token will be
   measured in Phase 8 against the < 3 s target.
7. **No live LLM end-to-end run yet** — still no API key in this
   environment (carried over from Phase 4 status). Retrieval, SSE protocol,
   caching, and UI are verified; generation runs through tested fakes. With
   a key in `.env`, no code changes are needed.
8. **Frontend has no automated UI tests beyond AppTest + the SSE parsing
   unit test.** Consistent with CLAUDE.md (tests required for extraction/
   chunking/retrieval scoring — all covered; UI glue is not on that list).

## How to run

```bash
uv run uvicorn src.api.main:app --port 8000   # terminal 1
uv run streamlit run frontend/app.py          # terminal 2 (needs API_URL if not localhost)
```

## Next (not started)

Phase 8 (50+ item golden set, ragas, 4-config ablation, cost tracking),
Phase 9 (docker compose full stack: the plan's Phase-7 deliverable line
"docker compose up starts Qdrant + API + Streamlit" belongs to Phase 9's
containerization and is intentionally not claimed here), Phase 10 (README,
blog post, demo video).
