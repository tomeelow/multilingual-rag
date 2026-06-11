# Architecture Decision Records

Decisions recorded before implementation. Each entry states the decision, the
alternative considered, and the reasoning. Status of all three: **accepted**.

## ADR-1: Qdrant over ChromaDB as the vector store

**Decision:** use Qdrant, run via Docker locally and in deployment.

**Alternative:** ChromaDB.

**Reasoning:**

- Qdrant persists to disk by design and ships as a single production-grade
  Docker image; the same image runs in development and deployment, so there is
  no behavioural gap between the two environments.
- Payload-based metadata filtering is a first-class, indexed feature. This
  corpus needs filtering on `language` and `jurisdiction` at query time
  (EN/PL/UK queries retrieve from matching-language sources by default), which
  must be cheap and exact.
- Hybrid search support is built in, leaving room to move the BM25 leg
  server-side later without an architecture change.
- ChromaDB is convenient for prototypes but its persistence story and
  filtering performance are weaker, and it would be replaced before
  deployment anyway — starting on Qdrant avoids one migration.

## ADR-2: `intfloat/multilingual-e5-large` over `paraphrase-multilingual-mpnet-base-v2`

**Decision:** embed all chunks and queries with `intfloat/multilingual-e5-large`
(1024-dim), pinned per index.

**Alternative:** `paraphrase-multilingual-mpnet-base-v2`.

**Reasoning:**

- e5-large is instruction-tuned with asymmetric `"query: "` / `"passage: "`
  prefixes. Queries and legal passages come from very different text
  distributions (short colloquial questions vs. long statutory prose); an
  asymmetric model bridges that gap, a symmetric paraphrase model does not.
- e5-large outperforms mpnet-base by a significant margin on cross-lingual
  retrieval benchmarks (MIRACL, MTEB retrieval tasks), and cross-lingual
  retrieval (Ukrainian question → Polish statute) is the core requirement
  here, not a nice-to-have.
- The cost is size (560M params, 1024-dim vectors vs. 278M, 768-dim). For a
  corpus of nine documents this is irrelevant at index time and acceptable at
  query time.
- Consequence: the prefixes are mandatory. Every embedding call goes through
  one module that applies them; nothing else may call the model directly.

## ADR-3: Hybrid retrieval (dense + BM25) over pure dense

**Decision:** retrieve with dense (Qdrant) and lexical (BM25 via `bm25s`) legs
in parallel, merged by Reciprocal Rank Fusion, then cross-encoder reranking.

**Alternative:** dense-only retrieval.

**Reasoning:**

- Legal text is named-entity-heavy: queries cite exact tokens — "Article 22",
  "Art. 6 ust. 1 lit. f", "Стаття 8", "2016/679". Dense embeddings generalize
  semantically and routinely blur exactly these distinctions (Article 6 vs.
  Article 26 are semantically near-identical to an embedding model). BM25
  matches them exactly.
- Conversely, BM25 fails cross-lingually (a Ukrainian query shares no tokens
  with an English regulation), which is where the dense leg carries.
  The two failure modes are complementary, which is the precondition for rank
  fusion actually helping.
- RRF over score fusion: dense cosine scores and BM25 scores live on
  incomparable scales; fusing by rank position needs no calibration.
- The evaluation harness (Phase 8) measures dense-only as a baseline, so this
  decision is checked against numbers rather than taken on faith.

## Notes on dependency choices

The plan lists `rank-bm25`/`langdetect`/`pypdf` alongside their chosen
replacements (`bm25s`, `lingua-language-detector`, `pdfplumber`+`PyMuPDF`).
Only the chosen tools are declared as dependencies; carrying the rejected
alternatives in the lockfile would be dead weight. Formatting uses
`ruff format` instead of black — one tool for lint and format, same style.
