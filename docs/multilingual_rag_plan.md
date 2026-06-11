# Implementation Plan: Multilingual RAG Legal/Regulatory Assistant (PL ↔ UK ↔ EN)

> **Time estimate:** ~9–10 weeks at a student part-time pace (~12–15h/week)  
> **Goal:** A production-shaped, portfolio-ready RAG system — not a tutorial reproduction

---

## Architecture Overview

```
User query (any language)
        │
        ▼
  Language detection
        │
        ▼
  Query rewriting / HyDE
        │
   ┌────┴────┐
   │         │
 Dense     BM25
retrieval  retrieval
(Qdrant)   (BM25s)
   │         │
   └────┬────┘
        │  Reciprocal Rank Fusion
        ▼
  Cross-encoder reranking
        │
        ▼
  Context assembly + source metadata
        │
        ▼
  LLM generation (language-aware prompt)
        │
        ▼
  Streamed response + citations (language of the query)
```

**Core design principle:** user gets a response in the language they asked in, pulling from documents in any language. The multilingual embedding model handles the cross-lingual semantic gap; citation metadata always includes the source document language.

---

## Document Corpus

Collect these before you write a single line of code — corpus quality defines project quality.

### Primary sources
| Document | Language(s) | Source |
|---|---|---|
| EU AI Act (full text, 2024) | EN | EUR-Lex PDF |
| GDPR Regulation No 2016/679 | EN + PL (official translation) | EUR-Lex |
| Polish Personal Data Protection Act (Ustawa o RODO) | PL | UODO / isap.sejm.gov.pl |
| Ukrainian Personal Data Protection Law (Закон № 2297-VI) | UK | zakon.rada.gov.ua |
| EU Digital Services Act (DSA) | EN | EUR-Lex PDF |
| Polish Labour Code (selected chapters) | PL | isap.sejm.gov.pl |
| AI regulatory compliance guidance — UODO opinions | PL | uodo.gov.pl |

### Why this corpus
It gives you deliberate cross-lingual query scenarios: someone asks in Ukrainian about GDPR (EU document, originally English), or in Polish about the Ukrainian personal data law. Those scenarios stress-test the system in a way a single-language corpus never would. And they are realistic — a Ukrainian person operating a business in Poland genuinely needs this kind of assistant.

### Corpus assembly script
```python
# Use this structure from day one — metadata discipline is what separates
# a real RAG from a demo RAG

from dataclasses import dataclass
from datetime import date

@dataclass
class DocumentMeta:
    source_id: str           # "eu_ai_act_2024"
    title: str
    language: str            # ISO 639-1: "en", "pl", "uk"
    doc_type: str            # "regulation", "act", "guidance"
    jurisdiction: str        # "EU", "PL", "UA"
    effective_date: date
    url: str
    page_count: int
```

---

## Phase 0 — Project Setup (Week 1)

**Goal:** clean scaffolding before any model code touches the repo.

### Repository structure
```
multilingual-rag/
├── data/
│   ├── raw/          # original PDFs, no git
│   └── processed/    # parsed text + metadata JSON, no git
├── src/
│   ├── ingestion/    # parsers, chunkers
│   ├── retrieval/    # embedding, vector store, BM25
│   ├── pipeline/     # RAG orchestration, reranker
│   ├── api/          # FastAPI app
│   └── eval/         # evaluation harness
├── frontend/         # Streamlit app
├── docker/
├── notebooks/        # exploration only, never production logic here
├── tests/
├── .env.example
├── docker-compose.yml
└── README.md
```

### Tooling setup
```bash
# Dependency management
uv init && uv add langchain langchain-community \
    qdrant-client sentence-transformers \
    rank-bm25 bm25s \
    openai anthropic \
    fastapi uvicorn \
    ragas streamlit \
    pypdf pymupdf pdfplumber \
    langdetect lingua-language-detector \
    python-dotenv pydantic loguru

# Code quality
uv add --dev ruff black pytest pytest-asyncio

# Pre-commit
pre-commit install
```

### Architecture Decision Record (write this before coding)
Create `docs/ADR.md` capturing three decisions with explicit reasoning:
1. Why Qdrant over ChromaDB (persistence, metadata filtering, hybrid search built-in)
2. Why `multilingual-e5-large` over `paraphrase-multilingual-mpnet-base-v2` (instruction-tuning advantage)
3. Why hybrid retrieval over pure dense (BM25 dominates on named-entity-heavy legal text where exact term matching matters more than semantic generalization)

Writing ADRs is what professional teams do. Including one in your repo is an immediately visible signal of engineering maturity.

---

## Phase 1 — Document Ingestion & Parsing (Week 2)

**Goal:** clean, metadata-rich text out of every source document.

### Parser selection logic
```python
def parse_document(path: str, meta: DocumentMeta) -> list[dict]:
    """
    PDFs with good text layer → pdfplumber (preserves layout better than pypdf)
    Scanned PDFs → raise NotImplementedError (out of scope, note it in README)
    HTML / web pages → trafilatura for clean main content extraction
    """
```

### Ukrainian-specific handling
Standard Python sentence tokenizers (spaCy `sent_tokenize`, nltk) will work poorly on Ukrainian unless you load the correct language model. Use `spacy` with `uk_core_news_sm` for Ukrainian, `pl_core_news_sm` for Polish, and the default for English.

```bash
python -m spacy download uk_core_news_sm
python -m spacy download pl_core_news_sm
python -m spacy download en_core_web_sm
```

### Language detection
Use `lingua-language-detector` rather than `langdetect` — it handles short snippets and Slavic scripts far more reliably.

```python
from lingua import Language, LanguageDetectorBuilder

detector = LanguageDetectorBuilder.from_languages(
    Language.ENGLISH, Language.POLISH, Language.UKRAINIAN
).build()

def detect_language(text: str) -> str:
    lang = detector.detect_language_of(text)
    return {Language.ENGLISH: "en", Language.POLISH: "pl", 
            Language.UKRAINIAN: "uk"}.get(lang, "en")
```

**Deliverable:** `data/processed/` populated, a `validate_corpus.py` script that prints per-language document counts and flags any chunk below 50 characters or above 2000 characters.

---

## Phase 2 — Chunking Strategy (Week 2, continued)

This is the step most tutorials get catastrophically wrong. Do not use a flat character splitter on legal documents.

### Recommended approach: hierarchical chunking

Legal documents have natural structure: Part → Chapter → Article → Paragraph → Sentence. Preserve it.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Level 1: split at article/section boundaries first (document-specific regex)
# Level 2: within each section, use RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,          # tokens, not characters
    chunk_overlap=64,
    separators=["\n\n", "\n", ". ", " "],
    length_function=token_counter,   # use tiktoken, not len()
)
```

### Parent-child chunking (implement this, it visually differentiates your project)
Store two granularities: a *child* chunk (256 tokens, for dense retrieval precision) and a *parent* chunk (the full article/paragraph, ~1024 tokens, for context quality). Retrieve by child similarity, but pass the parent to the LLM. This is what production systems like Cohere RAG and Llamacloud do internally.

```python
@dataclass
class Chunk:
    chunk_id: str
    parent_id: str           # links back to parent context
    text: str
    token_count: int
    language: str
    document_id: str
    article_number: str | None   # "Article 6", "Art. 22 ust. 1"
    section_title: str | None
    page: int | None
```

**Deliverable:** `chunking_analysis.ipynb` comparing chunk size distributions per language and justifying parameter choices. This notebook goes in your GitHub README as a link — it shows rigor, not just code.

---

## Phase 3 — Embedding & Vector Store (Week 3)

### Embedding model choice

Use `intfloat/multilingual-e5-large` (560M parameters). It outperforms `paraphrase-multilingual-mpnet-base-v2` on cross-lingual retrieval benchmarks by a significant margin and crucially supports instruction prefixes — essential for asymmetric retrieval (query vs passage are different distributions).

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-large")

# CRITICAL: use the instruction prefixes — this is the most commonly skipped step
def embed_passage(text: str) -> list[float]:
    return model.encode(f"passage: {text}", normalize_embeddings=True).tolist()

def embed_query(text: str) -> list[float]:
    return model.encode(f"query: {text}", normalize_embeddings=True).tolist()
```

### Vector store: Qdrant

Run it locally via Docker during development, same Docker image in production. No cognitive switch cost.

```yaml
# docker-compose.yml (partial)
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - ./qdrant_storage:/qdrant/storage
```

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient(host="localhost", port=6333)

client.create_collection(
    collection_name="legal_docs",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
)
```

Store all metadata fields as Qdrant payload — you'll filter on `language` and `jurisdiction` later.

### BM25 index

Build a parallel BM25 index using `bm25s` (significantly faster than `rank_bm25`). Index the same chunks, keyed by the same `chunk_id`.

**Deliverable:** ingestion script that populates both indexes, reports total chunks per language, and a smoke-test notebook confirming top-5 retrieval results on 10 hand-picked queries.

---

## Phase 4 — Core RAG Pipeline (Week 4)

### Hybrid retrieval with Reciprocal Rank Fusion

```python
def hybrid_retrieve(query: str, k: int = 20) -> list[Chunk]:
    query_vec = embed_query(query)
    
    dense_results = qdrant_search(query_vec, top_k=k)      # returns [(chunk_id, score)]
    bm25_results = bm25_search(query, top_k=k)             # returns [(chunk_id, score)]
    
    # RRF — position matters, not raw score
    rrf_scores: dict[str, float] = {}
    for rank, (chunk_id, _) in enumerate(dense_results):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (60 + rank)
    for rank, (chunk_id, _) in enumerate(bm25_results):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (60 + rank)
    
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [fetch_chunk(chunk_id) for chunk_id, _ in ranked[:k]]
```

### Cross-encoder reranking

After RRF gives you 20 candidates, rerank to top 5 with a cross-encoder. This is where precision actually comes from.

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, chunks: list[Chunk], top_k: int = 5) -> list[Chunk]:
    pairs = [(query, chunk.text) for chunk in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in ranked[:top_k]]
```

### Language-aware generation prompt

```python
SYSTEM_PROMPT = """You are a multilingual legal assistant specializing in EU and Central/Eastern European regulatory documents. 
You answer in the SAME LANGUAGE as the user's question — always match their language exactly.
Base your answer ONLY on the provided context. 
Every claim must be followed by a citation in the format [Source: {document_title}, {article_reference}].
If the context does not contain sufficient information, say so explicitly — do not infer or extrapolate."""

def build_prompt(query: str, chunks: list[Chunk]) -> list[dict]:
    context = "\n\n---\n\n".join([
        f"[{c.document_id} | {c.article_number or 'Section'} | lang:{c.language}]\n{c.text}"
        for c in chunks
    ])
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
```

### Cost management (non-optional)
Implement a caching layer from day one. Hash the (query, retrieved_chunk_ids) tuple as a cache key. Store responses in SQLite. Without this you'll burn through API credits running evals.

```python
import hashlib, json, sqlite3

def cache_key(query: str, chunk_ids: list[str]) -> str:
    payload = json.dumps({"q": query, "ids": sorted(chunk_ids)})
    return hashlib.sha256(payload.encode()).hexdigest()
```

**Deliverable:** `pipeline/rag_chain.py` with a single `answer(query: str) -> RAGResponse` function, where `RAGResponse` is a Pydantic model containing `text`, `language`, `sources`, `chunk_ids`, and `latency_ms`.

---

## Phase 5 — Query Intelligence (Week 4, continued)

Two additions that dramatically improve retrieval quality and are rarely included in junior projects.

### HyDE (Hypothetical Document Embeddings)

For complex queries, generate a hypothetical answer using the LLM (without retrieval), embed that answer, and use it as the retrieval vector. The hypothesis embedding often sits closer to relevant documents than the raw query embedding.

```python
async def hyde_query_vector(query: str) -> list[float]:
    hypothesis = await llm.complete(
        f"Write a one-paragraph excerpt from a legal regulation that would directly answer: {query}"
    )
    return embed_passage(hypothesis)   # embed as passage, not query
```

### Query decomposition for compound questions

"What are the GDPR rules on data retention and how do they differ in Polish and Ukrainian law?" — this is actually three sub-questions. Decompose, retrieve separately, merge.

```python
async def decompose_query(query: str) -> list[str]:
    # LLM call returns JSON list of sub-questions
    ...
```

Only implement HyDE in Phase 5; decomposition is a stretch goal for later.

---

## Phase 6 — FastAPI Backend (Week 5)

### Endpoints

```python
# src/api/main.py

POST /api/chat
  Body: { "query": str, "session_id": str | None, "filter_language": str | None }
  Response: StreamingResponse (SSE)

GET /api/sources
  Response: list of all indexed documents with metadata

POST /api/ingest
  Body: uploaded PDF + metadata
  Response: ingestion status (stretch goal for demo)

GET /api/health
  Response: { "status": "ok", "indexed_chunks": int, "languages": list[str] }
```

### Streaming implementation

Use Server-Sent Events — this is what makes the UI feel like a real product, not a slow API call.

```python
from fastapi.responses import StreamingResponse

@app.post("/api/chat")
async def chat(request: ChatRequest):
    async def event_stream():
        chunks = await retrieve_and_rerank(request.query)
        async for token in llm.stream(build_prompt(request.query, chunks)):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield f"data: {json.dumps({'sources': [c.to_dict() for c in chunks], 'done': True})}\n\n"
    
    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

### Structured logging

Use `loguru`. Log every request with: query language, detected language, retrieval latency, reranking latency, generation latency, total latency, number of chunks retrieved, source document IDs. This data feeds the performance analysis section of your README.

---

## Phase 7 — Streamlit Frontend (Week 6)

Start with Streamlit for speed. It will be enough for the portfolio demo and can always be replaced later.

### Features to implement (in priority order)
1. Chat interface with streaming token display
2. Sidebar showing retrieved source documents with language flag emoji (🇪🇺🇵🇱🇺🇦🇬🇧), title, article number
3. Language selector (auto-detect vs manual override)
4. Query examples in all three languages so reviewers can test it without thinking
5. Session history (last 5 exchanges)

### UX detail that matters for portfolio impression
When displaying citations, show a colour-coded badge per source language — a reviewer can visually confirm within 5 seconds that the system is genuinely cross-lingual. Without this visual proof, they might think you're faking it.

**Deliverable:** `docker compose up` starts Qdrant + API + Streamlit. No manual setup steps. Test this yourself on a clean machine or Docker environment before submitting applications.

---

## Phase 8 — Evaluation Framework (Week 7)

This is the phase that turns a "cool project" into a "serious candidate" signal. No junior has an eval suite. You will.

### Test set construction

Build 50+ question-answer pairs by hand across three languages. Structure:

```python
@dataclass
class EvalItem:
    question: str
    question_language: str           # "en", "pl", "uk"
    ground_truth_answer: str
    ground_truth_language: str       # should match question_language
    relevant_chunk_ids: list[str]    # which chunks should be retrieved
    relevant_document_ids: list[str]
    difficulty: str                  # "simple", "cross-lingual", "multi-hop"
```

Include all three difficulty tiers:
- **Simple:** question and answer in same language, single document
- **Cross-lingual:** question in Ukrainian, answer drawn from Polish regulation
- **Multi-hop:** question requires synthesizing two different documents

### Metrics with ragas

```python
from ragas import evaluate
from ragas.metrics import (
    context_precision,    # are retrieved chunks actually relevant?
    context_recall,       # did we retrieve all relevant chunks?
    faithfulness,         # does the answer stick to the context?
    answer_relevancy,     # does the answer actually address the question?
)
```

### Baselines to compare against (this is the differentiator)

| Configuration | Context Precision | Context Recall | Faithfulness | Answer Relevancy |
|---|---|---|---|---|
| Dense only, no reranking | — | — | — | — |
| Dense + BM25 hybrid (RRF) | — | — | — | — |
| Hybrid + cross-encoder reranking | — | — | — | — |
| Hybrid + reranking + HyDE | — | — | — | — |

Fill in real numbers from your runs. This table belongs in your README and in your blog post. Interviewers will ask "how did you measure that it works?" — point here.

### Cost tracking

Log per-query token usage and compute cumulative cost for the full eval run. Include it in the README: "Full eval suite (50 queries) costs approximately $0.38 at current GPT-4o-mini pricing." Showing you think about cost signals engineering maturity.

---

## Phase 9 — Containerization & Deployment (Week 8)

### Docker Compose (complete)

```yaml
version: "3.9"

services:
  qdrant:
    image: qdrant/qdrant:v1.9.0
    volumes:
      - qdrant_data:/qdrant/storage
    ports:
      - "6333:6333"
    
  api:
    build: ./docker/api
    env_file: .env
    ports:
      - "8000:8000"
    depends_on:
      - qdrant
    volumes:
      - ./data/processed:/app/data
    
  frontend:
    build: ./docker/frontend
    ports:
      - "8501:8501"
    depends_on:
      - api

volumes:
  qdrant_data:
```

### Deployment options (pick one)

**Hugging Face Spaces** — zero cost, shareable link, high recruiter recognition. Use the Docker SDK space type. Only caveat: no persistent Qdrant volume, so you'll need to pre-build and include the vector store snapshot or use Qdrant Cloud free tier.

**Qdrant Cloud free tier (1GB)** — keeps the vector store persistent across restarts. Combine with HF Spaces for compute. This is the cleanest production-like setup.

**Azure free tier** — better for Polish enterprise recruiters who see Azure as signal of fit. More setup overhead.

---

## Phase 10 — Documentation & Polish (Week 8–9)

### README sections (non-negotiable)
1. One-paragraph plain-language description of the problem and who it helps
2. Architecture diagram (draw with Excalidraw or Mermaid, export as PNG)
3. Retrieval pipeline diagram (the one at the top of this plan)
4. Evaluation results table with all four configurations
5. Corpus inventory table (what documents, what languages, what jurisdictions)
6. Setup: `git clone` → `docker compose up` in three commands
7. Query examples in all three languages with screenshots of actual responses
8. "What I would do with more time" — 3 bullets, be honest

### Blog post (400–600 words on Medium or LinkedIn)
Structure: problem → approach → key technical decision (hybrid retrieval) → one surprising finding from evaluation → link to demo. Publish before your first applications go out. Link it from the GitHub README.

### Demo video
90 seconds. No narration needed — just show: a Polish question → Ukrainian-sourced answer with citation, then an English question → Polish-sourced answer. Let the cross-linguality speak for itself.

---

## Common Pitfalls

**Chunking ignores language boundary — fixed Unicode normalization:** Polish and Ukrainian use different unicode normalization conventions. Run `unicodedata.normalize("NFC", text)` on all text before chunking.

**Embeddings not instruction-prefixed:** the `multilingual-e5` model degrades significantly without `"query: "` / `"passage: "` prefixes. Easy to miss, hard to diagnose later.

**Retrieval evaluated on accuracy but not latency:** measure p50 and p95 latency for the full pipeline. A system that takes 12 seconds per query is not deployable. Target under 3 seconds end-to-end including streaming first token.

**HyDE adds hallucination risk:** if the LLM generates a false hypothesis, you retrieve false context. Always run HyDE as an A/B option, not the only retrieval path.

**Mixing parent and child chunk embeddings in the same collection:** creates retrieval confusion. Keep them in separate Qdrant collections or use a `chunk_type` metadata filter.

**No versioning of the vector store:** if you change your chunking strategy mid-project, you need to re-ingest everything. Tag your Qdrant snapshots with a hash of the chunking config.

---

## Milestone Summary

| Week | Phase | Key Deliverable |
|---|---|---|
| 1 | Setup | Repo, ADR, Docker skeleton, all dependencies locked |
| 2 | Ingestion | Parsed corpus, metadata-tagged chunks, language-validated |
| 3 | Embedding | Both indexes populated, smoke-test retrieval notebook |
| 4 | Core pipeline | `answer()` function, caching, cost logging |
| 4 | Query intelligence | HyDE integrated as optional pipeline mode |
| 5 | API | FastAPI with SSE streaming, health endpoint |
| 6 | Frontend | Streamlit UI with citation display and language badges |
| 7 | Evaluation | 50-item eval set, ragas metrics, 4-config comparison table |
| 8 | Deploy | Docker Compose, HF Spaces or Azure, shareable URL |
| 9 | Polish | README complete, blog post published, demo video recorded |

---

## Tech Stack Summary

| Layer | Tool | Why |
|---|---|---|
| Orchestration | LangChain (core) + custom pipeline | Flexibility without over-coupling |
| Embeddings | `intfloat/multilingual-e5-large` | Best cross-lingual retrieval quality at manageable size |
| Vector store | Qdrant | Hybrid search built-in, Docker-native, production-grade |
| BM25 | bm25s | 10–100x faster than rank_bm25 |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Good quality/latency tradeoff |
| LLM | GPT-4o-mini (default) / Claude Haiku (alternative) | Cost-efficient, switch via env var |
| Evaluation | ragas | Industry standard for RAG eval |
| API | FastAPI + uvicorn | Async, SSE streaming support |
| Frontend | Streamlit | Fast for portfolio, good enough for demo |
| Containerization | Docker + Docker Compose | Reproducibility, portfolio standard |
| Language detection | lingua-language-detector | Far more accurate than langdetect on Slavic scripts |
| PDF parsing | pdfplumber (primary), PyMuPDF (fallback) | pdfplumber preserves layout better |
| Dependency management | uv | Modern, fast, reproducible |
| Code quality | ruff + black + pre-commit | Professional repo hygiene |
