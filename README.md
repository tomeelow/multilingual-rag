# Multilingual RAG — EU / Polish / Ukrainian AI & Data-Protection Law

A retrieval-augmented assistant for AI regulatory compliance and data
protection law across EU, Polish, and Ukrainian jurisdictions. Ask in
English, Polish, or Ukrainian; answers cite specific articles and recitals
from the indexed corpus, regardless of the source document's language.

> Work in progress — built phase by phase per
> [docs/multilingual_rag_plan.md](docs/multilingual_rag_plan.md).
> Architecture decisions: [docs/ADR.md](docs/ADR.md).

## Setup

```bash
git clone <repo>
cd multilingual-rag
uv sync
```

Full stack (Qdrant + API + frontend) via `docker compose up` — completed in
Phase 9.
