"""FastAPI app: SSE chat, corpus listing, health.

Privacy (CLAUDE.md): full query text is never logged — only a hash, length,
and detected language. Latency breakdown per stage is logged for every chat
request; this feeds the performance section of the README.

Run: uv run uvicorn src.api.main:app --port 8000
"""

import hashlib
import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from loguru import logger

from src.api.schemas import ChatRequest, Health, SourceDocument
from src.config import DATA_PROCESSED, INDEX_DIR


def _load_sources() -> list[SourceDocument]:
    sources = []
    for path in sorted(DATA_PROCESSED.glob("*.json")):
        if path.name == "chunks.json":
            continue
        doc = json.loads(path.read_text())
        meta = doc["meta"]
        sources.append(
            SourceDocument(
                source_id=meta["source_id"],
                title=meta["title"],
                language=meta["language"],
                jurisdiction=meta["jurisdiction"],
                doc_type=meta["doc_type"],
                official=meta["official"],
                url=meta["url"],
                effective_date=meta["effective_date"],
                page_count=meta["page_count"],
                units=len(doc["units"]),
            )
        )
    return sources


def create_app(pipeline=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if pipeline is not None:
            app.state.pipeline = pipeline
        else:  # warm everything at startup, not on the first request
            from src.pipeline.rag_chain import get_pipeline

            app.state.pipeline = get_pipeline()
            logger.info("pipeline warmed: retriever, reranker, llm ready")
        app.state.sources = _load_sources()
        try:
            yield
        finally:
            if pipeline is None:
                from src.retrieval.qdrant_store import get_client

                get_client().close()

    app = FastAPI(title="multilingual-rag", lifespan=lifespan)

    @app.post("/api/chat")
    async def chat(request: Request, body: ChatRequest) -> StreamingResponse:
        query_id = hashlib.sha256(body.query.encode()).hexdigest()[:12]

        def event_stream():
            t0 = time.perf_counter()
            final: dict = {}
            try:
                for event in request.app.state.pipeline.stream_answer(
                    body.query,
                    filter_language=body.filter_language,
                    cross_lingual=body.cross_lingual,
                    use_hyde=body.use_hyde,
                ):
                    if event.get("done"):
                        final = event
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception:
                logger.exception("chat failed query_id={}", query_id)
                err = {"error": "generation failed — see server logs", "done": True}
                yield f"data: {json.dumps(err)}\n\n"
                return
            logger.info(
                "chat query_id={} len={} session={} lang={} filter={} cross_lingual={} "
                "hyde={} chunks={} sources={} retrieval_ms={} rerank_ms={} "
                "generation_ms={} total_ms={}",
                query_id,
                len(body.query),
                body.session_id,
                final.get("language"),
                body.filter_language,
                body.cross_lingual,
                body.use_hyde,
                len(final.get("chunk_ids", [])),
                sorted({s["source_id"] for s in final.get("sources", [])}),
                final.get("retrieval_ms"),
                final.get("rerank_ms"),
                final.get("generation_ms"),
                int((time.perf_counter() - t0) * 1000),
            )

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/api/sources")
    async def sources(request: Request) -> list[SourceDocument]:
        return request.app.state.sources

    @app.get("/api/health")
    async def health() -> Health:
        meta = json.loads((INDEX_DIR / "index_meta.json").read_text())
        return Health(
            status="ok",
            indexed_chunks=meta["chunks_indexed"],
            languages=sorted(meta["per_language"]),
            collection=meta["collection"],
            embedding_model=meta["embedding_model"],
        )

    return app


app = create_app()
