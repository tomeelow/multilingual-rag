"""RAG orchestration: retrieve -> rerank -> generate, with latency breakdown.

`answer()` is the single synchronous entry point; `stream_answer()` yields
tokens for the SSE endpoint and reports the same metadata at the end.
"""

import time
from collections.abc import Iterator
from functools import lru_cache
from typing import Any

from pydantic import BaseModel

from src.config import pipeline_config
from src.language import detect_language
from src.models import Chunk
from src.pipeline.hyde import hyde_query_vector
from src.pipeline.llm import LLMClient, get_llm
from src.pipeline.prompts import build_prompt
from src.pipeline.rerank import Reranker, get_reranker
from src.pipeline.retriever import Retriever, get_retriever


class Source(BaseModel):
    source_id: str
    doc_title: str
    ref: str | None
    language: str
    jurisdiction: str
    official: bool
    url: str
    pages: list[int]


class RAGResponse(BaseModel):
    text: str
    language: str
    sources: list[Source]
    chunk_ids: list[str]
    latency_ms: int
    retrieval_ms: int
    rerank_ms: int
    generation_ms: int
    cached: bool
    input_tokens: int
    output_tokens: int


def _sources(parents: list[Chunk]) -> list[Source]:
    return [
        Source(
            source_id=c.source_id,
            doc_title=c.doc_title,
            ref=c.ref,
            language=c.language,
            jurisdiction=c.jurisdiction,
            official=c.official,
            url=c.url,
            pages=c.pages,
        )
        for c in parents
    ]


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever | None = None,
        reranker: Reranker | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self.retriever = retriever or get_retriever()
        self.reranker = reranker or get_reranker()
        self.llm = llm or get_llm()
        self.gen_cfg = pipeline_config()["generation"]

    def retrieve(
        self, query: str, language: str, cross_lingual: bool, use_hyde: bool = False
    ) -> tuple[list[Chunk], list[Chunk], int, int]:
        """-> (reranked child chunks, parent context chunks,
        retrieval_ms, rerank_ms)"""
        t0 = time.perf_counter()
        # HyDE replaces the dense query vector only; BM25 keeps the raw query
        # and the reranker scores against the real query, so a hallucinated
        # hypothesis cannot poison the lexical leg or the final ranking
        query_vector = hyde_query_vector(query, self.llm) if use_hyde else None
        candidates = self.retriever.hybrid(
            query, language=language, cross_lingual=cross_lingual, query_vector=query_vector
        )
        t1 = time.perf_counter()
        top_children = self.reranker.rerank(query, candidates)
        t2 = time.perf_counter()
        parents = self.retriever.parents_of(top_children)
        return top_children, parents, int((t1 - t0) * 1000), int((t2 - t1) * 1000)

    def answer(
        self,
        query: str,
        filter_language: str | None = None,
        cross_lingual: bool = False,
        use_hyde: bool = False,
    ) -> RAGResponse:
        t0 = time.perf_counter()
        language = filter_language or detect_language(query)
        children, parents, retrieval_ms, rerank_ms = self.retrieve(
            query, language, cross_lingual, use_hyde
        )
        t1 = time.perf_counter()
        result = self.llm.complete(
            build_prompt(query, parents),
            max_tokens=self.gen_cfg["max_tokens"],
            temperature=self.gen_cfg["temperature"],
        )
        t2 = time.perf_counter()
        return RAGResponse(
            text=result.text,
            language=language,
            sources=_sources(parents),
            chunk_ids=[c.chunk_id for c in children],
            latency_ms=int((t2 - t0) * 1000),
            retrieval_ms=retrieval_ms,
            rerank_ms=rerank_ms,
            generation_ms=int((t2 - t1) * 1000),
            cached=result.cached,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )

    def stream_answer(
        self,
        query: str,
        filter_language: str | None = None,
        cross_lingual: bool = False,
        use_hyde: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Yields {"token": str} events, then one final {"done": True, ...}
        event with sources and latency metadata."""
        t0 = time.perf_counter()
        language = filter_language or detect_language(query)
        children, parents, retrieval_ms, rerank_ms = self.retrieve(
            query, language, cross_lingual, use_hyde
        )
        t1 = time.perf_counter()
        for token in self.llm.stream(
            build_prompt(query, parents),
            max_tokens=self.gen_cfg["max_tokens"],
            temperature=self.gen_cfg["temperature"],
        ):
            yield {"token": token}
        t2 = time.perf_counter()
        yield {
            "done": True,
            "language": language,
            "sources": [s.model_dump() for s in _sources(parents)],
            "chunk_ids": [c.chunk_id for c in children],
            "latency_ms": int((t2 - t0) * 1000),
            "retrieval_ms": retrieval_ms,
            "rerank_ms": rerank_ms,
            "generation_ms": int((t2 - t1) * 1000),
        }


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


def answer(query: str, **kwargs) -> RAGResponse:
    return get_pipeline().answer(query, **kwargs)
