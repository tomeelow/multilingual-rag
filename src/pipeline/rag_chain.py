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
from src.pipeline.query_analysis import in_article_range, parse_article_range
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
    retrieval_only: bool = False


_RETRIEVAL_ONLY_NOTICE = {
    "en": (
        "Retrieval-only mode: no external LLM is configured. Below are the most relevant "
        "source excerpts. This is not a generated legal answer or translation."
    ),
    "pl": (
        "Tryb tylko wyszukiwania: nie skonfigurowano zewnętrznego modelu LLM. Poniżej "
        "znajdują się najtrafniejsze fragmenty źródeł. Nie jest to wygenerowana odpowiedź "
        "prawna ani tłumaczenie."
    ),
    "uk": (
        "Режим лише пошуку: зовнішню модель LLM не налаштовано. Нижче наведено "
        "найрелевантніші уривки з джерел. Це не згенерована юридична відповідь і не переклад."
    ),
}


def _retrieval_only_text(language: str, parents: list[Chunk]) -> str:
    blocks = [_RETRIEVAL_ONLY_NOTICE.get(language, _RETRIEVAL_ONLY_NOTICE["en"])]
    for chunk in parents[:3]:
        excerpt = chunk.text.strip()
        if len(excerpt) > 1200:
            excerpt = excerpt[:1200].rsplit(" ", 1)[0] + "…"
        blocks.append(f"### {chunk.doc_title} — {chunk.ref or chunk.kind}\n\n{excerpt}")
    return "\n\n".join(blocks)


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
        article_range = parse_article_range(query)
        if article_range:
            lang_filter = None if cross_lingual else language  # mirrors Retriever.hybrid
            top_children = self._rerank_within_range(query, candidates, article_range, lang_filter)
        else:
            top_children = self.reranker.rerank(query, candidates)
        t2 = time.perf_counter()
        parents = self.retriever.parents_of(top_children)
        return top_children, parents, int((t1 - t0) * 1000), int((t2 - t1) * 1000)

    def _rerank_within_range(
        self,
        query: str,
        candidates: list[Chunk],
        article_range: tuple[int, int],
        language: str | None,
    ) -> list[Chunk]:
        """An explicit range ("art. 10–18") is a hard constraint the
        semantic legs cannot honor: in-range articles missing from the
        candidates are injected from metadata, and out-of-range candidates
        can only fill leftover slots, never outrank in-range ones."""
        lo, hi = article_range
        in_range = [c for c in candidates if in_article_range(c, lo, hi)]
        seen_sources: list[str] = []
        for c in candidates:
            if c.source_id not in seen_sources:
                seen_sources.append(c.source_id)
        seen_ids = {c.chunk_id for c in in_range}
        injected = [
            c
            for c in self.retriever.children_in_range(lo, hi, language, seen_sources)
            if c.chunk_id not in seen_ids
        ]
        top = self.reranker.rerank(query, in_range + injected)
        if len(top) < self.reranker.final_k:
            out_of_range = [c for c in candidates if not in_article_range(c, lo, hi)]
            top += self.reranker.rerank(query, out_of_range, top_k=self.reranker.final_k - len(top))
        return top

    def answer(
        self,
        query: str,
        filter_language: str | None = None,
        cross_lingual: bool = False,
        use_hyde: bool = False,
        retrieval_only: bool = False,
    ) -> RAGResponse:
        t0 = time.perf_counter()
        language = filter_language or detect_language(query)
        children, parents, retrieval_ms, rerank_ms = self.retrieve(
            query, language, cross_lingual, use_hyde
        )
        t1 = time.perf_counter()
        if retrieval_only:
            result = None
            text = _retrieval_only_text(language, parents)
        else:
            result = self.llm.complete(
                build_prompt(query, parents),
                max_tokens=self.gen_cfg["max_tokens"],
                temperature=self.gen_cfg["temperature"],
            )
            text = result.text
        t2 = time.perf_counter()
        return RAGResponse(
            text=text,
            language=language,
            sources=_sources(parents),
            chunk_ids=[c.chunk_id for c in children],
            latency_ms=int((t2 - t0) * 1000),
            retrieval_ms=retrieval_ms,
            rerank_ms=rerank_ms,
            generation_ms=int((t2 - t1) * 1000),
            cached=result.cached if result else False,
            input_tokens=result.input_tokens if result else 0,
            output_tokens=result.output_tokens if result else 0,
            retrieval_only=retrieval_only,
        )

    def stream_answer(
        self,
        query: str,
        filter_language: str | None = None,
        cross_lingual: bool = False,
        use_hyde: bool = False,
        retrieval_only: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Yields {"token": str} events, then one final {"done": True, ...}
        event with sources and latency metadata."""
        t0 = time.perf_counter()
        language = filter_language or detect_language(query)
        children, parents, retrieval_ms, rerank_ms = self.retrieve(
            query, language, cross_lingual, use_hyde
        )
        t1 = time.perf_counter()
        if retrieval_only:
            text = _retrieval_only_text(language, parents)
            for i in range(0, len(text), 48):
                yield {"token": text[i : i + 48]}
        else:
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
            "retrieval_only": retrieval_only,
        }


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


def answer(query: str, **kwargs) -> RAGResponse:
    return get_pipeline().answer(query, **kwargs)
