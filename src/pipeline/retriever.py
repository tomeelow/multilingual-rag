"""Hybrid retrieval: dense (Qdrant) + lexical (BM25) fused by Reciprocal
Rank Fusion, returning child chunks for reranking and parent chunks for
generation context (ADR-3).

Cross-lingual retrieval is opt-in: by default a query retrieves only from
sources in its own language; `cross_lingual=True` opens all languages
(the dense leg then bridges e.g. a Ukrainian question to the English AI Act).
"""

from functools import lru_cache

from src.config import pipeline_config
from src.ingestion.chunker import load_chunks
from src.models import Chunk
from src.pipeline.query_analysis import article_base_number
from src.retrieval import qdrant_store
from src.retrieval.bm25_index import BM25Index
from src.retrieval.embedding import get_embedder


def rrf_fuse(rankings: list[list[tuple[str, float]]], rrf_k: int = 60) -> list[tuple[str, float]]:
    """Fuse result lists by rank position — raw scores from different legs
    live on incomparable scales and are ignored deliberately."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (chunk_id, _) in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class Retriever:
    def __init__(self) -> None:
        cfg = pipeline_config()["retrieval"]
        self.candidates_k: int = cfg["candidates_k"]
        self.rrf_k: int = cfg["rrf_k"]
        self._children = {c.chunk_id: c for c in load_chunks("child")}
        self._parents = {c.chunk_id: c for c in load_chunks("parent")}
        self._bm25 = BM25Index.load()
        self._qdrant = qdrant_store.get_client()

    def dense(
        self,
        query: str,
        k: int | None = None,
        language: str | None = None,
        source_ids: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        vector = get_embedder().embed_query(query)
        return self.dense_by_vector(vector, k, language, source_ids)

    def dense_by_vector(
        self,
        vector: list[float],
        k: int | None = None,
        language: str | None = None,
        source_ids: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        return qdrant_store.search(
            self._qdrant, vector, k or self.candidates_k, language, source_ids
        )

    def bm25(
        self,
        query: str,
        k: int | None = None,
        language: str | None = None,
        source_ids: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        k = k or self.candidates_k
        # post-filtering needs headroom: a small document (the UA law is 4%
        # of the corpus) may sit far down the unfiltered ranking
        multiplier = 10 if source_ids else (3 if language else 1)
        results = self._bm25.search(query, k * multiplier)
        if language:
            results = [r for r in results if self._children[r[0]].language == language]
        if source_ids:
            results = [r for r in results if self._children[r[0]].source_id in source_ids]
        return results[:k]

    def hybrid(
        self,
        query: str,
        k: int | None = None,
        language: str | None = None,
        cross_lingual: bool = False,
        query_vector: list[float] | None = None,
        source_ids: list[str] | None = None,
    ) -> list[Chunk]:
        """Top-k child chunks by RRF over both legs."""
        k = k or self.candidates_k
        lang_filter = None if cross_lingual else language
        if query_vector is None:
            dense = self.dense(query, k, lang_filter, source_ids)
        else:
            dense = self.dense_by_vector(query_vector, k, lang_filter, source_ids)
        sparse = self.bm25(query, k, lang_filter, source_ids)
        fused = rrf_fuse([dense, sparse], self.rrf_k)
        return [self._children[cid] for cid, _ in fused[:k]]

    def children_in_range(
        self,
        lo: int,
        hi: int,
        language: str | None = None,
        prefer_sources: list[str] | None = None,
        limit: int = 40,
    ) -> list[Chunk]:
        """All child article chunks with base number in [lo, hi], straight
        from metadata — articles the semantic legs missed must still enter
        the rerank pool when the query names an explicit range.

        `prefer_sources` orders the result (documents the query already
        retrieves first), so the `limit` cut drops the least likely
        documents, not arbitrary ones.
        """
        matches = [
            c
            for c in self._children.values()
            if c.kind == "article"
            and (base := article_base_number(c.article_number)) is not None
            and lo <= base <= hi
            and (language is None or c.language == language)
        ]
        rank = {s: i for i, s in enumerate(prefer_sources or [])}
        matches.sort(
            key=lambda c: (
                rank.get(c.source_id, len(rank)),
                article_base_number(c.article_number),
                c.chunk_id,
            )
        )
        return matches[:limit]

    def parents_of(self, children: list[Chunk]) -> list[Chunk]:
        """Parent context chunks, deduplicated, in child-ranking order."""
        seen: dict[str, Chunk] = {}
        for child in children:
            if child.parent_id not in seen:
                seen[child.parent_id] = self._parents[child.parent_id]
        return list(seen.values())


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()
