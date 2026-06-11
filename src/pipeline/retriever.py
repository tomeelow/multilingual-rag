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
        self, query: str, k: int | None = None, language: str | None = None
    ) -> list[tuple[str, float]]:
        vector = get_embedder().embed_query(query)
        return self.dense_by_vector(vector, k, language)

    def dense_by_vector(
        self, vector: list[float], k: int | None = None, language: str | None = None
    ) -> list[tuple[str, float]]:
        return qdrant_store.search(self._qdrant, vector, k or self.candidates_k, language)

    def bm25(
        self, query: str, k: int | None = None, language: str | None = None
    ) -> list[tuple[str, float]]:
        results = self._bm25.search(query, (k or self.candidates_k) * (3 if language else 1))
        if language:
            results = [r for r in results if self._children[r[0]].language == language]
        return results[: k or self.candidates_k]

    def hybrid(
        self,
        query: str,
        k: int | None = None,
        language: str | None = None,
        cross_lingual: bool = False,
        query_vector: list[float] | None = None,
    ) -> list[Chunk]:
        """Top-k child chunks by RRF over both legs."""
        k = k or self.candidates_k
        lang_filter = None if cross_lingual else language
        if query_vector is None:
            dense = self.dense(query, k, lang_filter)
        else:
            dense = self.dense_by_vector(query_vector, k, lang_filter)
        sparse = self.bm25(query, k, lang_filter)
        fused = rrf_fuse([dense, sparse], self.rrf_k)
        return [self._children[cid] for cid, _ in fused[:k]]

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
