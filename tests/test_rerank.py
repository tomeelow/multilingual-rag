"""Reranker post-processing: per-unit deduplication and intent weighting.

The cross-encoder itself is replaced by a fake — these tests cover what the
Reranker does around the model scores."""

import numpy as np

from src.models import Chunk
from src.pipeline.rerank import Reranker


class _FakeCrossEncoder:
    """Scores each pair by a number embedded in the chunk text."""

    def predict(self, pairs):
        return np.array([float(text.split("#")[-1]) for _, text in pairs])


def _chunk(chunk_id: str, score: float, **kw) -> Chunk:
    defaults = dict(
        parent_id=chunk_id.rsplit(":", 1)[0],
        chunk_type="child",
        text=f"tekst #{score}",
        token_count=5,
        source_id=chunk_id.split(":")[0],
        doc_title="Ustawa testowa",
        language="pl",
        jurisdiction="PL",
        doc_type="act",
        official=True,
        url="https://example.invalid",
        ingestion_date="2026-06-11",
        kind="article",
        article_number=chunk_id.split(":")[2],
        ref=None,
        section_title=None,
        pages=[1, 1],
    )
    return Chunk(chunk_id=chunk_id, **(defaults | kw))


def _reranker() -> Reranker:
    r = Reranker()
    r._model = _FakeCrossEncoder()
    return r


def test_duplicate_units_collapse_to_best_chunk():
    chunks = [
        _chunk("doc:art:6:c0", 0.9),
        _chunk("doc:art:6:c1", 0.8),  # same article -> dropped
        _chunk("doc:art:7:c0", 0.7),
        _chunk("doc:art:2:c0", 0.6),
    ]
    top = _reranker().rerank("zapytanie", chunks, top_k=3)
    assert [c.chunk_id for c in top] == ["doc:art:6:c0", "doc:art:7:c0", "doc:art:2:c0"]


def test_same_number_different_kind_or_document_survives():
    chunks = [
        _chunk("doc:art:6:c0", 0.9),
        _chunk("doc:rec:6:c0", 0.8, kind="recital"),  # recital 6 != article 6
        _chunk("inny:art:6:c0", 0.7),  # other document
    ]
    top = _reranker().rerank("zapytanie", chunks, top_k=3)
    assert len(top) == 3


def test_unnumbered_chunks_dedup_by_parent():
    chunks = [
        _chunk("doc:par:0:c0", 0.9, article_number=None, kind="paragraph"),
        _chunk("doc:par:0:c1", 0.8, article_number=None, kind="paragraph"),
        _chunk("doc:par:1:c0", 0.7, article_number=None, kind="paragraph"),
    ]
    top = _reranker().rerank("zapytanie", chunks, top_k=3)
    assert [c.chunk_id for c in top] == ["doc:par:0:c0", "doc:par:1:c0"]
