"""Cross-encoder reranking of RRF candidates — this is where precision
comes from. The model scores (query, chunk) pairs jointly instead of
comparing pre-computed vectors.

Cross-encoder scores are sigmoid-normalized, then adjusted by query intent:
on a rights question, chunks from an obligations chapter are demoted (and
vice versa) — semantic similarity alone ranks "employee is obliged to…"
top for "what are the employee's basic rights" because polarity is
invisible to topical closeness (see src.pipeline.query_analysis)."""

import math
from functools import lru_cache

from sentence_transformers import CrossEncoder

from src.config import pipeline_config
from src.models import Chunk
from src.pipeline.query_analysis import detect_rights_duties_intent, intent_weight


class Reranker:
    def __init__(self, model_name: str | None = None) -> None:
        cfg = pipeline_config()["retrieval"]
        self.model_name = model_name or cfg["reranker"]
        self.final_k: int = cfg["final_k"]
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, chunks: list[Chunk], top_k: int | None = None) -> list[Chunk]:
        if not chunks:
            return []
        raw = self.model.predict([(query, c.text) for c in chunks])
        intent = detect_rights_duties_intent(query)
        # sigmoid first: intent multipliers need scores on a positive scale,
        # raw cross-encoder logits can be negative
        scores = [
            1.0 / (1.0 + math.exp(-float(s))) * intent_weight(intent, c)
            for c, s in zip(chunks, raw, strict=True)
        ]
        ranked = sorted(zip(chunks, scores, strict=True), key=lambda x: x[1], reverse=True)
        return _dedup([c for c, _ in ranked])[: top_k or self.final_k]


def _dedup(chunks: list[Chunk]) -> list[Chunk]:
    """One result slot per legal unit: two child chunks of GDPR Article 6
    say less than Article 6 plus Article 7. Keyed by (source, kind, number)
    — recital 6 and article 6 of the same document are distinct units."""
    seen: set[tuple] = set()
    out: list[Chunk] = []
    for c in chunks:
        key = (c.source_id, c.kind, c.article_number or c.parent_id)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker()
