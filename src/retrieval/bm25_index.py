"""BM25 index over the same child chunks, keyed by the same chunk_ids.

bm25s over rank_bm25 (ADR-3 / tech-stack decision): 10-100x faster scoring.
Tokenization is a lowercase unicode word split — no stemmer, because no
stemmer covers all three corpus languages and legal retrieval leans on exact
tokens ("RODO", "2016/679", "18³a") which stemming would mangle.
"""

import json
import re
from pathlib import Path

import bm25s

from src.config import INDEX_DIR

_WORD = re.compile(r"[\w/§¹²³⁴⁵⁶⁷⁸⁹⁰-]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class BM25Index:
    def __init__(self, retriever: bm25s.BM25, chunk_ids: list[str]) -> None:
        self._retriever = retriever
        self._chunk_ids = chunk_ids

    @classmethod
    def build(cls, chunk_ids: list[str], texts: list[str]) -> "BM25Index":
        retriever = bm25s.BM25()
        retriever.index([tokenize(t) for t in texts], show_progress=False)
        return cls(retriever, chunk_ids)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        ids, scores = self._retriever.retrieve(
            [tokenize(query)], k=min(top_k, len(self._chunk_ids)), show_progress=False
        )
        return [
            (self._chunk_ids[i], float(s)) for i, s in zip(ids[0], scores[0], strict=True) if s > 0
        ]

    def save(self, path: Path | None = None) -> None:
        path = path or INDEX_DIR / "bm25"
        self._retriever.save(str(path))
        (path / "chunk_ids.json").write_text(json.dumps(self._chunk_ids))

    @classmethod
    def load(cls, path: Path | None = None) -> "BM25Index":
        path = path or INDEX_DIR / "bm25"
        retriever = bm25s.BM25.load(str(path))
        chunk_ids = json.loads((path / "chunk_ids.json").read_text())
        return cls(retriever, chunk_ids)
