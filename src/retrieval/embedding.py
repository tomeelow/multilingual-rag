"""multilingual-e5-large wrapper — the only module that touches the model.

e5 is instruction-tuned with asymmetric prefixes: queries and passages MUST
be embedded as "query: …" and "passage: …" respectively. Skipping the prefix
degrades retrieval significantly and fails silently (ADR-2). Nothing outside
this module may call the model directly.
"""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import pipeline_config


class E5Embedder:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or pipeline_config()["embedding"]["model"]
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _encode(self, prefix: str, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self.model.encode(
            [f"{prefix}: {t}" for t in texts],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode("query", texts)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode("passage", texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_queries([text])[0].tolist()

    def embed_passage(self, text: str) -> list[float]:
        return self.embed_passages([text])[0].tolist()


@lru_cache(maxsize=1)
def get_embedder() -> E5Embedder:
    return E5Embedder()
