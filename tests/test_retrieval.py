"""Retrieval components: e5 prefix contract, BM25 index round-trip."""

import numpy as np

from src.retrieval.bm25_index import BM25Index, tokenize
from src.retrieval.embedding import E5Embedder


class _FakeModel:
    def __init__(self):
        self.seen: list[str] = []

    def encode(self, texts, **kwargs):
        self.seen.extend(texts)
        return np.zeros((len(texts), 4))


def test_e5_prefixes_always_applied():
    embedder = E5Embedder()
    fake = _FakeModel()
    embedder._model = fake
    embedder.embed_query("co to są dane osobowe?")
    embedder.embed_passages(["Стаття 8. Права суб'єкта"])
    assert fake.seen[0] == "query: co to są dane osobowe?"
    assert fake.seen[1] == "passage: Стаття 8. Права суб'єкта"


def test_tokenize_preserves_legal_tokens():
    assert "2016/679" in tokenize("rozporządzenie 2016/679 (RODO)")
    assert "18³a" in tokenize("zgodnie z art. 18³a Kodeksu pracy")
    assert "стаття" in tokenize("Стаття 8 Закону")


def test_bm25_round_trip(tmp_path):
    ids = ["a:1", "a:2", "a:3"]
    texts = [
        "przetwarzanie danych osobowych na podstawie zgody",
        "kary pieniężne i sankcje administracyjne",
        "обробка персональних даних за згодою суб'єкта",
    ]
    index = BM25Index.build(ids, texts)
    top = index.search("zgody na przetwarzanie", top_k=2)
    assert top[0][0] == "a:1"

    index.save(tmp_path / "bm25")
    loaded = BM25Index.load(tmp_path / "bm25")
    assert loaded.search("персональних даних", top_k=1)[0][0] == "a:3"


def test_bm25_no_matches_returns_empty():
    index = BM25Index.build(["x"], ["całkowicie inny tekst"])
    assert index.search("nothing matches here", top_k=5) == []
