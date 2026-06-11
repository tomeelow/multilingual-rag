"""HyDE: hypothesis goes through the cached LLM and is embedded as a passage."""

import numpy as np
import pytest

from src.config import DATA_PROCESSED, INDEX_DIR
from src.pipeline import hyde
from src.pipeline.cache import LLMCache
from src.pipeline.hyde import hyde_query_vector
from src.pipeline.llm import LLMClient, LLMResult


class _HypothesisLLM(LLMClient):
    def __init__(self, cache):
        super().__init__(provider="openai", model="fake", cache=cache)
        self.raw_calls = 0

    def _raw_complete(self, messages, max_tokens, temperature):
        self.raw_calls += 1
        assert "Question: Jakie kary" in messages[0]["content"]
        return LLMResult(
            text="Art. 83. Kary pieniężne wynoszą do 20 mln EUR.",
            input_tokens=5,
            output_tokens=9,
            model="fake",
        )


class _SpyEmbedder:
    def __init__(self):
        self.passages: list[str] = []

    def embed_passage(self, text):
        self.passages.append(text)
        return np.zeros(4).tolist()


def test_hyde_embeds_hypothesis_as_passage(tmp_path, monkeypatch):
    spy = _SpyEmbedder()
    monkeypatch.setattr(hyde, "get_embedder", lambda: spy)
    llm = _HypothesisLLM(LLMCache(tmp_path / "c.sqlite3"))

    hyde_query_vector("Jakie kary przewiduje RODO?", llm)
    assert spy.passages == ["Art. 83. Kary pieniężne wynoszą do 20 mln EUR."]

    # second call for the same query: served from cache, no new LLM call
    hyde_query_vector("Jakie kary przewiduje RODO?", llm)
    assert llm.raw_calls == 1


@pytest.mark.skipif(
    not (INDEX_DIR / "bm25").exists() or not (DATA_PROCESSED / "chunks.json").exists(),
    reason="local indexes not built",
)
def test_pipeline_answer_with_hyde(tmp_path):
    from src.pipeline.rag_chain import RAGPipeline

    class _FakeLLM(LLMClient):
        def __init__(self):
            super().__init__(
                provider="openai", model="fake", cache=LLMCache(tmp_path / "c.sqlite3")
            )

        def _raw_complete(self, messages, max_tokens, temperature):
            # first call is the HyDE hypothesis, second the actual answer
            text = (
                "Art. 152. Pracownikowi przysługuje coroczny, nieprzerwany, płatny urlop."
                if "excerpt" in messages[0]["content"]
                else "Odpowiedź [Source: Kodeks pracy, Art. 152]"
            )
            return LLMResult(text=text, input_tokens=1, output_tokens=1, model="fake")

    pipeline = RAGPipeline(llm=_FakeLLM())
    r = pipeline.answer("Ile urlopu przysługuje pracownikowi?", use_hyde=True)
    assert r.sources and r.chunk_ids
    assert any(s.source_id == "pl_labour_code" for s in r.sources)
