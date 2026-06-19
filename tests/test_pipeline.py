"""Core pipeline: RRF fusion, cache layer, cache-first LLM, prompt format."""

from src.models import Chunk
from src.pipeline.cache import LLMCache, request_key
from src.pipeline.llm import (
    LLMClient,
    LLMResult,
    _gemini_payload,
    credentials_available,
)
from src.pipeline.prompts import SYSTEM_PROMPT, build_prompt
from src.pipeline.rag_chain import _retrieval_only_text
from src.pipeline.retriever import rrf_fuse


def test_rrf_prefers_items_ranked_by_both_legs():
    dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    sparse = [("c", 11.0), ("d", 9.0), ("a", 2.0)]
    fused = rrf_fuse([dense, sparse], rrf_k=60)
    ids = [cid for cid, _ in fused]
    # a (ranks 0+2) and c (ranks 2+0) tie and beat single-leg items
    assert set(ids[:2]) == {"a", "c"}
    assert ids.index("a") < ids.index("b")


def test_rrf_uses_rank_not_score():
    # huge raw score on one leg must not dominate: position is what counts
    dense = [("x", 0.51), ("y", 0.50)]
    sparse = [("y", 9999.0)]
    fused = dict(rrf_fuse([dense, sparse]))
    assert fused["y"] > fused["x"]  # rank 1+0 beats rank 0 alone
    assert abs(fused["y"] - (1 / 61 + 1 / 60)) < 1e-9


def test_cache_round_trip(tmp_path):
    cache = LLMCache(tmp_path / "c.sqlite3")
    key = request_key({"q": "test"})
    assert cache.get(key) is None
    cache.set(key, {"text": "odpowiedź", "input_tokens": 10, "output_tokens": 5, "model": "m"})
    assert cache.get(key)["text"] == "odpowiedź"


def test_request_key_sensitivity():
    base = {"model": "m", "messages": [{"role": "user", "content": "q"}], "max_tokens": 10}
    assert request_key(base) == request_key(dict(base))
    assert request_key(base) != request_key(base | {"model": "other"})
    assert request_key(base) != request_key(base | {"max_tokens": 11})


class _CountingLLM(LLMClient):
    """LLMClient with the network transport replaced by a counter."""

    def __init__(self, cache):
        super().__init__(provider="openai", model="fake-model", cache=cache)
        self.raw_calls = 0

    def _raw_complete(self, messages, max_tokens, temperature):
        self.raw_calls += 1
        return LLMResult(
            text="answer [Source: X, Art. 1]", input_tokens=7, output_tokens=3, model=self.model
        )

    def _raw_stream(self, messages, max_tokens, temperature, usage):
        self.raw_calls += 1
        usage["input_tokens"], usage["output_tokens"] = 7, 3
        yield from ["ans", "wer"]


def test_llm_complete_is_cache_first(tmp_path):
    llm = _CountingLLM(LLMCache(tmp_path / "c.sqlite3"))
    messages = [{"role": "user", "content": "pytanie"}]
    first = llm.complete(messages)
    second = llm.complete(messages)
    assert llm.raw_calls == 1
    assert not first.cached and second.cached
    assert first.text == second.text
    assert second.input_tokens == 7


def test_llm_stream_fills_and_replays_cache(tmp_path):
    llm = _CountingLLM(LLMCache(tmp_path / "c.sqlite3"))
    messages = [{"role": "user", "content": "stream me"}]
    assert "".join(llm.stream(messages)) == "answer"
    assert "".join(llm.stream(messages)) == "answer"  # replayed
    assert llm.raw_calls == 1
    # the streamed response is also visible to complete()
    assert llm.complete(messages).cached


def test_gemini_provider_defaults(tmp_path):
    llm = LLMClient(provider="gemini", cache=LLMCache(tmp_path / "c.sqlite3"))
    assert llm.model == "gemini-2.5-flash"


def test_gemini_credentials_follow_env(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert not credentials_available("gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert credentials_available("gemini")


def test_gemini_payload_splits_system_and_maps_roles():
    system, contents = _gemini_payload(
        [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
    )
    assert system == "rules"
    assert contents == [
        {"role": "user", "parts": [{"text": "q"}]},
        {"role": "model", "parts": [{"text": "a"}]},  # assistant -> Gemini's "model"
    ]


def _chunk(**kw) -> Chunk:
    defaults = dict(
        chunk_id="d:art:1",
        parent_id="d:art:1",
        chunk_type="parent",
        text="Treść artykułu.",
        token_count=5,
        source_id="d",
        doc_title="Ustawa testowa",
        language="pl",
        jurisdiction="PL",
        doc_type="act",
        official=True,
        url="https://example.invalid",
        ingestion_date="2026-06-11",
        kind="article",
        article_number="1",
        ref="Art. 1",
        section_title=None,
        pages=[1, 1],
    )
    return Chunk(**(defaults | kw))


def test_build_prompt_format():
    chunks = [
        _chunk(),
        _chunk(chunk_id="e:para:2", ref="pkt 2", doc_title="Opinia EROD", official=False),
    ]
    messages = build_prompt("Jakie są zasady?", chunks)
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    user = messages[1]["content"]
    assert "[Ustawa testowa | Art. 1 | lang:pl]" in user
    assert "UNOFFICIAL TRANSLATION" in user  # unofficial sources are labeled
    assert user.endswith("Question: Jakie są zasady?")


def test_retrieval_only_text_is_explicit_and_cited():
    text = _retrieval_only_text("pl", [_chunk()])
    assert "Tryb tylko wyszukiwania" in text
    assert "Nie jest to wygenerowana odpowiedź" in text
    assert "Ustawa testowa — Art. 1" in text
    assert "Treść artykułu." in text
