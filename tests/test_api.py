"""API contract tests with a fake pipeline (no models, no network)."""

import json

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.config import INDEX_DIR


class _FakePipeline:
    def stream_answer(self, query, filter_language=None, cross_lingual=False, use_hyde=False):
        yield {"token": "Odpo"}
        yield {"token": "wiedź"}
        yield {
            "done": True,
            "language": filter_language or "pl",
            "sources": [
                {
                    "source_id": "gdpr_pl",
                    "doc_title": "RODO",
                    "ref": "Art. 6",
                    "language": "pl",
                    "jurisdiction": "EU",
                    "official": True,
                    "url": "https://example.invalid",
                    "pages": [36, 37],
                }
            ],
            "chunk_ids": ["gdpr_pl:art:6:c0"],
            "latency_ms": 10,
            "retrieval_ms": 5,
            "rerank_ms": 2,
            "generation_ms": 3,
        }


class _ExplodingPipeline:
    def stream_answer(self, *a, **kw):
        raise RuntimeError("boom")
        yield  # pragma: no cover


@pytest.fixture
def client():
    with TestClient(create_app(pipeline=_FakePipeline())) as c:
        yield c


def _events(response) -> list[dict]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


def test_chat_streams_tokens_then_metadata(client):
    r = client.post("/api/chat", json={"query": "Jakie są podstawy przetwarzania?"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _events(r)
    tokens = [e["token"] for e in events if "token" in e]
    assert "".join(tokens) == "Odpowiedź"
    final = events[-1]
    assert final["done"] and final["sources"][0]["ref"] == "Art. 6"


def test_chat_validates_request(client):
    assert client.post("/api/chat", json={"query": ""}).status_code == 422
    assert client.post("/api/chat", json={"query": "q", "filter_language": "de"}).status_code == 422


def test_chat_error_becomes_sse_event():
    with TestClient(create_app(pipeline=_ExplodingPipeline())) as c:
        events = _events(c.post("/api/chat", json={"query": "q"}))
    assert events[-1]["done"] and "error" in events[-1]


def test_sources_lists_corpus(client):
    docs = client.get("/api/sources").json()
    assert len(docs) == 9
    by_id = {d["source_id"]: d for d in docs}
    assert by_id["gdpr_pl"]["language"] == "pl"
    assert by_id["edpb_opinion_28_2024_pl"]["official"] is False
    assert by_id["eu_ai_act_2024"]["units"] == 306


@pytest.mark.skipif(not (INDEX_DIR / "index_meta.json").exists(), reason="indexes not built")
def test_health(client, monkeypatch):
    # pin the provider: health reports whatever LLM_PROVIDER is set to, so
    # asserting a literal would fail on any checkout configured differently
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    h = client.get("/api/health").json()
    assert h["status"] == "ok"
    assert h["indexed_chunks"] == 3710
    assert h["languages"] == ["en", "pl", "uk"]
    assert h["embedding_model"] == "intfloat/multilingual-e5-large"
    assert h["generation_available"] is True
    assert h["generation_provider"] == "anthropic"
