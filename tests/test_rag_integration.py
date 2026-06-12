"""End-to-end pipeline against the real local indexes with a fake LLM.

Skipped automatically when the indexes have not been built (CI without
artifacts); run `src.retrieval.build_indexes` first to enable.
"""

import pytest

from src.config import DATA_PROCESSED, INDEX_DIR
from src.pipeline.cache import LLMCache
from src.pipeline.llm import LLMClient, LLMResult

pytestmark = pytest.mark.skipif(
    not (INDEX_DIR / "bm25").exists() or not (DATA_PROCESSED / "chunks.json").exists(),
    reason="local indexes not built",
)


class _FakeLLM(LLMClient):
    def __init__(self, tmp_path):
        super().__init__(provider="openai", model="fake", cache=LLMCache(tmp_path / "c.sqlite3"))

    def _raw_complete(self, messages, max_tokens, temperature):
        return LLMResult(
            text="Odpowiedź. [Source: Test, Art. 1]",
            input_tokens=1,
            output_tokens=1,
            model="fake",
        )

    def _raw_stream(self, messages, max_tokens, temperature, usage):
        yield "Odpowiedź."


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    from src.pipeline.rag_chain import RAGPipeline
    from src.pipeline.retriever import get_retriever
    from src.retrieval.qdrant_store import get_client

    instance = RAGPipeline(llm=_FakeLLM(tmp_path_factory.mktemp("cache")))
    yield instance
    get_client().close()
    get_client.cache_clear()
    get_retriever.cache_clear()


def test_answer_polish_query(pipeline):
    r = pipeline.answer("Jakie kary pieniężne przewiduje RODO?")
    assert r.language == "pl"
    assert r.chunk_ids and r.sources
    # same-language retrieval by default: all sources Polish
    assert all(s.language == "pl" for s in r.sources)
    assert any(s.source_id == "gdpr_pl" for s in r.sources)
    assert r.latency_ms >= r.retrieval_ms


def test_cross_lingual_flag_controls_language_filter(pipeline):
    query = "Які системи штучного інтелекту заборонені в ЄС?"
    same_lang = pipeline.retriever.hybrid(query, language="uk", cross_lingual=False)
    assert {c.language for c in same_lang} == {"uk"}
    # the AI Act exists only in English in this corpus — the dense leg must
    # bridge languages once the filter is lifted
    open_lang = pipeline.retriever.hybrid(query, language="uk", cross_lingual=True)
    assert any(c.source_id == "eu_ai_act_2024" for c in open_lang)


def test_cross_lingual_answer_language_follows_query(pipeline):
    r = pipeline.answer("Які системи штучного інтелекту заборонені в ЄС?", cross_lingual=True)
    assert r.language == "uk"
    assert any(s.source_id == "eu_ai_act_2024" for s in r.sources)
    assert any(s.ref == "Article 5" for s in r.sources)


def test_rights_query_not_topped_by_duties_article(pipeline):
    """Regression: 'podstawowe prawa pracownika' used to rank Art. 100 KP
    (employee *duties*, 'Pracownik jest obowiązany…') first, above the
    Chapter II basic-principles articles that actually answer it."""
    children, _, _, _ = pipeline.retrieve(
        "Jakie są podstawowe prawa pracownika według Kodeksu pracy?",
        language="pl",
        cross_lingual=False,
    )
    top = [(c.source_id, c.article_number) for c in children]
    assert ("pl_labour_code", "100") not in top
    # the top result comes from the basic-principles chapter
    assert "Podstawowe zasady" in (children[0].section_title or "")


def test_article_range_query_filters_out_of_range(pipeline):
    """Regression: 'art. 10–18 Kodeksu pracy' used to surface Art. 100 and
    Art. 210 while Arts. 11¹–11³, 15, 17 never reached the candidate pool."""
    children, _, _, _ = pipeline.retrieve(
        "Jakie prawa pracownika wynikają z art. 10–18 Kodeksu pracy?",
        language="pl",
        cross_lingual=False,
    )
    from src.pipeline.query_analysis import article_base_number

    bases = [article_base_number(c.article_number) for c in children]
    assert children and all(c.kind == "article" for c in children)
    assert all(b is not None and 10 <= b <= 18 for b in bases)
    assert all(c.source_id == "pl_labour_code" for c in children)


def test_stream_answer_events(pipeline):
    events = list(pipeline.stream_answer("What are the lawful bases for processing?"))
    tokens = [e["token"] for e in events if "token" in e]
    final = events[-1]
    assert "".join(tokens)
    assert final["done"] is True
    assert final["sources"] and final["chunk_ids"]
