"""Query-analysis heuristics: rights/duties intent and rerank weighting."""

from src.models import Chunk
from src.pipeline.query_analysis import (
    article_base_number,
    detect_rights_duties_intent,
    in_article_range,
    intent_weight,
    parse_article_range,
)


def _chunk(**kw) -> Chunk:
    defaults = dict(
        chunk_id="d:art:1:c0",
        parent_id="d:art:1",
        chunk_type="child",
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


def test_intent_detection_three_languages():
    assert detect_rights_duties_intent("Jakie są podstawowe prawa pracownika?") == "rights"
    assert detect_rights_duties_intent("What are the data subject's rights?") == "rights"
    assert detect_rights_duties_intent("Які права має суб'єкт даних?") == "rights"
    assert detect_rights_duties_intent("Jakie obowiązki ma pracownik?") == "duties"
    assert detect_rights_duties_intent("What are the controller's obligations?") == "duties"
    assert detect_rights_duties_intent("Які обов'язки має володілець даних?") == "duties"


def test_intent_neutral_when_absent_or_mixed():
    assert detect_rights_duties_intent("Kiedy zgłosić naruszenie danych?") is None
    # both sides mentioned -> never guess
    assert detect_rights_duties_intent("Jakie prawa i obowiązki ma pracownik?") is None


def test_opposing_section_is_demoted_and_aligned_boosted():
    duties = _chunk(section_title="Rozdział II — Obowiązki pracownika")
    rights = _chunk(section_title="Rozdział II — Podstawowe zasady prawa pracy")
    assert intent_weight("rights", duties) < 1.0
    assert intent_weight("rights", rights) > 1.0
    # symmetric for a duties query
    assert intent_weight("duties", duties) > 1.0
    assert intent_weight("duties", rights) < 1.0


def test_mixed_section_stays_neutral():
    mixed = _chunk(section_title="Rozdział II — Prawa i obowiązki pracownika")
    assert intent_weight("rights", mixed) == 1.0
    assert intent_weight("duties", mixed) == 1.0


def test_obligation_language_in_text_is_mildly_demoted():
    c = _chunk(text="Pracownik jest obowiązany wykonywać pracę sumiennie.")
    assert 0.0 < intent_weight("rights", c) < 1.0
    # but text mentioning both sides is neutral
    both = _chunk(text="Kodeks pracy określa prawa i obowiązki pracowników.")
    assert intent_weight("rights", both) == 1.0


def test_no_intent_no_adjustment():
    c = _chunk(section_title="Rozdział II — Obowiązki pracownika")
    assert intent_weight(None, c) == 1.0


def test_parse_article_range_variants():
    assert parse_article_range("prawa z art. 10–18 Kodeksu pracy") == (10, 18)
    assert parse_article_range("art. 10-18") == (10, 18)  # ASCII hyphen
    assert parse_article_range("artykuły 5–7 RODO") == (5, 7)
    assert parse_article_range("Articles 12–22 of the GDPR") == (12, 22)
    assert parse_article_range("статті 6–8 Закону") == (6, 8)
    # superscript suffix on a bound is consumed, base numbers returned
    assert parse_article_range("art. 18–18³ Kodeksu pracy") == (18, 18)
    # reversed bounds normalize
    assert parse_article_range("art. 18–10") == (10, 18)


def test_parse_article_range_absent():
    assert parse_article_range("Jakie są podstawowe prawa pracownika?") is None
    # a bare number range without an article token is not a legal reference
    assert parse_article_range("w latach 2016-2018") is None
    # single article mention is not a range
    assert parse_article_range("co mówi art. 100 Kodeksu pracy?") is None


def test_article_base_number():
    assert article_base_number("10") == 10
    assert article_base_number("11¹") == 11
    assert article_base_number("18³a") == 18
    assert article_base_number("8-1") == 8  # UA inserted-article numbering
    assert article_base_number("III") is None  # roman annex numbers
    assert article_base_number(None) is None


def test_in_article_range_checks_kind_and_base():
    assert in_article_range(_chunk(article_number="11¹"), 10, 18)
    assert not in_article_range(_chunk(article_number="100"), 10, 18)
    # recital 14 is not article 14
    assert not in_article_range(_chunk(kind="recital", article_number="14"), 10, 18)
