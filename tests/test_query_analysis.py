"""Query-analysis heuristics: rights/duties intent and rerank weighting."""

from src.models import Chunk
from src.pipeline.query_analysis import detect_rights_duties_intent, intent_weight


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
