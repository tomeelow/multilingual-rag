"""Structural parsing: heading grammars, sequence filters, page mapping."""

from src.ingestion.corpus import CorpusEntry, by_source_id
from src.ingestion.structure import _ascii_number, _Doc, _main_number, parse_units
from src.models import DocumentMeta


def _entry(structure: str, language: str = "en") -> CorpusEntry:
    from datetime import date

    return CorpusEntry(
        meta=DocumentMeta(
            source_id="test_doc",
            title="Test",
            language=language,
            doc_type="regulation",
            jurisdiction="EU",
            effective_date=date(2024, 1, 1),
            url="https://example.invalid",
        ),
        file="none.pdf",
        structure=structure,
    )


EURLEX_EN = """\
REGULATION (EU) 2024/0001
Whereas:
(1) First recital text.
(2) Second recital, which mentions
(3) an inline number that is not a recital start.
(3) Third recital.
Article 1
Subject matter
This Regulation lays down rules.
Article 2
Definitions
(1) 'system' means something; looks like a recital but is inside an article.
Article 3
Final provisions
This text ends the body.
Done at Brussels, 13 June 2024.
ANNEX I
List of things
First annex content.
"""


def test_eurlex_units():
    units = parse_units(EURLEX_EN.splitlines(), _entry("eurlex"))
    by_id = {u.unit_id: u for u in units}
    recitals = [u for u in units if u.kind == "recital"]
    articles = [u for u in units if u.kind == "article"]
    assert [u.number for u in recitals] == ["1", "2", "3"]
    assert [u.number for u in articles] == ["1", "2", "3"]
    assert by_id["test_doc:art:1"].title == "Subject matter"
    # the definitions "(1)" inside Article 2 must not become a recital
    assert "(1) 'system'" in by_id["test_doc:art:2"].text
    # the signature block is cut from the last article
    assert "Done at" not in by_id["test_doc:art:3"].text
    assert by_id["test_doc:annex:I"].title == "List of things"


ISAP = """\
DZIAŁ PIERWSZY
Przepisy ogólne
Rozdział I
Przepisy wstępne
Art. 1. Kodeks pracy określa prawa i obowiązki.
Art. 18. § 1. Postanowienia umów o pracę.
Art. 18³. Pracodawcy oraz organy administracji.
Art. 18³a. § 1. Pracownicy powinni być równo traktowani.
Art. 97. Świadectwo pracy, po luce po przepisach uchylonych.
Art. 183. Pracownik ma prawo.
Art. 240. Układ zbiorowy pracy.
Art. 305. Przepis końcowy.
"""


def test_isap_superscript_articles():
    units = parse_units(ISAP.splitlines(), _entry("isap", "pl"))
    ids = [u.unit_id for u in units]
    # 18³ and 183 are different articles and must not collide
    assert "test_doc:art:18^3" in ids
    assert "test_doc:art:18^3a" in ids
    assert "test_doc:art:183" in ids
    assert ids.index("test_doc:art:18^3") < ids.index("test_doc:art:183")
    assert units[0].section == "Rozdział I — Przepisy wstępne"
    # gaps from repealed articles are allowed
    assert "test_doc:art:97" in ids
    assert "test_doc:art:305" in ids


UA = """\
Закон України
Про захист персональних даних
Розділ I
Загальні положення
Стаття 1. Сфера дії Закону
Цей Закон регулює правові відносини.
Стаття 2. Визначення термінів
У цьому Законі терміни вживаються.
Стаття 8-1. Додаткова стаття.
Текст додаткової статті.
"""


def test_ua_law_units():
    units = parse_units(UA.splitlines(), _entry("ua_law", "uk"))
    assert [u.number for u in units] == ["1", "2", "8-1"]
    assert units[0].title == "Сфера дії Закону"
    assert units[0].section == "Розділ I — Загальні положення"


def test_numbered_paragraphs_tolerates_gap():
    lines = [f"{n}. Treść akapitu numer {n}." for n in range(1, 15) if n != 7]
    units = parse_units(lines, _entry("numbered_paragraphs", "pl"))
    numbers = [int(u.number) for u in units]
    assert 7 not in numbers
    assert numbers[-1] == 14  # numbering resumes after the gap


def test_plain_fallback_single_unit():
    units = parse_units(["Tylko zwykły tekst.", "Bez numeracji."], _entry("plain", "pl"))
    assert len(units) == 1
    assert units[0].kind == "document"


def test_number_helpers():
    assert _main_number("18³a") == 18
    assert _main_number("8-1") == 8
    assert _ascii_number("18³a") == "18^3a"
    assert _ascii_number("186⁸") == "186^8"
    assert _ascii_number("99") == "99"


def test_page_mapping():
    doc = _Doc.from_pages(["page one text", "page two text", "page three"])
    assert doc.page_of(0) == 1
    assert doc.page_of(len("page one text") + 1) == 2
    assert doc.page_of(len(doc.text) - 1) == 3


def test_registry_has_all_nine_documents():
    from src.ingestion.corpus import CORPUS

    assert len(CORPUS) == 9
    assert by_source_id("gdpr_pl").meta.language == "pl"
    unofficial = [e.meta.source_id for e in CORPUS if not e.meta.official]
    assert unofficial == ["edpb_opinion_28_2024_pl"]
