"""Parent-child chunking: budgets, hierarchy, provenance."""

import pytest

from src.ingestion.chunker import Chunker, make_ref

DOC = {
    "meta": {
        "source_id": "test_doc",
        "title": "Test Regulation",
        "language": "pl",
        "jurisdiction": "EU",
        "doc_type": "regulation",
        "official": True,
        "url": "https://example.invalid",
        "ingestion_date": "2026-06-11",
    },
    "units": [
        {
            "unit_id": "test_doc:art:1",
            "kind": "article",
            "number": "1",
            "title": None,
            "section": "Rozdział I",
            "text": "Art. 1. "
            + "Przetwarzanie danych osobowych odbywa się zgodnie z prawem. " * 200,
            "pages": [1, 3],
        },
        {
            "unit_id": "test_doc:art:2",
            "kind": "article",
            "number": "2",
            "title": None,
            "section": None,
            "text": "Art. 2. Krótki przepis o stosowaniu ustawy do przetwarzania danych.",
            "pages": [3, 3],
        },
        {
            "unit_id": "test_doc:art:3",
            "kind": "article",
            "number": "3",
            "title": None,
            "section": None,
            "text": "Art. 3. (uchylony)",
            "pages": [3, 3],
        },
    ],
}


@pytest.fixture(scope="module")
def chunks():
    return Chunker().chunk_document(DOC)


def test_budgets(chunks):
    for c in chunks:
        if c.chunk_type == "parent":
            assert c.token_count <= 1024
        else:
            assert c.token_count <= 256


def test_hierarchy(chunks):
    parents = {c.chunk_id for c in chunks if c.chunk_type == "parent"}
    children = [c for c in chunks if c.chunk_type == "child"]
    assert children, "long unit must produce child chunks"
    for c in children:
        assert c.parent_id in parents
    # the long article splits into several parents, each with its own children
    art1_parents = [p for p in parents if p.startswith("test_doc:art:1")]
    assert len(art1_parents) > 1
    # a short unit keeps the unit_id as its single parent id
    assert "test_doc:art:2" in parents


def test_repealed_stub_dropped(chunks):
    assert not any(c.article_number == "3" for c in chunks)


def test_provenance_populated(chunks):
    for c in chunks:
        assert c.source_id == "test_doc"
        assert c.language == "pl"
        assert c.jurisdiction == "EU"
        assert c.url and c.ingestion_date and c.doc_title
        assert c.ref == f"Art. {c.article_number}"


def test_make_ref_labels():
    assert make_ref("article", "6", "en") == "Article 6"
    assert make_ref("recital", "14", "pl") == "Motyw 14"
    assert make_ref("article", "8-1", "uk") == "Стаття 8-1"
    assert make_ref("paragraph", "23", "pl") == "pkt 23"
    assert make_ref("annex", "III", "en") == "Annex III"
    assert make_ref("document", None, "pl") is None
