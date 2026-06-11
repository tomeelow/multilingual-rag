"""Shared data structures: document metadata and parsed structural units."""

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any


@dataclass
class DocumentMeta:
    source_id: str  # "eu_ai_act_2024"
    title: str
    language: str  # ISO 639-1: "en", "pl", "uk"
    doc_type: str  # "regulation", "act", "guidance"
    jurisdiction: str  # "EU", "PL", "UA"
    effective_date: date
    url: str
    official: bool = True  # False for unofficial translations
    consolidated_as_of: date | None = None  # version of the consolidated text, if known
    page_count: int = 0

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["effective_date"] = self.effective_date.isoformat()
        if self.consolidated_as_of:
            d["consolidated_as_of"] = self.consolidated_as_of.isoformat()
        return d


@dataclass
class Unit:
    """One structural unit of a document: a recital, an article, an annex,
    a numbered paragraph, or the whole document for unstructured sources."""

    unit_id: str  # "gdpr_en:art:6"
    kind: str  # "recital" | "article" | "annex" | "paragraph" | "document"
    number: str | None  # "6", "18³", "8-1", "IV"
    title: str | None  # article/annex title where the document has one
    section: str | None  # nearest chapter/section heading, if tracked
    text: str
    pages: list[int] = field(default_factory=list)  # [first_page, last_page]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Chunk:
    """One indexable chunk. Children (256 tokens) are what retrieval scores;
    parents (a full article, ≤1024 tokens) are what the LLM reads. Every chunk
    carries full provenance — without it citations cannot be reconstructed."""

    chunk_id: str  # "gdpr_en:art:6:c0"
    parent_id: str  # the parent chunk this child belongs to (parents: itself)
    chunk_type: str  # "parent" | "child"
    text: str
    token_count: int
    # provenance
    source_id: str
    doc_title: str
    language: str
    jurisdiction: str
    doc_type: str
    official: bool
    url: str
    ingestion_date: str
    # citation
    kind: str  # unit kind: "recital" | "article" | "annex" | "paragraph" | "document"
    article_number: str | None  # raw number: "6", "18³", "8-1", "III"
    ref: str | None  # display citation: "Article 6", "Motyw 14", "Стаття 8"
    section_title: str | None
    pages: list[int] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Chunk":
        return cls(**d)
