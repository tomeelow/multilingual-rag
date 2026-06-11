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
