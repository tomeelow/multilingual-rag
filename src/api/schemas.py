"""API request/response schemas."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None
    filter_language: str | None = Field(default=None, pattern="^(en|pl|uk)$")
    cross_lingual: bool = False  # retrieve across languages (dense leg bridges)
    use_hyde: bool = False  # HyDE retrieval mode (A/B option)


class SourceDocument(BaseModel):
    source_id: str
    title: str
    language: str
    jurisdiction: str
    doc_type: str
    official: bool
    url: str
    effective_date: str
    page_count: int
    units: int


class Health(BaseModel):
    status: str
    indexed_chunks: int
    languages: list[str]
    collection: str
    embedding_model: str
