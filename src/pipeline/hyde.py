"""HyDE — Hypothetical Document Embeddings.

The LLM writes a hypothetical regulation excerpt answering the query; that
text is embedded as a *passage* (not a query — the hypothesis lives in
document space) and used as the dense retrieval vector. The BM25 leg keeps
the raw query: a hallucinated hypothesis must not poison exact-term matching,
and HyDE stays an A/B option rather than the only retrieval path.

The generation goes through the cache-first LLM client, so repeated HyDE
retrievals for the same query cost one API call total.
"""

from src.pipeline.llm import LLMClient
from src.retrieval.embedding import get_embedder

HYDE_PROMPT = (
    "Write a one-paragraph excerpt from a legal regulation or statute that would "
    "directly answer the question below. Write it in the same language as the "
    "question, in formal statutory style. Do not mention that it is hypothetical.\n\n"
    "Question: {query}"
)


def hyde_query_vector(query: str, llm: LLMClient, max_tokens: int = 256) -> list[float]:
    hypothesis = llm.complete(
        [{"role": "user", "content": HYDE_PROMPT.format(query=query)}],
        max_tokens=max_tokens,
        temperature=0.0,
    ).text
    return get_embedder().embed_passage(hypothesis)
