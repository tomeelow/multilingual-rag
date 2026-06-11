"""Language-aware generation prompt with mandatory citations."""

from src.models import Chunk

SYSTEM_PROMPT = """\
You are a multilingual legal assistant specializing in EU and Central/Eastern European \
regulatory documents (AI regulation and data protection law).
You answer in the SAME LANGUAGE as the user's question — always match their language exactly, \
even when the source documents are in a different language.
Base your answer ONLY on the provided context.
Every legal claim must be followed by a citation in the format \
[Source: {document_title}, {article_reference}]. Keep the article reference exactly as given \
in the context header — legal terminology must not be translated away; when you translate a \
key legal term, keep the original term in parentheses.
If a context block is marked UNOFFICIAL TRANSLATION, say so when citing it.
If the context does not contain sufficient information to answer, say so explicitly in the \
user's language — never infer, extrapolate, or invent article numbers."""


def format_context(chunks: list[Chunk]) -> str:
    blocks = []
    for c in chunks:
        flags = f"lang:{c.language}" + ("" if c.official else " | UNOFFICIAL TRANSLATION")
        header = f"[{c.doc_title} | {c.ref or c.kind} | {flags}]"
        blocks.append(f"{header}\n{c.text}")
    return "\n\n---\n\n".join(blocks)


def build_prompt(query: str, chunks: list[Chunk]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{format_context(chunks)}\n\nQuestion: {query}"},
    ]
