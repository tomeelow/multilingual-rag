"""Hierarchical parent-child chunking.

Units (articles, recitals, …) are the semantic boundaries. Each unit becomes
one or more *parent* chunks (≤ parent_max_tokens); each parent is split into
*child* chunks (child_tokens, with overlap). Children are embedded and
retrieved; parents are passed to the LLM. Token counts use tiktoken, not
len() — chunk budgets must be in the same currency as model context windows.
"""

import json
import re

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import DATA_PROCESSED, pipeline_config
from src.models import Chunk

# stubs of repealed provisions carry no legal content and pollute retrieval
_REPEALED = re.compile(r"\((uchylon[ya]|skreślon[ya]|pominięte|втратила чинність)\)")

_REF_LABEL = {
    ("recital", "en"): "Recital {}",
    ("recital", "pl"): "Motyw {}",
    ("article", "en"): "Article {}",
    ("article", "pl"): "Art. {}",
    ("article", "uk"): "Стаття {}",
    ("annex", "en"): "Annex {}",
    ("annex", "pl"): "Załącznik {}",
    ("paragraph", "pl"): "pkt {}",
    ("paragraph", "en"): "para. {}",
}


def make_ref(kind: str, number: str | None, language: str) -> str | None:
    if number is None:
        return None
    label = _REF_LABEL.get((kind, language))
    return label.format(number) if label else number


class Chunker:
    def __init__(self) -> None:
        cfg = pipeline_config()["chunking"]
        self._enc = tiktoken.get_encoding(cfg["tokenizer"])
        separators = ["\n\n", "\n", ". ", " "]
        self._parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg["parent_max_tokens"],
            chunk_overlap=0,  # parents tile the unit; overlap belongs to children
            separators=separators,
            length_function=self.count_tokens,
        )
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg["child_tokens"],
            chunk_overlap=cfg["child_overlap"],
            separators=separators,
            length_function=self.count_tokens,
        )

    def count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    def chunk_document(self, doc: dict) -> list[Chunk]:
        meta = doc["meta"]
        provenance = dict(
            source_id=meta["source_id"],
            doc_title=meta["title"],
            language=meta["language"],
            jurisdiction=meta["jurisdiction"],
            doc_type=meta["doc_type"],
            official=meta["official"],
            url=meta["url"],
            ingestion_date=meta["ingestion_date"],
        )
        chunks: list[Chunk] = []
        for unit in doc["units"]:
            if _REPEALED.search(unit["text"]) and len(unit["text"]) < 80:
                continue  # "Art. 4. (uchylony)" — no legal content
            citation = dict(
                kind=unit["kind"],
                article_number=unit["number"],
                ref=make_ref(unit["kind"], unit["number"], meta["language"]),
                section_title=unit["section"] or unit["title"],
                pages=unit["pages"],
            )
            parent_texts = self._parent_splitter.split_text(unit["text"])
            for p_i, parent_text in enumerate(parent_texts):
                parent_id = (
                    unit["unit_id"] if len(parent_texts) == 1 else f"{unit['unit_id']}:p{p_i}"
                )
                chunks.append(
                    Chunk(
                        chunk_id=parent_id,
                        parent_id=parent_id,
                        chunk_type="parent",
                        text=parent_text,
                        token_count=self.count_tokens(parent_text),
                        **provenance,
                        **citation,
                    )
                )
                for c_i, child_text in enumerate(self._child_splitter.split_text(parent_text)):
                    if len(child_text.strip()) < 50:
                        continue  # splitter tail — too short to embed meaningfully
                    chunks.append(
                        Chunk(
                            chunk_id=f"{parent_id}:c{c_i}",
                            parent_id=parent_id,
                            chunk_type="child",
                            text=child_text,
                            token_count=self.count_tokens(child_text),
                            **provenance,
                            **citation,
                        )
                    )
        return chunks


def chunk_corpus() -> list[Chunk]:
    chunker = Chunker()
    chunks: list[Chunk] = []
    for path in sorted(DATA_PROCESSED.glob("*.json")):
        if path.name == "chunks.json":
            continue
        chunks.extend(chunker.chunk_document(json.loads(path.read_text())))
    return chunks


def load_chunks(chunk_type: str | None = None) -> list[Chunk]:
    raw = json.loads((DATA_PROCESSED / "chunks.json").read_text())
    return [Chunk.from_json(c) for c in raw if chunk_type in (None, c["chunk_type"])]


def write_chunks() -> list[Chunk]:
    from collections import Counter

    from loguru import logger

    chunks = chunk_corpus()
    out = DATA_PROCESSED / "chunks.json"
    out.write_text(json.dumps([c.to_json() for c in chunks], ensure_ascii=False))
    by_type = Counter(c.chunk_type for c in chunks)
    by_lang = Counter(f"{c.language}/{c.chunk_type}" for c in chunks)
    logger.info("wrote {} chunks to {} — {}", len(chunks), out, dict(by_type))
    logger.info("per language: {}", dict(sorted(by_lang.items())))
    return chunks


if __name__ == "__main__":
    write_chunks()
