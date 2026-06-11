"""Populate both retrieval indexes from data/processed/chunks.json.

Embeds child chunks with e5 "passage: " prefixes into Qdrant and builds the
parallel BM25 index. Records index metadata (model, chunking config, build
date) in indexes/index_meta.json — an index without its build provenance
cannot be trusted after the first config change.

Usage: uv run python -m src.retrieval.build_indexes
"""

import json
from collections import Counter
from datetime import date

from loguru import logger

from src.config import INDEX_DIR, pipeline_config
from src.ingestion.chunker import load_chunks
from src.retrieval.bm25_index import BM25Index
from src.retrieval.embedding import get_embedder
from src.retrieval.qdrant_store import (
    collection_name,
    count,
    get_client,
    recreate_collection,
    upsert_chunks,
)


def build() -> None:
    children = load_chunks("child")
    logger.info("loaded {} child chunks", len(children))

    embedder = get_embedder()
    vectors = embedder.embed_passages([c.text for c in children])
    logger.info("embedded with {} -> {}", embedder.model_name, vectors.shape)

    client = get_client()
    recreate_collection(client)
    upsert_chunks(client, children, vectors)
    logger.info("qdrant collection '{}': {} points", collection_name(), count(client))

    INDEX_DIR.mkdir(exist_ok=True)
    bm25 = BM25Index.build([c.chunk_id for c in children], [c.text for c in children])
    bm25.save()
    logger.info("bm25 index saved to {}", INDEX_DIR / "bm25")

    cfg = pipeline_config()
    meta = {
        "collection": collection_name(),
        "embedding_model": embedder.model_name,
        "chunking": cfg["chunking"],
        "index_version": cfg["index"]["version"],
        "chunks_indexed": len(children),
        "per_language": dict(Counter(c.language for c in children)),
        "build_date": date.today().isoformat(),
    }
    (INDEX_DIR / "index_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("index metadata: {}", meta)


if __name__ == "__main__":
    build()
