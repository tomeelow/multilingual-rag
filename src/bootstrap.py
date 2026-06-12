"""Idempotent index bootstrap — the docker-compose `indexer` service.

Runs parse -> chunk -> embed/index, skipping every step whose output already
exists, so `docker compose up` is self-initializing on first run and a no-op
on every restart. Safe to run locally too.

Usage: uv run python -m src.bootstrap
"""

from loguru import logger

from src.config import DATA_PROCESSED, INDEX_DIR
from src.ingestion.corpus import CORPUS


def _ingestion_done() -> bool:
    return all((DATA_PROCESSED / f"{e.meta.source_id}.json").exists() for e in CORPUS)


def _qdrant_done(expected: int) -> bool:
    from src.retrieval import qdrant_store

    client = qdrant_store.get_client()
    name = qdrant_store.collection_name()
    return client.collection_exists(name) and qdrant_store.count(client) == expected


def main() -> None:
    if _ingestion_done():
        logger.info("ingestion: up to date ({} documents)", len(CORPUS))
    else:
        from src.ingestion.run_ingestion import ingest_all

        ingest_all()

    if (DATA_PROCESSED / "chunks.json").exists():
        logger.info("chunking: up to date")
    else:
        from src.ingestion.chunker import write_chunks

        write_chunks()

    from src.ingestion.chunker import load_chunks

    children = len(load_chunks("child"))
    bm25_done = (INDEX_DIR / "bm25" / "chunk_ids.json").exists()
    if bm25_done and _qdrant_done(children):
        logger.info("indexes: up to date ({} child chunks)", children)
    else:
        from src.retrieval.build_indexes import build

        build()

    from src.retrieval import qdrant_store

    # close explicitly: the embedded client's __del__ during interpreter
    # shutdown prints a scary (but harmless) ImportError traceback otherwise
    qdrant_store.get_client().close()
    logger.info("bootstrap complete")


if __name__ == "__main__":
    main()
