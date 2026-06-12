"""Qdrant collection management and dense search.

QDRANT_URL set -> server mode (docker compose); unset -> embedded local mode
under ./qdrant_storage (no daemon needed for development). The collection
name carries the index version from configs/pipeline.yaml — changing the
chunking or embedding config means bumping the version and rebuilding.
"""

import uuid
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.config import ROOT, env, pipeline_config
from src.models import Chunk


def collection_name() -> str:
    cfg = pipeline_config()["index"]
    return f"{cfg['collection']}_v{cfg['version']}"


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    url = env("QDRANT_URL")
    if url:
        return QdrantClient(url=url)
    return QdrantClient(path=str(ROOT / "qdrant_storage"))


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def recreate_collection(client: QdrantClient) -> None:
    name = collection_name()
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(
            size=pipeline_config()["embedding"]["dimensions"], distance=Distance.COSINE
        ),
    )


def upsert_chunks(client: QdrantClient, chunks: list[Chunk], vectors) -> None:
    points = [
        PointStruct(id=_point_id(c.chunk_id), vector=v.tolist(), payload=c.to_json())
        for c, v in zip(chunks, vectors, strict=True)
    ]
    for i in range(0, len(points), 256):
        client.upsert(collection_name=collection_name(), points=points[i : i + 256])


def search(
    client: QdrantClient,
    query_vector: list[float],
    top_k: int,
    language: str | None = None,
    source_ids: list[str] | None = None,
) -> list[tuple[str, float]]:
    """Dense search over child chunks -> [(chunk_id, score)]."""
    conditions = []
    if language:
        conditions.append(FieldCondition(key="language", match=MatchValue(value=language)))
    if source_ids:
        conditions.append(FieldCondition(key="source_id", match=MatchAny(any=source_ids)))
    query_filter = Filter(must=conditions) if conditions else None
    hits = client.query_points(
        collection_name=collection_name(),
        query=query_vector,
        limit=top_k,
        query_filter=query_filter,
    ).points
    return [(h.payload["chunk_id"], h.score) for h in hits]


def count(client: QdrantClient) -> int:
    return client.count(collection_name(), exact=True).count
