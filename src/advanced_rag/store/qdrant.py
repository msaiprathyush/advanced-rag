"""Qdrant access: Cloud when QDRANT_URL is set, embedded local mode otherwise."""

import time
import uuid
from collections.abc import Iterable
from functools import lru_cache

from qdrant_client import QdrantClient, models

from advanced_rag.config import get_settings
from advanced_rag.embeddings.models import SparseVec
from advanced_rag.ingestion.models import Chunk

_NS = uuid.UUID("6f1f6a3e-5f0e-4a0b-9c57-0d6b1a2c4e11")


def point_id(arxiv_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_NS, f"{arxiv_id}:{chunk_index}"))


@lru_cache
def get_client() -> QdrantClient:
    s = get_settings()
    if s.qdrant_url:
        key = s.qdrant_api_key.get_secret_value() if s.qdrant_api_key else None
        return QdrantClient(url=s.qdrant_url, api_key=key, timeout=30)
    return QdrantClient(path=s.qdrant_local_path)


def ensure_collection(client: QdrantClient | None = None, name: str | None = None) -> None:
    s = get_settings()
    client = client or get_client()
    name = name or s.collection
    if not client.collection_exists(name):
        client.create_collection(
            name,
            vectors_config={
                "dense": models.VectorParams(size=s.dense_dim, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
    # Idempotent, and also upgrades collections created before an index was added. Qdrant Cloud's
    # strict mode rejects filters on unindexed fields.
    for field, schema in (
        ("arxiv_id", models.PayloadSchemaType.KEYWORD),
        ("categories", models.PayloadSchemaType.KEYWORD),
        ("published", models.PayloadSchemaType.DATETIME),
        ("chunk_index", models.PayloadSchemaType.INTEGER),
    ):
        client.create_payload_index(name, field, schema)


def upsert_chunks(
    chunks: list[Chunk],
    dense: list[list[float]],
    sparse: list[SparseVec],
    client: QdrantClient | None = None,
    name: str | None = None,
) -> None:
    client = client or get_client()
    name = name or get_settings().collection
    points = [
        models.PointStruct(
            id=point_id(c.arxiv_id, c.chunk_index),
            vector={
                "dense": d,
                "bm25": models.SparseVector(indices=sp.indices, values=sp.values),
            },
            payload=c.payload(),
        )
        for c, d, sp in zip(chunks, dense, sparse, strict=True)
    ]
    client.upsert(name, points, wait=True)


def existing_versions(
    arxiv_ids: Iterable[str], client: QdrantClient | None = None, name: str | None = None
) -> dict[str, int]:
    client = client or get_client()
    name = name or get_settings().collection
    ids = list(arxiv_ids)
    found: dict[str, int] = {}
    if not ids:
        return found
    offset = None
    flt = models.Filter(
        must=[
            models.FieldCondition(key="arxiv_id", match=models.MatchAny(any=ids)),
            # chunk 0 is written last, so its presence means the paper was fully ingested
            models.FieldCondition(key="chunk_index", match=models.MatchValue(value=0)),
        ]
    )
    while True:
        points, offset = client.scroll(
            name,
            scroll_filter=flt,
            limit=256,
            offset=offset,
            with_payload=["arxiv_id", "version"],
            with_vectors=False,
        )
        for p in points:
            found[p.payload["arxiv_id"]] = max(
                found.get(p.payload["arxiv_id"], 0), p.payload["version"]
            )
        if offset is None:
            return found


def delete_paper(
    arxiv_id: str, client: QdrantClient | None = None, name: str | None = None
) -> None:
    client = client or get_client()
    name = name or get_settings().collection
    client.delete(
        name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(key="arxiv_id", match=models.MatchValue(value=arxiv_id))
                ]
            )
        ),
        wait=True,
    )


def count_points(client: QdrantClient | None = None, name: str | None = None) -> int:
    client = client or get_client()
    return client.count(name or get_settings().collection, exact=True).count


_CATALOG_TTL_S = 600
_catalog_cache: tuple[float, list[dict]] | None = None


def catalog(client: QdrantClient | None = None, name: str | None = None) -> list[dict]:
    """One entry per indexed paper (arxiv_id, title, published, categories), newest first.

    Read from the abstract chunk (chunk_index 0), which every fully ingested paper has exactly once.
    Cached for a few minutes: the index only changes when the re-index job runs.
    """
    global _catalog_cache
    now = time.monotonic()
    if _catalog_cache and now - _catalog_cache[0] < _CATALOG_TTL_S:
        return _catalog_cache[1]
    client = client or get_client()
    name = name or get_settings().collection
    flt = models.Filter(
        must=[models.FieldCondition(key="chunk_index", match=models.MatchValue(value=0))]
    )
    papers: list[dict] = []
    offset = None
    while True:
        points, offset = client.scroll(
            name,
            scroll_filter=flt,
            limit=256,
            offset=offset,
            with_payload=["arxiv_id", "title", "published", "categories"],
            with_vectors=False,
        )
        papers += [p.payload for p in points]
        if offset is None:
            break
    papers.sort(key=lambda p: p.get("published", ""), reverse=True)
    _catalog_cache = (now, papers)
    return papers
