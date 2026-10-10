"""Retrieval modes: dense | hybrid (dense+BM25, RRF) | hybrid_rerank (cross-encoder on top)."""

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient, models

from advanced_rag.config import get_settings
from advanced_rag.embeddings.models import embed_query, sparse_query
from advanced_rag.store.qdrant import get_client

Mode = Literal["dense", "hybrid", "hybrid_rerank"]


@dataclass
class RetrievedChunk:
    payload: dict
    score: float  # retrieval score (cosine / RRF)
    rerank_score: float | None = None  # cross-encoder logit, when reranked

    @property
    def arxiv_id(self) -> str:
        return self.payload["arxiv_id"]

    @property
    def text(self) -> str:
        return self.payload["text"]


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk] = field(default_factory=list)
    top_score: float | None = None  # best rerank score (None unless reranked)


@lru_cache
def _cross_encoder() -> TextCrossEncoder:
    s = get_settings()
    return TextCrossEncoder(s.rerank_model, cache_dir=s.fastembed_cache_path)


def _to_chunks(points) -> list[RetrievedChunk]:
    return [RetrievedChunk(payload=p.payload, score=p.score) for p in points]


def dense_search(
    client: QdrantClient, query: str, limit: int, flt: models.Filter | None = None
) -> list[RetrievedChunk]:
    res = client.query_points(
        get_settings().collection,
        query=embed_query(query),
        using="dense",
        limit=limit,
        query_filter=flt,
        with_payload=True,
    )
    return _to_chunks(res.points)


def hybrid_search(
    client: QdrantClient, query: str, limit: int, flt: models.Filter | None = None
) -> list[RetrievedChunk]:
    s = get_settings()
    sp = sparse_query(query)
    res = client.query_points(
        s.collection,
        prefetch=[
            models.Prefetch(
                query=embed_query(query), using="dense", limit=s.prefetch_k, filter=flt
            ),
            models.Prefetch(
                query=models.SparseVector(indices=sp.indices, values=sp.values),
                using="bm25",
                limit=s.prefetch_k,
                filter=flt,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    return _to_chunks(res.points)


def rerank(query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    if not chunks:
        return []
    scores = list(
        _cross_encoder().rerank(
            query, [c.text for c in chunks], batch_size=get_settings().rerank_batch_size
        )
    )
    for c, sc in zip(chunks, scores, strict=True):
        c.rerank_score = float(sc)
    return sorted(chunks, key=lambda c: c.rerank_score or 0.0, reverse=True)[:top_k]


def cap_per_paper(chunks: list[RetrievedChunk], cap: int) -> list[RetrievedChunk]:
    seen: dict[str, int] = {}
    out = []
    for c in chunks:
        if seen.get(c.arxiv_id, 0) < cap:
            out.append(c)
            seen[c.arxiv_id] = seen.get(c.arxiv_id, 0) + 1
    return out


class Retriever:
    def __init__(self, mode: Mode = "hybrid_rerank", client: QdrantClient | None = None) -> None:
        self.mode = mode
        self._client = client

    @property
    def client(self) -> QdrantClient:
        return self._client or get_client()

    def retrieve(self, query: str, k: int | None = None) -> RetrievalResult:
        s = get_settings()
        k = k or s.final_k
        if self.mode == "dense":
            return RetrievalResult(dense_search(self.client, query, k))
        if self.mode == "hybrid":
            return RetrievalResult(hybrid_search(self.client, query, k))
        candidates = hybrid_search(self.client, query, s.fusion_k)
        ranked = rerank(query, candidates, len(candidates))
        ranked = cap_per_paper(ranked, s.max_chunks_per_paper)[:k]
        return RetrievalResult(ranked, ranked[0].rerank_score if ranked else None)


def warm_up_reranker() -> None:
    """Load the cross-encoder and run one inference so the first real request doesn't pay for it."""
    rerank("warm up", [RetrievedChunk(payload={"text": "warm up", "arxiv_id": "-"}, score=0.0)], 1)
