"""Local ONNX embedders (fastembed): dense bge + sparse BM25. Lazy process-wide singletons."""

from dataclasses import dataclass
from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding

from advanced_rag.config import get_settings


@dataclass(frozen=True)
class SparseVec:
    indices: list[int]
    values: list[float]


@lru_cache
def _dense() -> TextEmbedding:
    s = get_settings()
    return TextEmbedding(s.dense_model, cache_dir=s.fastembed_cache_path)


@lru_cache
def _sparse() -> SparseTextEmbedding:
    s = get_settings()
    return SparseTextEmbedding(s.sparse_model, cache_dir=s.fastembed_cache_path)


def embed_passages(texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in _dense().passage_embed(texts)]


def embed_query(query: str) -> list[float]:
    # fastembed's query_embed does NOT add bge's retrieval instruction, so add it explicitly.
    prefixed = get_settings().dense_query_prefix + query
    return next(iter(_dense().query_embed(prefixed))).tolist()


def sparse_passages(texts: list[str]) -> list[SparseVec]:
    return [
        SparseVec(v.indices.tolist(), v.values.tolist()) for v in _sparse().passage_embed(texts)
    ]


def sparse_query(query: str) -> SparseVec:
    v = next(iter(_sparse().query_embed(query)))
    return SparseVec(v.indices.tolist(), v.values.tolist())


def warm_up() -> None:
    embed_query("warm up")
    sparse_query("warm up")
