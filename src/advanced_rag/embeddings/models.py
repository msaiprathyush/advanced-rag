"""Local ONNX embedders (fastembed): dense bge + sparse BM25. Lazy process-wide singletons."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from fastembed import SparseTextEmbedding, TextEmbedding

from advanced_rag.config import get_settings


@dataclass(frozen=True)
class SparseVec:
    indices: list[int]
    values: list[float]


@lru_cache
def _dense() -> TextEmbedding:
    s = get_settings()
    return TextEmbedding(
        s.dense_model, cache_dir=s.fastembed_cache_path, local_files_only=s.models_local_only
    )


def _baked_snapshot(model: str, cache_dir: str | None) -> str | None:
    """Path of a model snapshot baked into the image, e.g. <cache>/models--Qdrant--bm25/snapshots/<rev>."""
    if not cache_dir:
        return None
    snaps = sorted(Path(cache_dir).glob(f"models--{model.replace('/', '--')}/snapshots/*"))
    return str(snaps[-1]) if snaps else None


@lru_cache
def _sparse() -> SparseTextEmbedding:
    s = get_settings()
    kwargs = {}
    if s.models_local_only:
        # fastembed's local-cache check for Qdrant/bm25 requires files the Hub repo never ships
        # (mock.file, tamil.txt), so it always falls through to the network and crashes startup when
        # the Hub is slow. Point it straight at the baked snapshot instead.
        kwargs["specific_model_path"] = _baked_snapshot(s.sparse_model, s.fastembed_cache_path)
    return SparseTextEmbedding(
        s.sparse_model,
        cache_dir=s.fastembed_cache_path,
        local_files_only=s.models_local_only,
        **kwargs,
    )


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
