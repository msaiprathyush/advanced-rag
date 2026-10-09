# syntax=docker/dockerfile:1
# Single image for both the API service and the re-index Cloud Run Job.
# Embedding/re-ranking models run via ONNX (fastembed), so there is no torch in this image.

FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# Dependencies first (cached unless the lockfile changes). Runtime deps only: no dev/eval groups.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-default-groups --no-install-project

COPY src ./src
RUN uv sync --frozen --no-default-groups

# Bake the three ONNX models into the image so cold starts never download anything.
ENV FASTEMBED_CACHE_PATH=/models
RUN /app/.venv/bin/python - <<'EOF'
from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

TextEmbedding("BAAI/bge-small-en-v1.5")
SparseTextEmbedding("Qdrant/bm25")
TextCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2")
EOF


FROM python:3.12-slim AS runtime
RUN useradd --system --create-home --uid 10001 app
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder --chown=app:app /models /models
COPY src ./src
COPY jobs ./jobs
COPY evals/pinned_papers.txt ./evals/pinned_papers.txt

# NOTE: do not set HF_HUB_OFFLINE=1 here. fastembed's Qdrant/bm25 loader fails with it even when the
# model is baked into FASTEMBED_CACHE_PATH (reproduced locally); without it the baked cache is used
# and nothing is re-downloaded.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/models \
    LOG_JSON=true \
    PORT=8080
USER app
EXPOSE 8080

# The re-index Job overrides this with: python -m jobs.reindex_job
CMD ["sh", "-c", "exec python -m uvicorn advanced_rag.api.main:app --host 0.0.0.0 --port ${PORT}"]
