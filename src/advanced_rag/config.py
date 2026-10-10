"""Single source of runtime configuration, loaded from env / .env."""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Secrets / endpoints ---
    groq_api_key: SecretStr = SecretStr("")
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None
    qdrant_local_path: str = ".qdrant_local"
    ingest_api_key: SecretStr = SecretStr("")

    # --- Index & models ---
    collection: str = "arxiv_papers"
    dense_model: str = "BAAI/bge-small-en-v1.5"
    dense_dim: int = 384
    # bge models expect this instruction on queries (not passages); fastembed does not add it.
    dense_query_prefix: str = "Represent this sentence for searching relevant passages: "
    sparse_model: str = "Qdrant/bm25"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    fastembed_cache_path: str | None = None

    # --- LLMs (Groq) ---
    generator_model: str = "openai/gpt-oss-120b"
    judge_model: str = "openai/gpt-oss-20b"
    llm_max_concurrency: int = 2
    llm_timeout_s: float = 60.0

    # --- Retrieval ---
    prefetch_k: int = 40
    fusion_k: int = 15  # candidates sent to the cross-encoder; its cost is linear in this
    rerank_batch_size: int = 4  # small batches were fastest on CPU (less padding)
    final_k: int = 6
    max_chunks_per_paper: int = 3

    # --- Self-correction loop ---
    rerank_weak_threshold: float = 0.0  # cross-encoder logit; calibrated via retrieval_eval
    max_rewrites: int = 2
    max_regenerations: int = 1

    # --- Ingestion / maintenance ---
    arxiv_query: str = (
        '(cat:cs.CL OR cat:cs.IR OR cat:cs.LG OR cat:cs.AI) AND abs:"retrieval augmented"'
    )
    arxiv_delay_s: float = 3.0
    arxiv_user_agent: str = "advanced-rag/0.1 (+https://github.com/msaiprathyush/advanced-rag)"
    # Parsing guards: huge PDFs spike memory in the layout parser and OOM-killed the 2 GiB Job.
    # Over-limit papers are indexed abstract-only (with a warning) instead of crashing the batch.
    pdf_max_pages: int = 60
    pdf_max_mb: int = 20
    pdf_cache_dir: str | None = (
        None  # opt-in disk cache; Cloud Run is stateless and /app is read-only
    )
    reindex_max_new: int = 40  # ~40 s/paper to embed on CPU -> ~30 min, inside the 1 h job timeout
    max_papers_in_index: int = 1500
    pinned_papers_file: str = "evals/pinned_papers.txt"
    api_ingest_max_results: int = 10

    # --- Drift monitor (jobs/drift_monitor.py): re-index only when quality drifts ---
    gcp_region: str = "us-central1"
    reindex_job_name: str = "arxiv-reindex"
    drift_min_recall: float = 0.90  # retrieval canary floors (same as the CI retrieval gate)
    drift_min_mrr: float = 0.85
    drift_window_days: int = 7
    drift_min_queries: int = 20  # below this, live-traffic signals are too noisy to act on
    drift_max_decline_rate: float = 0.35  # INITIAL values: tune against real traffic
    drift_min_median_top_score: float = 2.0
    drift_cooldown_hours: int = 24  # never re-index twice inside this window (prevents loops)
    drift_dry_run: bool = False

    # --- Logging / API ---
    log_level: str = "INFO"
    log_json: bool = True
    query_rate_limit: str = (
        "3/minute"  # Groq free tier (8k tokens/min) sustains ~1-2 queries/min org-wide
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
