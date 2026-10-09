# Advanced Production RAG — Project Brief

## Goal
Build an advanced, production-grade RAG system — including evaluation and ongoing
maintenance, not just a retrieve-and-generate demo — and push it to GitHub
(https://github.com/msaiprathyush) as a portfolio piece for Senior AI/ML Engineer
interviews.

**Narrative for interviews:** "A research assistant that answers questions across
recent ML/AI papers with cited sources, using hybrid retrieval, re-ranking, and a
self-correction loop that checks its own groundedness before answering — plus an
automated eval suite and a scheduled re-indexing job so the system stays current."

This should read as a system with real production and maintenance thinking, not a
weekend notebook project — that's the gap most portfolio RAG projects have.

---

## Hard constraints

- **Free tier only.** No paid cloud spend. GCP free tier for deployment (Cloud Run,
  Cloud Scheduler, Secret Manager). Qdrant Cloud free tier (1GB) for the vector store.
- **No Gemini / Google AI API.** Gemini is used at the author's day job — this is a
  personal project, so LLM calls must come from an independent free provider.
- **Data must be free and unambiguously legal to use.** No scraping, no copyrighted
  material. arXiv's public API is built for exactly this kind of programmatic use.

---

## Data source: arXiv API

- Official query endpoint (no auth, no key): `http://export.arxiv.org/api/query`
- Example query (ML/RAG papers, newest first):
  ```
  http://export.arxiv.org/api/query?search_query=cat:cs.CL+AND+abs:retrieval+augmented+generation&sortBy=submittedDate&sortOrder=descending&max_results=50
  ```
- Returns Atom/XML metadata (title, abstract, authors, categories, links).
- PDF fetch pattern: `https://arxiv.org/pdf/{arxiv_id}`
- **Rate limit: 3 requests/second.** No daily cap. Build in a small delay/backoff
  between calls regardless — don't hammer it even though there's no hard cap enforced.
- Not for bulk historical scraping (tens of thousands of papers) — arXiv has a
  separate bulk-data channel for that; this project's scale (periodic small batches)
  is exactly what the query API is for.

---

## Tech stack

| Layer | Choice | Notes |
|---|---|---|
| Ingestion | arXiv API + PyMuPDF (PDF parsing) | Respect the 3 req/s limit |
| Chunking | Recursive/semantic chunking | Tune chunk size for academic paper structure (abstract, sections, references) |
| Embeddings | **Local, open-source** — `sentence-transformers`, e.g. `BAAI/bge-small-en-v1.5` | Runs in-process, zero API cost, zero rate limit. Adds model weight (~100s of MB) to the deploy image — flag this as a deliberate trade-off in the README |
| LLM (generation) | **Groq API** (free tier) | No card required, fast inference, open models (Llama/Qwen/etc). Free tier is rate-limited (RPM/RPD, account-level) — build retry/backoff into every call from day one |
| Vector DB | Qdrant (Qdrant Cloud free tier, 1GB) | Supports hybrid (dense + sparse) search natively |
| Re-ranking | Cross-encoder (local, e.g. `sentence-transformers` cross-encoder model) | Keep local to stay free |
| Orchestration | LangChain (base pipeline) + LangGraph (corrective/self-grading loop) | LangGraph handles: query rewrite on weak retrieval, groundedness check before returning an answer |
| Serving | FastAPI | `/query`, `/ingest`, `/health` endpoints minimum |
| Containerization | Docker | Single image for the API service |
| Deployment | Cloud Run (GCP free tier) | Stateless — talks out to Qdrant Cloud and Groq; no persistent local storage needed |
| Secrets | GCP Secret Manager (free tier) | Groq API key, Qdrant API key/URL |
| Eval | RAGAS (open-source) | Faithfulness, answer relevancy, context precision/recall — wired into CI, not just ad hoc |
| CI/CD | GitHub Actions (free for public repos) | Lint → test → build image → deploy on merge to main |
| Scheduled maintenance | Cloud Scheduler + Cloud Run Job | Periodically pulls new arXiv papers, re-embeds, updates the index — the "maintenance" differentiator |
| Logging | Structured logging → Cloud Logging (free tier) | No separate observability vendor needed for this scale |

---

## What makes this "advanced" (not naive RAG)

Naive RAG = embed query → top-k similarity search → stuff into prompt → generate.
This project should go further:

1. **Hybrid retrieval** — dense (vector) + sparse (BM25/keyword) search combined, not
   vector-only.
2. **Re-ranking** — cross-encoder re-ranks the hybrid candidates before they reach the
   LLM.
3. **Query rewriting** — if the first retrieval pass scores poorly, LangGraph rewrites
   the query and retries before giving up.
4. **Groundedness / self-correction check** — before returning an answer, verify the
   LLM's claim is actually supported by the retrieved context (judge-style check,
   echoing the LLM-as-judge pattern already used professionally). If not grounded,
   re-retrieve or explicitly decline rather than silently hallucinate.
5. **Automated evaluation** — RAGAS metrics run in CI on a fixed eval set, not manual
   spot-checks.
6. **Scheduled re-indexing** — the index is kept fresh automatically, not a one-time
   static ingest.

---

## Suggested repo structure

```
advanced-rag/
├── README.md                  # architecture diagram, setup, demo, trade-offs write-up
├── src/
│   ├── ingestion/              # arXiv client, PDF parsing, chunking
│   ├── embeddings/              # local embedding model wrapper
│   ├── retrieval/                # hybrid search, re-ranking
│   ├── graph/                    # LangGraph: query rewrite + groundedness loop
│   ├── api/                      # FastAPI app (query/ingest/health)
│   └── eval/                      # RAGAS eval harness + fixed eval set
├── jobs/
│   └── reindex_job.py           # scheduled maintenance ingestion job
├── tests/                         # unit + integration tests
├── .github/workflows/            # CI/CD pipeline
├── Dockerfile
├── docker-compose.yml            # local dev (API + local Qdrant for testing)
├── requirements.txt / pyproject.toml
└── .env.example
```

---

## Build plan (3–4 weeks)

**Week 1 — Core pipeline, local only.**
arXiv ingestion → chunking → local embeddings → Qdrant → naive retrieve-then-generate
via Groq → minimal FastAPI wrapper. Goal: working end-to-end, correctness over polish.

**Week 2 — Advanced retrieval.**
Hybrid search, cross-encoder re-ranking, LangGraph graph for query rewrite +
groundedness-check loop. This is where "advanced" gets earned.

**Week 3 — Production hardening.**
Dockerize, deploy to Cloud Run + Qdrant Cloud, Secret Manager for keys, structured
logging, retry/backoff on all external calls (Groq + arXiv), unit + integration
tests, GitHub Actions CI.

**Week 4 — Maintenance layer + polish.**
RAGAS eval suite wired into CI, Cloud Scheduler re-ingestion job, README with
architecture diagram, and a short trade-offs write-up (this becomes the interview
cheat-sheet).

---

## Open notes for whoever picks this up

- Keep every external call (Groq, arXiv) behind a thin client with retry/backoff —
  free-tier rate limits make this a correctness issue, not just robustness polish.
- The embedding model choice affects both retrieval quality and deploy image size —
  document the trade-off explicitly rather than treating it as a throwaway default.
- Fixed eval set for RAGAS should be built early (even a small one) so eval isn't
  bolted on at the end.
