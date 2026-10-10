# Design trade-offs

The decisions in this project, what each one cost, and what I would change at larger scale.
Numbers are measured on this repo's 15-paper corpus unless noted.

## 1. Free-tier LLM capacity is the real constraint

Groq's free tier gives each model **8,000 tokens/minute, 200,000 tokens/day and 1,000 requests/day**. One query
costs roughly 5k tokens (a ~2.5k-token context for generation, then again for the judge), so the deployment
sustains **1-2 queries per minute and about 40 per day**.

- Every Groq call goes through one client with bounded concurrency and retry that honours `Retry-After`.
- The public API is limited to 3 queries/minute per IP.
- The daily cap shaped the eval design (section 10): an early RAGAS run exhausted a model's whole daily budget, after which every judge call was refused no matter how patiently it retried. Per-minute limits can be waited out; a daily cap cannot.
- **At scale:** a paid tier or a self-hosted model removes this. The graph and clients don't change.

## 2. ONNX (fastembed) instead of sentence-transformers + torch

The brief suggested sentence-transformers. I run the same model weights (`bge-small-en-v1.5`) through
ONNX Runtime via fastembed.

- Removes torch from the image: the built image is **382 MB** (measured in Artifact Registry), versus the 2 GB+ a torch-based image would typically be. That matters for cold starts on scale-to-zero Cloud Run, and keeps two images inside the 0.5 GB free registry tier.
- Qdrant's BM25 sparse encoder comes from the same library, so there is one model dependency.
- Cost: CPU inference is slow. Re-ranking 30 candidates took 12-25 s on a 4-core WSL machine.

## 3. Re-ranking cost drove the candidate count

Re-ranking is linear in the number of candidates. After measuring, I rerank the top **15** fused
results (not 30) with batch size 4, which cut end-to-end latency from ~40 s to ~5-9 s. Hybrid recall at
k=6 was already 1.0, so there was headroom. A GPU or a quantised/smaller cross-encoder would change this.

## 4. Hybrid retrieval and RRF

Dense + BM25 fused with reciprocal-rank fusion in a single Qdrant call. RRF needs no score
normalisation, since cosine similarity and BM25 scores aren't comparable. Measured gain over dense-only:
recall@6 0.962 to 1.0, MRR 0.933 to 0.981. The cross-encoder showed no gain at paper level on this small
corpus (the metric is saturated); I report that instead of claiming one.

## 5. Retrieval grading uses the cross-encoder, not an LLM

Corrective-RAG-style systems often ask an LLM whether retrieval was good. Here the top re-rank score
decides, which costs no tokens, which matters given decision 1. Trade-off: a fixed threshold must be
calibrated (unanswerable questions averaged -6.4, answerable p10 was +1.2) and can be fooled by on-topic
questions the corpus doesn't answer. Those are caught later by the model's own refusal and the groundedness check.

## 6. Groundedness fails closed

The check runs in two stages: a free code check (no valid `[n]` citation means ungrounded) and then an
LLM judge. If the judge errors, the answer is treated as ungrounded. Trade-off: a judge outage causes
declines instead of unchecked answers. For a research assistant I prefer a missing answer to a fabricated one.
One observed false decline: the model's claim about RAG-Sequence was *true* but not present in the
retrieved chunks, so it was correctly rejected. That is a retrieval-coverage gap, not a judge error.

## 7. Model output formats can't be trusted

Observed in practice and handled in code:
- The 120B model sometimes cites as `【4†L1-L4】` instead of `[4]`. Citations are normalised after generation, and the prompt names the format.
- Function-calling structured output from the 20B judge intermittently produced malformed JSON (Groq 400 `tool_use_failed`). The judge uses schema-constrained `json_schema` mode, which was 8/8 reliable in testing.

## 8. Chunking

Section-aware splitting (markdown headers from PyMuPDF), ~1,400-character chunks with 200 overlap,
references stripped, and the abstract as its own chunk. Each chunk is *embedded* with its title and
section as a header but *stored and shown* without it, so the extra context improves retrieval without
polluting the prompt. Not ablated separately; I'd measure it with a larger eval set.

## 9. Crash-safe, idempotent ingestion

Point IDs are deterministic (`uuid5(arxiv_id:chunk_index)`), so re-ingest overwrites. The abstract chunk is
written **last** and is the completion marker: a run killed mid-paper leaves a paper that is retried, never
one that looks finished. This was a real bug found when a background job was killed partway.

## 10. Eval design

- A fixed hand-written set: 26 answerable questions with paraphrased reference answers, plus 5 unanswerable ones (4 off-topic, 1 on-topic but absent).
- **Two tiers, because of the token cap (section 1).** Every PR runs a deterministic retrieval gate (recall and MRR floors) that costs no LLM tokens. RAGAS runs nightly on a small rotating subset so a week covers the whole set.
- Decline metrics (decline rate on unanswerable, false-decline rate on answerable) are computed in code, with no LLM judge.
- The gate **fails when a metric was scored on too few samples**. An early run silently scored faithfulness on 2 of 30 samples because the judge was being refused; a gate that passed that would be worse than none.
- RAGAS needed three accommodations for Groq: `strictness=1` (Groq only supports `n=1`), a small adapter over our own embedder (the LangChain wrapper's `model` attribute isn't a string), and `reasoning_effort="low"` with a larger `max_tokens` (gpt-oss spends hidden reasoning tokens from the output budget, which made faithfulness prompts fail with `LLMDidNotFinishException`).
- The eval index is built hermetically from the pinned papers in local Qdrant, never from production.
- Limitation: the judge (gpt-oss-20b) is the same family as the generator, which can bias faithfulness upward. A different judge family would be better.
- **Known flaky case (rag-2, RAG-Sequence vs RAG-Token).** The judge correctly rejects a true claim whose supporting sentence isn't in the retrieved chunks. The sentence lives in a chunk that ranks 5th, and a per-paper diversity cap of 3 can drop it. Raising the cap to 4 did not reliably help in a single comparison (the LLM is non-deterministic and the index was growing), so I did not tune to it. A larger eval set, repeated runs per setting, and parent-section retrieval are the right next steps.

## 11. Operational choices

- **Synchronous `/ingest` is capped at 10 papers.** Cloud Run throttles CPU after a response is sent, so bulk work belongs in the Job.
- **Drift-triggered re-indexing, not a timer.** A fixed daily re-index spends compute and arXiv requests even when nothing is wrong. A daily check that costs no LLM tokens (retrieval canary on the production index, plus decline-rate and top-score drift from request logs) re-indexes only on a breach, with a 24-hour cooldown so a persistent problem alerts a human instead of looping. Trade-offs: the thresholds are initial guesses until there is real traffic; the live-traffic signal is silent below 20 queries; and the canary can only see papers in the fixed set, so a purely stale corpus is detected by traffic or not at all. On this project's own index the canary moved when 24 papers were added (recall 1.0 to 0.981), which shows it is sensitive to real change.
- **Retention policy.** The nightly job evicts the oldest non-pinned papers past a cap, keeping the index far below Qdrant's 1 GB.
- **arXiv politeness.** One request per 3 seconds (stricter than the "3 req/s" in the brief), a descriptive User-Agent, long backoff on 429. arXiv returns `429` with no `Retry-After`, and an early version retried too fast.
- **Licensing.** PyMuPDF is AGPL; fine for this open-source repo, a consideration for closed-source reuse.
- **Secrets.** Never in the repo or image: local `.env` (gitignored), Secret Manager in production, Workload Identity Federation instead of JSON keys.
