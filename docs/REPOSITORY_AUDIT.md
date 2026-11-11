# Repository Audit — Phase 0 baseline

Audited: `rag-ultimate-main.zip` (99 files, ~10 k lines of Python/Markdown), 2026-11-10.
No previous `PROJECT_EXECUTION_TRACKER.md` or checkpoint existed in the archive, so the
transformation starts from Phase 0. Baseline commit: `8dfaa76`.

## Baseline evidence

| Check | Command | Result |
|---|---|---|
| Unit/integration tests | `python -m pytest -q` | **96 passed** (31 s, fully offline: fake Groq/OpenRouter server + fake fastembed) |
| Lint (errors only) | `ruff check . --select E9,F` | clean |
| Static security | `bandit -r . -x ./tests -ll` | 1 medium (B615: HF `snapshot_download` without revision pin), 8 low |
| App start | `streamlit run app.py --server.headless true` | starts, `/_stcore/health` → `ok` |
| Environment | Python 3.13, 2 vCPU, 7 GB RAM, tesseract, PostgreSQL 16, Docker daemon | available |

## Feature inventory

| Area | Status | Where | Notes |
|---|---|---|---|
| PDF parsing (text, tables, equations, figures) | **Functional** | `rag/pdf_parser.py`, `rag/table_extractor.py`, `rag/equation_extractor.py`, `rag/figure_extractor.py` | PyMuPDF 2.x |
| Other formats (DOCX, PPTX, XLSX, CSV, TXT, MD, HTML, JSON, images) | **Missing** | — | `validate_pdf_bytes` rejects non-PDF |
| OCR of scanned pages | **Partial** | `rag/vision.py` | Cloud vision model only (OpenRouter); no local OCR |
| Chunking | **Functional (1 strategy)** | `rag/chunking.py` | section-aware recursive; no fixed / semantic options |
| Embeddings | **Functional** | `rag/embeddings.py` | local fastembed (nomic) or OpenRouter Qwen3 |
| Vector store | **Functional** | `storage/vector_store.py` | NumPy matrix per document; justified for datasheet scale |
| Dense + fuzzy + BM25 + RRF + cross-encoder rerank | **Functional** | `rag/retrieval.py`, `retrieval/*`, `agents/specialists.py` | good base |
| Query rewriting / expansion / multi-query | **Partial** | `agents/orchestrator.py` (follow-up rewrite), `concepts/expand.py` | no general rewriter |
| Metadata filtering, parent-child, compression | **Missing** | — | |
| Retrieval debugger | **Missing** | — | agent stats only |
| LLM providers | **Partial** | `llm/providers.py` | Groq + OpenRouter only (both cloud). No Ollama/OpenAI/Anthropic/Bedrock |
| Model routing + fallback | **Functional (role chains)** | `llm/model_router.py`, `llm/catalog.py` | no local/cloud modes, no sensitivity routing, no health/circuit breaker, no retries with backoff |
| Agents | **Functional (deterministic)** | `agents/*` | rule-based planner, 5 retrieval specialists, master (ReAct tools), verifier |
| Summarisation agent / answer-quality agent / per-agent timeouts | **Missing** | — | |
| Prompt-injection guard | **Partial** | `ops/guard.py` | regex neutralisation of document lines; user input not screened |
| PII / harmful content / topic / output guards | **Missing** | — | |
| Grounding | **Functional** | `agents/verifier.py` | numeric + citation + cross-encoder checks; abstention is weak |
| Authentication | **Minimal** | `ui/state.py` | single shared `APP_PASSWORD`; no users/roles |
| Tenant isolation | **Partial** | `storage/workspace.py` | unguessable link-based workspaces; document caches are shared by content hash |
| Persistence | **Functional (files)** | `chat/`, `learning/`, `storage/` | JSON files + optional HF dataset sync; no RDBMS, no migrations |
| Evaluation | **Partial** | `metrics/benchmark.py`, `metrics/validation.py`, `metrics/rag_metrics.py` | P@K/R@K/MRR/NDCG exist; no faithfulness/hallucination rate, no stored runs |
| Telemetry | **Partial** | `ops/telemetry.py` | JSONL request log (no question text); no request IDs, tracing, Prometheus |
| API / health endpoints | **Missing** | — | only Streamlit's `/_stcore/health` |
| Docker | **Functional (basic)** | `Dockerfile`, `docker-compose.yml` | single service, no profiles |
| CI/CD | **Missing** | — | no `.github/workflows` |
| UI | **Functional** | `ui/pages.py` (5 pages) | Chat, Library, Validate, Insights, Settings |

## Key findings

1. **Strong core** — retrieval, verification and the offline test harness are well engineered and must be preserved.
2. **Cloud-only generation** — every LLM path needs a Groq/OpenRouter key; the master prompt requires a fully offline mode.
3. **No data-sensitivity controls** — any configured cloud provider receives document text.
4. **Security is single-layer** — no input screening, PII handling, output filtering, RBAC or audit trail.
5. **Shared document caches** — two workspaces uploading the same PDF share one cache (same hash). Fine for
   de-duplication, but authorization must be enforced at retrieval time, not by path secrecy.
6. **No relational store** — needed for users, roles, document versions, jobs, audit events, evaluation runs.
7. **PDF-only ingestion** — main functional gap for an enterprise platform.
