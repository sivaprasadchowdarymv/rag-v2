# Project Execution Tracker

Statuses: NOT STARTED · IN PROGRESS · COMPLETED · BLOCKED · FAILED.
COMPLETED requires evidence (test command + result). Checkpoints are git commits in this repository.

| Phase | Title | Status | Evidence / checkpoint |
|---|---|---|---|
| 0 | Repository audit & baseline | COMPLETED | 96/96 tests, app health `ok`, ruff clean, bandit 1 medium; docs/REPOSITORY_AUDIT.md |
| 1 | Modular enterprise architecture | COMPLETED | 106 tests pass (10 new); AppTest renders; health ok |
| 2 | Advanced document ingestion | COMPLETED | 134 tests pass (28 new: 7 formats, OCR, 9 malicious-file cases, 4 chunkers, versioning, jobs+retry) |
| 3 | Advanced retrieval engine | COMPLETED | 146 tests pass (12 new); upgraded ≥ baseline recall/MRR on controlled set (fake embedder) |
| 4 | Multi-LLM integration | COMPLETED | 161 tests pass (15 new); adapters verified against protocol fakes + botocore Stubber; live calls BLOCKED (no network/keys) |
| 5 | Multi-agent orchestration | COMPLETED | 169 tests pass (8 new: classification, full trace, summarisation, timeout, fallback, abstention) |
| 6 | Security & guardrails | COMPLETED | 223 tests pass (54 security: 15 direct-injection variants, 9 benign false-positive checks, indirect injection, PII, harmful, topics, grounding, output leaks, RBAC, lockout, cross-tenant via service + REST API) |
| 7 | RAG evaluation framework | COMPLETED | 228 tests pass (5 new); real retrieval benchmark run (lexical only): MRR 0.690→0.806, NDCG@5 0.755→0.842, recall@5 0.944→0.944 |
| 8 | Professional Streamlit UI | COMPLETED | 247 tests pass (19 UI: all 14 pages render via AppTest, chat round-trip, blocked prompt, debugger scores, policy probe, RBAC gating) |
| 9 | Database & storage | COMPLETED | 258 tests pass (11 new; PostgreSQL 16 variants ran, not skipped); backup→wipe→restore verified on SQLite and PostgreSQL |
| 10 | Observability | COMPLETED | 263 tests (5 new): request-ID propagation, spans + OTel in-memory export, Prometheus metrics, /healthz /readyz /metrics, logs free of question text and PII |
| 11 | CI/CD & deployment | COMPLETED | ruff clean, bandit 0 high/0 medium, pip-audit: no known vulnerabilities (lock), 265 tests pass incl. PostgreSQL, compose config valid for all 4 profiles; image build BLOCKED |
| 12 | Final validation & release | COMPLETED | 268 tests pass (0 skipped, PostgreSQL on); live stack smoke 17/17; CLI backup/restore on PostgreSQL; ruff/bandit/pip-audit clean |
| 13 | 2.1 · Audit both archives, baseline, rollback point | COMPLETED | baseline 277 passed / 4 skipped, ruff clean, bandit 0 high/0 medium; git tag `baseline-v1`; UX reference `uxui-workflow` read |
| 14 | 2.1 · D. Safe repeated-question cache | COMPLETED | 26 new tests incl. two-tenant identical-file isolation; live API: 2nd query `cached=true`, 0 tokens |
| 15 | 2.1 · E. Metadata intelligence | COMPLETED | 18 new tests (extraction, SI units, provenance, edits, scopes, filters, API RBAC) |
| 16 | 2.1 · A/B/C. UI redesign, onboarding, help assistant | COMPLETED | 18 new AppTest UI tests; 21 original UI tests unchanged; headless-Chrome visual review (desktop + phone) |
| 17 | 2.1 · F. Analytics & admin | COMPLETED | Dashboard KPIs + ⚡ Cache & help page, real counters only (UI tests) |
| 18 | 2.1 · G. Regression, security, compatibility | COMPLETED | 344 passed / 4 skipped; 348/0 with PostgreSQL 16; ruff clean; bandit 0 high/0 medium; pip-audit clean; benchmark identical; smoke 17/17; compose valid; image build BLOCKED (no Docker daemon) |
| 19 | 2.1 · Docs & delivery | COMPLETED | UI_UX_UPGRADE_REPORT.md, IMPLEMENTATION_SUMMARY.md, CHANGELOG 2.1.0, USER_GUIDE, BEGINNER_GUIDE, ARCHITECTURE diagrams, checklist, test report, ZIP |
| 20 | 2.2 · Independent free-model fallback chains | COMPLETED | 50 new tests (all 20 required scenarios + catalogue/cost-guard extras, mocked APIs); 394 passed / 4 skipped, 398/0 with PostgreSQL 16; ruff clean; bandit 0 high/0 medium; pip-audit clean; live smoke 17/17 with discovery on (AnyRouter key rejected → fail closed, Groq answered); docs/MODEL_ROUTING.md |

## Phase log

### Phase 0 — COMPLETED
- Baseline commit `8dfaa76`. No prior tracker existed in the ZIP; work starts here.
- Created: `docs/REPOSITORY_AUDIT.md`, `docs/EXISTING_ARCHITECTURE.md`, `docs/TECHNICAL_DEBT.md`,
  `docs/IMPLEMENTATION_ROADMAP.md`, this tracker.

### Phase 1 — COMPLETED
- New: `core/service.py` (RAGService: single server-side entry, DI for router/limiter/policy/repo/workflow), `config/logging_config.py` (JSON/text logs, request IDs, secret+PII redaction), `database/` (SQLAlchemy models, repository, versioned migrations), `security/policy_engine.py` v1 (injection screen, document ACL, audit), `agents/workflow.py` v1.
- Modified: `config/settings.py` (all 2.0 settings, defaults keep 1.x behaviour), `rag/models.py` (QueryResult: request_id, security, retrieval_debug, agent_trace, routing), `ui/pages.py` (ask → RAGService), `ui/state.py`, `app.py`.
- Decision: target layout adopted by *adding* packages; existing `rag/ llm/ retrieval/ agents/ storage/` stay (mapping in ARCHITECTURE.md).
- Tests: `tests/unit/test_core_and_db.py`.

### Phase 2 — COMPLETED
- New: `ingestion/validation.py` (magic-byte sniffing, allow-list, size, macro/zip-bomb/path-traversal/executable rejection), `ingestion/loaders/` (DOCX, PPTX, XLSX, CSV, TXT, MD, HTML, JSON, PNG/JPG), `ingestion/ocr.py` (tesseract, offline; scanned PDF pages + images), `ingestion/chunker.py` (recursive, fixed, semantic, structure), `ingestion/processor.py`, `ingestion/jobs.py` (sync + background ingestion, retries with backoff, failed-job tracking, progress), `storage/objects.py` (local / S3 originals), `security/pii_detector.py` (auto-classifies PII documents as confidential).
- Modified: `rag/pipeline.py` (any format), `rag/pdf_parser.py` (CHUNK_STRATEGY), `storage/cache.py` (keep real extensions), UI uploader (all formats, sensitivity).
- Versioning: same name + new content → new version; superseded vectors deleted; ACL denies superseded hashes. Delete removes DB rows, vectors and originals.
- Bugs found+fixed during testing: heading-only text produced no chunk (OCR, title slides); RLock deadlock in job pool.
- Decision: semantic chunking is lexical-cohesion (TextTiling-style), not embedding-based, to stay deterministic/offline.

### Phase 3 — COMPLETED (live-model evaluation BLOCKED)
- New: `retrieval/query_rewriter.py` (filler removal, abbreviation expansion, synonyms, multi-query variants, optional LLM paraphrase), `retrieval/hybrid_search.py` (HybridSearcher: dense / bm25 / fuzzy / hybrid-RRF / rerank modes with every score exposed; pre-filtering MetadataFilter; parent-child; compression).
- Modified: `agents/retrieval_stage.py` (variants vote in RRF as `hybrid@q1` etc., metadata filters, small-to-big parent expansion within the token budget, optional compression, full debug trace), `agents/rag_pro.py` (filters, debug in QueryResult, cache key includes retrieval settings).
- Existing (kept): dense+fuzzy+BM25 hybrid, RRF, cross-encoder rerank, configurable TOP_K.
- BLOCKED: real nomic/MiniLM models cannot be downloaded in this sandbox (HF blocked). Offline tests use the repo's deterministic fake embedder/cross-encoder, so dense/rerank *quality* numbers are not representative; run `python -m evaluation.benchmark` on a machine with model access.

### Phase 4 — COMPLETED (live provider calls BLOCKED)
- New: `llm/adapters.py` (Ollama native API incl. tools/images/embeddings, OpenAI-compatible, Anthropic Messages incl. tool_use/tool_result/images, Bedrock Converse via boto3), `llm/pricing.py` (cost estimates, context windows, LLM_PRICES_JSON override).
- Rewritten: `llm/model_router.py` — candidates → exclusions (not configured, LLM_MODE local/cloud, **sensitive data never to cloud unless ALLOW_SENSITIVE_CLOUD + CLOUD_PROVIDERS_ALLOWED**, vision capability, context window, circuit open) → strategy (priority/cost/latency/auto) → retries with backoff + Retry-After → fallback; usage/cost/latency per model; routing explanation in every QueryResult.
- Modified: `llm/catalog.py` (chain_for_settings, MANUAL_MODEL), `llm/providers.py` (prompt/completion tokens, Retry-After), `rag/embeddings.py` (EMBED_BACKEND=ollama), settings.
- Security decision (default, reversible): documents with detected PII or labelled confidential/restricted are routed to local models only.
- BLOCKED: real Ollama (registry.ollama.ai, ollama.com, GitHub downloads refused by sandbox proxy), real OpenAI/Anthropic/Bedrock (no credentials). Verified: wire formats against a strict fake server (tests/fakes/fake_llm_server.py) and botocore Stubber.

### Phase 5 — COMPLETED
- New: `agents/workflow.py` v2 — QueryUnderstandingAgent (task: question/summarization/calculation/comparison/figure/greeting), SummarizationAgent (section-level whole-document evidence), AnswerQualityAgent (accept/warn/abstain), per-agent timeouts (thread futures), fallbacks for every agent, execution trace without document text.
- Reused (not duplicated): retrieval specialists (`agents/retrieval_stage.py`), ReasoningAgent (`agents/master.py`, ReAct bounded by AGENT_MAX_STEPS), CitationVerificationAgent (`agents/verifier.py`, ≤ MAX_REGENERATIONS).
- `answer_pro(retrieval_override=…)` lets the summariser supply evidence while reasoning + verification stay identical.
- Design: fixed sequence — agents cannot call each other, so loops are impossible; greetings and no-document requests skip the LLM entirely.
- Limitation: a timed-out agent thread cannot be killed in Python; its result is discarded (documented).

### Phase 6 — COMPLETED
- New `security/`: `injection_detector.py` (7 categories incl. obfuscation: zero-width chars, base64 payloads), `pii_detector.py` (11 types, Luhn/IBAN/Verhoeff checks; block/redact/mask/allow), `harmful_content.py` (6 categories, supportive self-harm response, tenant blocked terms), `topic_guard.py` (restricted + allowed topics, out-of-scope message), `grounding_guard.py` (claim-to-source matching, abstention), `output_guard.py` (secrets, system-prompt leakage, PII output policy incl. 'authorized'), `policy_engine.py` v2 (tenant overrides from DB, audit), `auth.py` (PBKDF2 600k, lockout, password policy, RBAC admin/editor/viewer, API tokens hashed).
- New `api/` (FastAPI): token auth, documents CRUD + permissions, query, retrieval debug, jobs, audit, health — all through RAGService/IngestionService (server-side enforcement).
- New `core/documents.py` (tenant-scoped readable indexes). UI: AUTH_MODE=users login, per-account workspace, DB-backed document list.
- Real false positive found by tests and fixed: 'kill a hung process' matched the violence rule.
- Residual risks + FP/FN limitations: docs/SECURITY_AUDIT.md (Phase 12).

### Phase 7 — COMPLETED (generation quality on real models BLOCKED)
- New `evaluation/`: `corpus.py` (deterministic 4-file corpus: PDF datasheet, DOCX, XLSX, MD), `test_datasets/benchmark_v1.json` (18 questions with expected answer, expected source file+page, relevant context, difficulty), `metrics.py` (P@K, R@K, hit, MRR, NDCG; answer correctness, faithfulness, answer/context relevance, citation correctness, hallucination; p50/p95), `benchmark.py` (baseline 1.x vs upgraded 2.0; DB storage + JSON/CSV/Markdown export), `ragas_evaluator.py` (optional; reports 'unavailable' instead of scores).
- Real run in this sandbox: docs/evaluation/BASELINE_VS_UPGRADED_lexical.md (+ .json/.csv). Dense retrieval and reranker could not load (HF download blocked) → lexical-only numbers, stated in the report.
- **Real defect found and fixed**: without a downloadable model, every embed/rerank call retried the download for ~40 s. Added `rag/model_failures.py` negative cache (MODEL_RETRY_S, default 600 s): offline mode now fails fast and falls back to keyword search.
- BLOCKED: generation metrics with a real LLM (no provider reachable). The generation path is exercised in tests with the scripted fake model only to verify mechanics; those numbers are not quality measurements and are not reported as such.

### Phase 8 — COMPLETED
- New `ui/admin_pages.py`: Dashboard, Ingestion monitor (background upload, progress, failed docs), Retrieval debugger (original/rewritten queries, dense/BM25/fuzzy/RRF/rerank scores, filters, final context, parent), Agent traces, Evaluation (run + history + downloads), Models (providers, routing preview with exclusions, usage/circuits, tenant chains), Security (policy editor, usage limits, prompt probe, events), Users & roles, Audit log (filter, CSV), document permissions + version history + delete.
- Chat: model-mode and model selectors, streamed display of the verified answer (never before verification), agent workflow trace, 'Why this model' routing explanation, distinct rendering for blocked/out-of-scope/abstained. Copy + feedback kept.
- Navigation grouped (Workspace / Operations / Administration); admin pages hidden and guarded by permission.
- `core/model_config.py`: tenant model chains and usage limits from the admin UI applied server-side (UI and API).
- Test harness `tests/ui_harness.py` renders any page for a signed-in principal.

### Phase 9 — COMPLETED (pgvector not available)
- Database: SQLite (local) and PostgreSQL (production) through one repository; all 14 tables; migration 0002 (conversation archiving + composite indexes) proves in-place upgrades of existing data; m0001 now creates the exact v1 schema.
- Persisted: users, roles, tokens (hashed), documents, versions, chunk metadata, ingestion jobs, conversations + messages (`chat/db_store.py`, used with AUTH_MODE=users, scoped per user), feedback, model chains, evaluation runs, audit events, policies.
- Object storage: local filesystem or S3/MinIO (`storage/objects.py`), tested with moto.
- Backup & recovery: `scripts/backup.py` create/verify/restore — SQLite online backup API, PostgreSQL pg_dump/pg_restore or portable JSON (with sequence repair), data tar.gz, SHA-256 manifest, tamper detection, safe extraction.
- Passwords: PBKDF2 hashes only (Phase 6).
- Not implemented: pgvector — the extension is not installed on this PostgreSQL and cannot be fetched; vectors stay in the NumPy store keyed by content hash (documented trade-off; interface isolated in storage/vector_store.py).

### Phase 10 — COMPLETED
- New `monitoring/`: `metrics.py` (Prometheus: HTTP, queries, stage/query latency, LLM calls/tokens/cost, guardrail events, ingestion, errors; label values never contain user text), `tracing.py` (in-process spans per request; OpenTelemetry export when OTEL_EXPORTER_OTLP_ENDPOINT + SDK installed), `analytics.py` (usage by model/day from telemetry).
- Instrumented: RAGService (spans policy.input/authorize/workflow/retrieval/policy.output; one summary log line without content), router (per-model outcomes), ingestion, policy engine; API middleware (X-Request-ID in/out, access log, HTTP metrics).
- Bugs found by tests and fixed: (1) agent thread pool dropped request context → contextvars copied into agent threads; (2) service minted a second request id → inbound X-Request-ID now propagates end to end.
- Deployment: `deployment/prometheus/prometheus.yml`, Grafana provisioning + 10-panel dashboard, Caddy reverse proxy (HTTPS, security headers, metrics not public — ordering bug fixed before release).
- Monitoring stays optional (METRICS_ENABLED, no OTel by default).

### Phase 11 — COMPLETED (container build BLOCKED in sandbox)
- Dependencies: `requirements.txt` (ranges), `requirements.lock` (pip-compile, 241 lines, exact pins), `requirements-optional.txt`, `requirements-dev.txt`; `pyproject.toml` (ruff E9/F/B, bandit config).
- Dockerfile: multi-stage wheels → slim runtime with tesseract + pg client, non-root uid 10001, healthcheck for UI or API, JSON logs.
- docker-compose.yml profiles: local (Ollama + app, SQLite), hybrid (+ PostgreSQL, API), production (app, API, PostgreSQL, MinIO, Caddy HTTPS, backup job), monitoring (Prometheus, Grafana). Validated with `docker compose config`.
- CI (`.github/workflows/ci.yml`): lint, bandit, pytest with PostgreSQL service + tesseract, benchmark artifact, pip-audit --strict, Docker build + smoke test + Trivy; release workflow → GHCR.
- Security fixes from scanning: literal bidi/zero-width chars in source (B613) → escapes; Pillow 12.3 (39 advisories) → 13.0; reviewed and annotated B608/B104/B615.
- Also: `scripts/create_admin.py`, Makefile, .env.example documents every setting (checked by script).
- BLOCKED: `docker build` (Docker Hub refused by sandbox proxy, no local base image); Trivy not installable; GitHub Actions not executed (no push requested). The same lint/test/audit commands were run locally.

### Phase 12 — COMPLETED
- Live validation (real processes): API (uvicorn) + UI (Streamlit) + PostgreSQL 16, AUTH_MODE=users, JSON logs — `scripts/smoke_test.py` 17/17 (`docs/evaluation/smoke_test_results.json`); backup → verify → restore via CLI.
- **Defect found by the live run and fixed**: first question took 119.5 s (three ~40 s blocked model downloads on the request path) → model loads are cache-only on the query path with one background download (`rag/model_failures.py`); now 0.32 s. Also fixed answer_relevance comparing vectors from two embedding models.
- Performance tests added (`tests/performance`).
- Release docs: README, ARCHITECTURE.md (final diagrams), CHANGELOG.md, docs/FINAL_TEST_REPORT.md (27-item matrix), SECURITY_AUDIT.md, DEPLOYMENT_GUIDE.md, USER_GUIDE.md, ADMIN_GUIDE.md, PRODUCTION_READINESS_CHECKLIST.md.

### Phases 13-19 — 2.1 UI/UX + intelligence upgrade — COMPLETED
- Modified files (additive): `ops/answer_cache.py`, `ops/telemetry.py`, `agents/rag_pro.py`, `agents/workflow.py`,
  `core/service.py`, `config/settings.py`, `rag/models.py`, `retrieval/hybrid_search.py`, `ingestion/jobs.py`,
  `api/main.py`, `api/schemas.py`, `app.py`, `ui/{pages,admin_pages,guide,help_content,widgets,styles}.py`,
  `tests/ui_harness.py`, `.env.example`, docs. New: `rag/metadata_intel.py`, `storage/metadata_store.py`,
  `ui/faq.py`, 4 test files, reports, diagrams, screenshots.
- Decisions: in-memory cache only (Redis rejected: pickle-over-network risk + new service); metadata stored as
  JSON next to each document version (no DB migration needed); default retrieval ranking untouched (benchmark
  identical), metadata improves precision through explicit filters; help assistant deterministic (no LLM).
- Pre-existing defects fixed: cache hit crashed the Chat page; live agent progress never rendered.
- Remaining issues: container build not runnable here; answer quality with real models unmeasured here (as in 2.0);
  per-process cache in multi-replica setups; pre-existing stray `*.whl` files in the repo root left in place.

### Phase 20 — 2.2 free-model fallback chains — COMPLETED
- New: `llm/discovery.py`, `docs/MODEL_ROUTING.md`, `tests/unit/test_free_model_routing.py`,
  `tests/integration/test_routing_ui.py`. Modified: `llm/providers.py` (AnyRouterProvider, Groq models_raw),
  `llm/catalog.py` (discovery chains, cost_guard), `llm/model_router.py` (discovery retry/skip semantics, image
  validation, attempt log, public pipeline methods), `config/settings.py`, `ui/admin_pages.py` (routing page),
  `ui/help_content.py`, `ui/faq.py`, `app.py`, `tests/ui_harness.py`, `.env.example`, docs.
- Decision: exact AnyRouter JSON keys are not publicly documented → classifier accepts common shapes and rejects
  anything unreadable (fail closed). Virtual aliases (anyrouter/free) not used.
- Remaining: real Groq/AnyRouter calls not possible in the sandbox; verify on your account with the routing page.

## Blocked items (need your infrastructure or credentials)
| Item | Why blocked here | How to finish |
|---|---|---|
| Live Ollama answers | registry.ollama.ai / ollama.com refused by sandbox proxy | `ollama pull llama3.1:8b`, set OLLAMA_BASE_URL, run smoke test |
| Live OpenAI / Anthropic / Bedrock | no credentials | add keys, run `python -m evaluation.benchmark --generation` |
| Real embedding + reranker quality | Hugging Face downloads refused | let fastembed download once (or pre-seed data/models), rerun benchmark |
| docker build, Trivy, compose up | Docker Hub refused, no local base image | push to GitHub (CI builds + scans) or build on your host |
| GitHub Actions run | no push requested | push the repository |

## Continuation
All 20 phases (0-19, incl. the 2.1 upgrade) are complete; each is a git commit in this repository (`git log --oneline`). Next steps are the operator items in docs/PRODUCTION_READINESS_CHECKLIST.md.
