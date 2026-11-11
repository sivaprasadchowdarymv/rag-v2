# Changelog

## 2.2.1 — 2026-11-11

### Added
- **Backup API keys**: `GROQ_API_KEY_2` … `_9` and `OPENROUTER_API_KEY_2` … `_9` (or `GROQ_API_KEYS` /
  `OPENROUTER_API_KEYS` comma lists). On 401 (rejected), 402 (out of credit) or 429 (rate limit) the same request is
  repeated with the next key before the next model is tried. Rate limits cool a key for that model only; rejected
  keys are re-checked after 1 hour. Keys are stored only as hashes in memory and shown as "key #n" on the Models page.
- OpenRouter free-model discovery (`:free` id **and** zero pricing) and chain placeholders `@groq`,
  `@anyrouter_text`, `@anyrouter_vision`, `@openrouter_text`, `@openrouter_vision` to combine all three providers.
- `docs/env.complete.example`: a complete `.env` with Groq + AnyRouter + OpenRouter free chains.

- **`launch.py`** — one launcher: `setup` (venv, packages, .env from questions, migrations), `check` (pre-flight,
  `--online` key test), `run [--api]`, `secrets` (.env → Streamlit Cloud TOML with deployment warnings), `docker
  up|down|logs|status|backup`. `.streamlit/secrets.toml.example` now covers the upgraded features.
  Guide: docs/STREAMLIT_CLOUD_DEPLOY.md.
- Docker Compose **PROFILE D `cloud`**: Streamlit UI + REST API + PostgreSQL without the Ollama container; native
  PostgreSQL steps in docs/DEPLOYMENT_GUIDE.md.

### Fixed
- `.env.example`: 15 settings written as `KEY=   # comment` were read by python-dotenv as the comment text (e.g.
  `APP_PASSWORD` turned on a password prompt, `OLLAMA_BASE_URL` pointed to a junk URL, `DATA_DIR` created a folder
  named `# default ./data`). Comments now sit on their own line, and the settings loader ignores such values.

## 2.2.0 — 2026-11-11 — Independent free-model fallback chains

Backwards compatible: without `ANYROUTER_API_KEY` (or with `MODEL_DISCOVERY=off`) routing is exactly 2.1.

### Added
- **Dynamic model discovery** (`llm/discovery.py`): Groq `/models` → primary + 3 fallbacks; AnyRouter `/models`
  → 2 free text + 3 free vision models. No hard-coded ids; ranked, cached (memory + `model_catalog.json`), refreshed
  every `MODEL_CATALOG_REFRESH_MINUTES`, re-validated before reuse, fail-closed.
- **Free-only cost guard** (`llm/catalog.cost_guard`): a model is free only if `can_route` is true, `access` is a free
  route (not BYOK, not credit-priced) and every price is zero. Paid providers need `MODEL_ALLOW_PAID_FALLBACK=true`
  and `MODEL_FREE_ONLY=false`. A 402 or reported cost excludes a model at once.
- `AnyRouterProvider`; `ModelRouter.generate_text()`, `analyze_image()`, `generate_multimodal_response()`.
- Retries on 429/5xx/timeout/connection with exponential backoff (`MODEL_MAX_RETRIES`, `MODEL_REQUEST_TIMEOUT`).
  Invalid keys and missing entitlement skip that provider without retrying; bad input and content-policy refusals
  stop at once.
- Image validation before any vision call (type, size, encoding); images are refused on text roles.
- Health tracking with automatic exclusion and replacement (`MODEL_FAILURE_EXCLUDE`); structured attempt log.
- **🔀 Model routing** admin page: both chains, free verification, vision capability, health, attempts, latency,
  refresh time, Test connection / Test vision model.
- `docs/MODEL_ROUTING.md`; 50 new tests (mocked APIs, no credits).

## 2.1.0 — 2026-11-11 — UI/UX + intelligence upgrade

Backwards compatible: no endpoint, setting, table, page or workflow removed or renamed. 277 original tests pass
unchanged; 67 new tests (344 passed + 4 PostgreSQL variants = 348 with `TEST_DATABASE_URL`).

### Added
- **Intelligent verified-answer cache**: canonical + equivalent-question matching with critical-term guard;
  context isolation by tenant, user/roles, workspace, document versions, filters, models, answer settings and
  security policy; permission revalidation on hit; invalidation on new version/delete; TTL, LRU, enable switch;
  statistics, savings and not-cached reasons; per-tenant clear. Settings `ANSWER_CACHE_ENABLED`,
  `ANSWER_CACHE_SIMILARITY`, `ANSWER_CACHE_SCOPE`. API: `cached`/`cache` in `/v1/query`, `GET /v1/cache/stats`,
  `DELETE /v1/cache`.
- **Metadata intelligence**: per-version profile (title, author, type, size, pages, category, tags, version,
  sensitivity, owner, part numbers, standards, technical parameters with SI-normalised units, references) with
  source (auto/file/you), confidence and page provenance; user edits per tenant/workspace; library search/filter/
  sort; Chat filters; document- and passage-level metadata filters; API `GET/PATCH /v1/documents/{id}/metadata`,
  `GET /v1/metadata/search`.
- **❔ Help assistant** on every page (deterministic FAQ, 47 answers, 12 categories, page-aware, search, quick
  replies, navigation, fallback to help docs, admin-managed custom questions) + Guide → *Help center* tab.
- **UI/UX**: new design system, uniform page headers, help panels answering what/why/how/after/errors, onboarding
  checklist, empty states, answer execution summary, ⚡ From cache badge, KPI dashboard, **⚡ Cache & help** admin
  page, mobile-friendly sidebar, keyboard focus styles.
- Telemetry: optional `saved_tokens`, `saved_cost`, `verdict`, hashed tenant/user; summary adds cache, verification
  rate, active users, blocked, abstained, average latency.

### Fixed
- Chat crashed when an answer was served from the cache (unguarded step callback in an agent thread).
- Live "Thinking…" progress was never shown (agent threads lacked the Streamlit session context); the context is
  now bound only for each write and restored, so progress can never leak between sessions.

## 2.0.5 — 2026-11-10

### Added
- **🎓 Guide & Test Lab** page: setup check with fixes, upload and question pipelines explained step by step,
  one-click Test Lab (sample documents with known answers; 6 answer tests and 5 safety tests run through the real
  RAGService and are auto-checked: ✅ pass · ❌ fail · ➖ not tested, never a pass that was not checked), checklist.
- **ℹ️ What is this page?** help on every page (what, how to use, what to try, expected result).
- **🧑‍🏫 How this answer was made**: plain-English explanation under every answer, built from the real security
  decisions, agent trace, retrieval, routing and verification. *Beginner help* toggle in the sidebar.
- Chat onboarding: load the sample documents or open the guide when no documents exist.

## 2.0.0 — 2026-11-10

Production platform release. All 1.x functionality preserved (96 original tests still pass unchanged in
behaviour); 172 new tests.

### Added
- **Ingestion**: DOCX, PPTX, XLSX, CSV, TXT, Markdown, HTML, JSON, PNG/JPG (tesseract OCR) besides PDF; local OCR
  of scanned PDF pages; magic-byte validation, macro/encryption/zip-bomb/executable rejection; chunk strategies
  recursive · fixed · semantic · structure; document versioning, de-duplication, deletion without stale vectors;
  background ingestion with retries and failed-job tracking; originals in local or S3/MinIO storage.
- **Retrieval**: query rewriting, abbreviation/synonym expansion, multi-query RRF, metadata pre-filters,
  parent-child (small-to-big), context compression, `HybridSearcher` with every score exposed, debugger page.
- **Models**: Ollama (chat, tools, vision, embeddings), OpenAI-compatible, Anthropic, AWS Bedrock; local/cloud/
  hybrid modes; manual model; sensitivity-aware routing (confidential → local only); context-window checks;
  cost/latency strategies; retries with backoff + Retry-After; circuit breaker; token and cost tracking.
- **Agents**: controlled workflow — query understanding, retrieval or summarisation, reasoning, citation
  verification, answer quality (accept/warn/abstain); timeouts, fallbacks, execution trace.
- **Security**: injection (direct, indirect, obfuscated), PII (11 types), harmful content, topic restrictions,
  grounding guard, output guard (secrets, prompt leakage, PII), tenant policies; accounts, RBAC, tenants, API
  tokens, lockout, audit log.
- **API**: FastAPI with token auth, documents, query, retrieval debug, jobs, audit, `/healthz` `/readyz` `/metrics`.
- **Data**: SQLAlchemy schema (SQLite / PostgreSQL), versioned migrations, DB conversations, backup/verify/restore.
- **Evaluation**: controlled corpus + 18-question dataset, retrieval/generation/operational metrics,
  baseline-vs-upgraded runner, stored runs, JSON/CSV/Markdown export, optional RAGAS adapter.
- **Observability**: JSON logs with request ids and redaction, spans + OpenTelemetry, Prometheus metrics,
  Grafana dashboard, usage analytics.
- **UI**: Dashboard, Ingestion, Retrieval debugger, Agent traces, Evaluation, Models, Security, Users & roles,
  Audit log; model/mode selectors; streamed verified answers; routing explanation.
- **Deployment**: lock file, multi-stage non-root Dockerfile, compose profiles local/hybrid/production/monitoring,
  Caddy HTTPS, GitHub Actions CI (lint, bandit, tests with PostgreSQL, pip-audit, image build + Trivy) and release.

### Fixed
- Offline mode: local model loads no longer stall questions (~40 s per call before; first question 119.5 s →
  0.32 s); failed downloads are not retried for MODEL_RETRY_S.
- Answer relevance compared vectors from two different embedding models when remote/Ollama embeddings were used.
- Heading-only page text produced no chunk; uploads were always renamed to `.pdf`.
- See `docs/FINAL_TEST_REPORT.md` for the full defect list.

### Security
- Pillow ≥ 13.0 (39 advisories in 12.x); invisible/bidi characters removed from source; metrics not public.

### Known limitations
- pgvector not used (NumPy vector store); live Ollama/cloud providers and container builds not verified in the
  build environment — see the test report.
