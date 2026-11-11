# Final Test Report — RAG Ultimate 2.2.0 (free-model fallback chains)

| Gate | Result |
|---|---|
| `python -m pytest` | **394 passed, 4 skipped** (PostgreSQL variants) |
| `TEST_DATABASE_URL=postgresql://… python -m pytest` | **398 passed** |
| `ruff check .` / `bandit -ll` / `pip-audit` | clean / 0 high, 0 medium / no known vulnerabilities |
| Live smoke (uvicorn + Streamlit, discovery on, fake LLM server) | **17/17**; Groq catalogue discovered and ranked; AnyRouter key rejected by the server → AnyRouter disabled with a key-free message (fail closed); no key in logs |
| New tests | `tests/unit/test_free_model_routing.py` (46), `tests/integration/test_routing_ui.py` (4) — mocked HTTP, no credits |

Required scenarios → tests: 1–6 `test_text_chain_falls_back_in_order[groq-primary … anyrouter-text-2]` · 7–9
`test_vision_chain_falls_back_in_order[vision-1..3]` · 10 `test_all_text_models_unavailable` · 11
`test_all_vision_models_unavailable_keeps_text_answer` · 12–14 `test_non_free_models_are_rejected[paid…, byok…,
unknown…]` + `test_paid_model_never_called_even_when_configured_explicitly` · 15 `test_vision_capability_is_verified`,
`test_model_rejecting_image_format_is_replaced` · 16 `test_image_encoding_errors_fail_before_any_call` · 17
`test_429_is_retried_with_backoff_then_succeeds`, `test_429_retries_are_bounded_then_next_model` · 18
`test_invalid_groq_key_skips_groq_and_uses_anyrouter_without_retry`, `test_invalid_anyrouter_key_for_vision`,
`test_free_tier_entitlement_missing_is_reported` · 19 `test_answer_cache_isolated_between_users_and_documents` · 20
`test_multimodal_citations_preserved_when_vision_fails`, `test_multimodal_answer_uses_vision_chain_and_keeps_figure_page`.

---

# Final Test Report — RAG Ultimate 2.1.0 (UI/UX + intelligence upgrade)

Date: 2026-11-11 · Python 3.13.18 · PostgreSQL 16 · Streamlit 1.67 · Linux x86-64.

| Gate | Command | Result |
|---|---|---|
| Baseline (original ZIP, before any change) | `python -m pytest` | 277 passed, 4 skipped (PostgreSQL not set) |
| Full suite after 2.1 | `python -m pytest` | **344 passed, 4 skipped** |
| Full suite incl. PostgreSQL 16 | `TEST_DATABASE_URL=postgresql://… python -m pytest` | **348 passed, 0 skipped** |
| Lint | `ruff check .` | clean |
| Static security | `bandit -c pyproject.toml -r . -ll` | 0 high, 0 medium (17 low: try/except/pass in UI display guards) |
| Dependencies | `pip-audit -r requirements.lock` | no known vulnerabilities |
| Retrieval regression | `python -m evaluation.benchmark` | identical to 2.0: MRR 0.8056, NDCG@5 0.8419, recall@5 0.9444 — `docs/evaluation/POST_2.1_REGRESSION_lexical.md` |
| Live smoke test (uvicorn + Streamlit + fake LLM) | `python -m scripts.smoke_test …` | **17/17** — `docs/evaluation/smoke_test_results_2.1.json` |
| Live API cache check | 2× `POST /v1/query` | 1st `stored` (928 tokens), 2nd `cached=true`, `hit`, 0 tokens, 1 ms; abstained answer → `skipped: not verified` |
| Compose profiles | `docker compose --profile … config` | local, hybrid, production, monitoring valid |
| Visual review | headless Chrome, 1400 px + 390 px | screenshots in `docs/screenshots/`; 3 layout fixes applied |
| Image build | `docker build` | **BLOCKED** — no Docker daemon in the sandbox |

## New tests (67)

| File | Tests | Covers |
|---|---|---|
| `tests/unit/test_answer_cache_v2.py` | 26 | normalisation/equivalence, critical terms, contexts, TTL expiry, LRU, permission revalidation, invalidation, per-tenant clear, disabled cache, v1 API, savings, filters/settings in key, follow-ups, **cross-tenant isolation with identical files**, tenant scope only for same roles, delete invalidation, blocked never cached, quality rejection discards, API flags + admin endpoints |
| `tests/unit/test_metadata_intel.py` | 18 | SI units, datasheet profile (provenance, confidence, table params, part numbers, refs, category), dedupe, user override, sanitisation, scope isolation, filter resolution, passage filters, service filters, API RBAC + tenant isolation |
| `tests/integration/test_ui_upgrade.py` | 18 | Cache & help page, admin gating, help launcher on 6 pages, FAQ search/quick reply/fallback, first-time onboarding + empty states, cache badge + answer summary, dashboard KPIs, metadata view/search/edit, viewer read-only, chat filters, help center, page headers |
| `tests/unit/test_offline_and_config.py` | 5 | FAQ/metadata/cache with network disabled, `.env.example` documents new settings, setting bounds, no secrets/prompts in FAQ, custom FAQ entries |

Existing tests: all 277 unchanged and passing (the UI harness gained the new page and the help launcher, rendered
after the page so element indices are unchanged).

---

# Final Test Report — RAG Ultimate 2.0.0

Date: 2026-11-10 · Python 3.13.18 · PostgreSQL 16.15 · tesseract 5.3.4 · Streamlit 1.67 · Linux x86-64 (2 vCPU).

## Summary

| Gate | Command | Result |
|---|---|---|
| Unit + integration + security + performance tests | `TEST_DATABASE_URL=postgresql://… python -m pytest` | **268 passed, 0 failed, 0 skipped** (≈ 91 s) — baseline was 96 |
| Lint | `ruff check .` | clean |
| Static security analysis | `bandit -c pyproject.toml -r . -ll` | **0 high, 0 medium** (11 low) |
| Dependency vulnerabilities | `pip-audit -r requirements.lock` | **No known vulnerabilities** |
| Live stack smoke test (real processes) | `python -m scripts.smoke_test …` | **17 / 17 passed** — `docs/evaluation/smoke_test_results.json` |
| Backup → verify → restore (CLI, PostgreSQL) | `python -m scripts.backup create / verify / restore` | passed; 4 documents, 39 chunks, 10 audit events back after restore |
| Retrieval benchmark (real run) | `python -m evaluation.benchmark` | `docs/evaluation/BASELINE_VS_UPGRADED_lexical.md` |
| Compose profiles | `docker compose --profile … config` | local, hybrid, production, monitoring: valid |
| Container image build / Trivy scan | `docker build .` | **BLOCKED** — Docker Hub refused by the sandbox proxy |

Live stack = API (uvicorn) + UI (Streamlit) + PostgreSQL, `AUTH_MODE=users`, JSON logs. The only stand-in was the
LLM provider: the repository's scripted fake server played Groq and Ollama, because no real model endpoint is
reachable from the build sandbox. **Answer quality on real models was therefore not measured here.**

## Validation matrix (master prompt §Phase 12)

Status: ✅ tested successfully · 🟡 partially tested · ⛔ blocked by missing external credentials/infrastructure.

| # | Area | Status | Evidence |
|---|---|---|---|
| 1 | Application startup | ✅ | live: `/healthz`, `/readyz`, `/_stcore/health`; AppTest of `app.py` in both auth modes |
| 2 | Document upload | ✅ | UI uploader (AppTest), REST upload (smoke, `test_api_end_to_end_isolation`) |
| 3 | PDF ingestion | ✅ | `tests/test_ingestion.py`, smoke upload of the datasheet (text, tables, equations, figures) |
| 4 | OCR | ✅ | `test_image_ocr` (real tesseract), smoke PNG upload; scanned PDF pages via `_local_ocr_pdf` + vision OCR tests |
| 5 | Chunking | ✅ | 4 strategies × size bounds, cache-key test, Markdown structure, heading-only fallback |
| 6 | Embeddings | 🟡 | local fastembed path, OpenRouter and **Ollama** embeddings tested with deterministic fakes; the real nomic model could not be downloaded here |
| 7 | Vector indexing | ✅ | NumPy store, cache keys, stale-vector removal on new version / delete |
| 8 | BM25 retrieval | ✅ | real (no model needed): benchmark + `test_every_mode_returns_scored_hits[bm25]` |
| 9 | Hybrid retrieval | ✅ | RRF of dense + BM25 + fuzzy; multi-query variants vote; filters; parent-child |
| 10 | Reranking | 🟡 | code path tested with the offline fake cross-encoder and the OpenRouter reranker fake; real MiniLM not downloadable here |
| 11 | Local LLM responses | 🟡 | Ollama adapter (chat, tools, images, embeddings) against a protocol-faithful fake; **live Ollama ⛔** (registry blocked) |
| 12 | Cloud LLM responses | ⛔ / 🟡 | no credentials; OpenAI + Anthropic wire formats tested against strict fakes, Bedrock with botocore Stubber |
| 13 | Model fallback | ✅ | 429 with Retry-After, 5xx retries, circuit breaker, chain fallback, sensitive-data exclusion |
| 14 | Multi-agent execution | ✅ | `tests/integration/test_agent_workflow.py` (trace, summarisation, timeout, fallbacks, abstention) |
| 15 | Prompt injection | ✅ | 15 direct variants incl. zero-width and base64, indirect injection in a document, 9 benign false-positive checks |
| 16 | PII detection | ✅ | 11 entity types, checksums, policies, input redaction verified at the provider, output masking |
| 17 | Harmful-content controls | ✅ | 6 categories + supportive self-harm response + false-positive cases (one real FP fixed) |
| 18 | Topic restrictions | ✅ | restricted, allowed boundary, tenant overrides, UI policy probe |
| 19 | Grounding | ✅ | grounding guard + verifier + quality agent; unsupported answers abstained or corrected |
| 20 | Citation correctness | ✅ | verifier (invalid / mis-citations), benchmark `citation_correctness` metric |
| 21 | Tenant isolation | ✅ | service + REST API: cross-tenant listing, query, retrieval-debug and delete all denied and audited |
| 22 | Authentication | ✅ | PBKDF2, lockout, password policy, tokens (hashed, revocable), UI login (AppTest), API bearer auth |
| 23 | Database persistence | ✅ | SQLite **and PostgreSQL 16**: repository parity, migrations 0001→0002 on existing data, DB conversations |
| 24 | Conversation history | ✅ | JSON store (1.x tests) + DB store (user-scoped, variants on regenerate) |
| 25 | Error recovery | ✅ | ingestion retries/backoff, failed-job tracking, provider fallback, agent timeouts, fail-fast model loading, evidence-only answers |
| 26 | Performance | ✅ | `tests/performance`: retrieval p50 2.9 ms / p95 3.1 ms over 300 nodes; 16 concurrent API queries p95 0.27 s (fake model); first live question 0.32 s after the model-loading fix (was 119.5 s) |
| 27 | Backup and restoration | ✅ | automated round trips on SQLite + PostgreSQL, tamper detection, CLI cycle on the live stack |

## Defects found and fixed during the transformation

| Phase | Defect | Impact |
|---|---|---|
| 2 | Heading-only text produced no chunk (short OCR results, title slides lost) | data loss in index |
| 2 | Re-entrant lock deadlock in the background ingestion pool | uploads hang |
| 2 | `sanitize_filename` forced every file to `.pdf` | wrong format detection |
| 6 | Violence rule matched "kill a hung process" | false refusals |
| 6 | Superseded versions readable without accounts | stale retrieval |
| 7, 12 | Offline model loads retried a blocked download for ~40 s per call; first question took 119.5 s | unusable offline mode |
| 10 | Agent thread pool dropped request context; service minted a second request id | broken log/trace correlation |
| 11 | Literal bidi/zero-width characters in source; Pillow 12.3 advisories; Caddy exposed `/api/metrics` | security |
| 12 | `answer_relevance` compared vectors from two different embedding models | wrong metric with Ollama/OpenRouter embeddings |

## How to reproduce

```bash
pip install -r requirements-dev.txt
sudo apt-get install -y tesseract-ocr          # OCR tests
export TEST_DATABASE_URL=postgresql://rag:ragtest@127.0.0.1:5432/rag_test   # optional: PostgreSQL variants
python -m pytest                               # full suite
ruff check . && bandit -c pyproject.toml -r . -ll && pip-audit -r requirements.lock
python -m evaluation.benchmark                 # retrieval benchmark (add --generation with a configured model)
python -m scripts.smoke_test --api http://127.0.0.1:8000 --ui http://127.0.0.1:8501 --user admin --password '…'
```

## Not verified here (do this on your infrastructure before go-live)

1. `docker build` + Trivy scan + `docker compose --profile production up` (needs Docker Hub access) — CI does this.
2. Live Ollama with a pulled model; live OpenAI / Anthropic / Bedrock with real keys.
3. Real fastembed models (dense retrieval + cross-encoder): rerun `python -m evaluation.benchmark` and
   `--generation` once models are available; compare with the lexical-only numbers in `docs/evaluation/`.
4. Caddy TLS issuance on a public domain; Grafana dashboard against live Prometheus.
