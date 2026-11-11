# Production Readiness Checklist

Legend: ✅ implemented and tested here · 🟡 implemented, partially tested · ⛔ blocked here (needs your
infrastructure/credentials) · ☐ operator action before go-live.

## Definition of done (master prompt §5)

| Item | Status | Evidence / note |
|---|---|---|
| Existing core functionality preserved | ✅ | original 96 tests pass; 1.x link workspaces still work |
| Application starts successfully | ✅ | live UI + API processes, AppTest |
| Document ingestion works | ✅ | 10 formats + OCR, live uploads |
| Hybrid retrieval works | ✅ | real lexical benchmark; dense via deterministic fake embedder |
| Reranking works | 🟡 | code path tested with fake cross-encoder; real model download blocked |
| Local Ollama integration works | 🟡 / ⛔ | protocol-faithful fake; live Ollama blocked here |
| Cloud integration works when configured | 🟡 / ⛔ | OpenAI/Anthropic fakes + Bedrock Stubber; no credentials |
| Model routing works | ✅ | modes, sensitivity, context, strategies, fallback, circuit breaker |
| Multi-agent orchestration works | ✅ | workflow tests |
| Prompt injection tests pass | ✅ | `tests/security` |
| PII protection tests pass | ✅ | `tests/security` |
| Harmful-content tests pass | ✅ | `tests/security` |
| Topic restrictions work | ✅ | `tests/security`, UI probe |
| Grounding and citation validation work | ✅ | verifier, grounding guard, quality agent |
| User authentication works | ✅ | UI login + API tokens |
| Tenant isolation is tested | ✅ | service + REST API |
| Database persistence works | ✅ | SQLite + PostgreSQL 16 |
| Evaluation dashboard works | ✅ | Evaluation page (AppTest) + stored runs |
| Logging and monitoring work | ✅ | JSON logs, request ids, /metrics, OTel export |
| Docker deployment works | ⛔ | Dockerfile + compose validated (`compose config`); image build blocked (no registry) |
| CI/CD pipeline works | 🟡 | workflows written and their commands run locally; not executed on GitHub |
| Documentation is complete | ✅ | README, docs/*.md |
| Backup and recovery are tested | ✅ | automated + CLI round trips |

## Before go-live (operator)

- ☐ Push to GitHub and confirm the CI workflow is green (image build, smoke test, Trivy).
- ☐ `docker compose --profile production up -d` on the target host; run `scripts.smoke_test` against it.
- ☐ Strong secrets in `.env` (POSTGRES_PASSWORD, MINIO_*, ADMIN_PASSWORD, GRAFANA_ADMIN_PASSWORD); never commit it.
- ☐ DNS for `DOMAIN`, ports 80/443 open; verify HTTPS certificate issuance.
- ☐ Pull Ollama models; pre-seed or allow download of fastembed models; rerun `python -m evaluation.benchmark
  --generation` with your real models and keep the report as the quality baseline.
- ☐ Decide the data policy: `ALLOW_SENSITIVE_CLOUD`, `CLOUD_PROVIDERS_ALLOWED`, PII policies, topics.
- ☐ Schedule `docker compose --profile production run --rm backup` daily; copy backups off-host; test a restore.
- ☐ Enable `--profile monitoring`; set alerting on 5xx rate, error counters, failed ingestion jobs, cost.
- ☐ Review `docs/SECURITY_AUDIT.md` residual risks; consider an external penetration test.


## 2.1 additions

| Item | Status | Evidence / note |
|---|---|---|
| All 2.0 features unchanged | ✅ | 277 original tests pass; live smoke 17/17 |
| Repeated questions served from a safe cache | ✅ | `tests/unit/test_answer_cache_v2.py`, live API check |
| Cache tenant/user isolation and invalidation | ✅ | identical-file two-tenant test, new version/delete tests |
| Help assistant works offline, exposes no secrets | ✅ | `tests/unit/test_offline_and_config.py` |
| Metadata searchable, editable, used by filters | ✅ | `tests/unit/test_metadata_intel.py`, UI tests |
| UI renders on desktop and phone | ✅ | AppTest + headless Chrome screenshots |
| Multi-replica deployments | ☐ | the answer cache is per process; each replica warms its own (by design, no Redis) |
| Review `ANSWER_CACHE_SCOPE` | ☐ | `user` (default, strictest) or `tenant` (share across users with the same roles) |
| Tune `ANSWER_CACHE_TTL` to how often documents change | ☐ | default 3600 s; new versions invalidate immediately anyway |
| Add organisation-specific help questions | ☐ | ⚡ Cache & help → Help questions |
| Container image build | ⛔ here | build on your host or CI (Dockerfile unchanged; no new dependencies) |
