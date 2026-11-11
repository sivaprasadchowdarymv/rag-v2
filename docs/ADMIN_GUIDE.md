# Administrator Guide

## Accounts, roles, tenants
Enable accounts with `AUTH_MODE=users`. The first admin is created at start-up from `ADMIN_USERNAME` /
`ADMIN_PASSWORD` (or `python -m scripts.create_admin --username … --tenant …`). Each user belongs to one
**tenant**; nothing crosses tenants (documents, conversations, audit, policies, model chains).

| Role | Can |
|---|---|
| admin | everything in the tenant: users, policies, model chains, all documents, audit log, evaluation |
| editor | upload, relabel and delete own documents, ask, run evaluations |
| viewer | ask questions about documents its roles may read |

*Users & roles*: create users, change roles, disable accounts (disabling revokes API tokens). Passwords need
≥ 10 characters with letters and digits; 5 failed logins lock an account for 15 minutes.

## Document permissions
*Documents → Access control & versions*: set **sensitivity** and **readable-by roles** (empty = whole tenant),
view version history, delete a document (all versions, vectors and stored originals). Changes are audited and
apply immediately to retrieval (UI and API).

## Guardrails — *Security*
| Setting | Options | Default |
|---|---|---|
| Prompt injection | block · warn | block |
| Harmful content | block · warn (self-harm always gets a supportive response) | block |
| PII in questions | block · redact · mask · allow | redact |
| PII in answers | authorized (admins / "pii" role see values) · mask · redact · block · allow | authorized |
| Allowed topics | one per line, synonyms with `|`; empty = no boundary | — |
| Restricted topics / blocked terms | one per line | — |
| Abstain when ungrounded | on / off | on |
| Usage limits | questions/min per user and per server, daily AI budget (USD) | 8 / 25 / off |

Use **Test a prompt** to see how a text would be treated. Detectors are heuristic — review
`docs/SECURITY_AUDIT.md` (residual risks) before relying on them.

## Models — *Models*
Providers are enabled by environment variables (`OLLAMA_BASE_URL`, `GROQ_API_KEY`, `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `BEDROCK_REGION`, `OPENROUTER_API_KEY`). The page shows reachability, the routing chain per
role with the reason each model is excluded, usage (calls, failures, tokens, cost, latency) and circuit-breaker
state. **Tenant model chains** override the order per role (`provider:model`, one per line).

Data-sensitivity rule: confidential/restricted/PII documents are answered by local models only. To allow a cloud
provider for them set `ALLOW_SENSITIVE_CLOUD=true` and `CLOUD_PROVIDERS_ALLOWED=anthropic` (for example) — this is
a policy decision for your organisation.

`ROUTING_STRATEGY`: `priority` (configured order), `cost`, `latency`, `auto` (cheapest/fastest for simple
lookups, priority for calculations/comparisons).

## Monitoring
* *Dashboard*: documents, questions, latency, spend, blocked requests, provider health, usage by model.
* *Ingestion*: job status, attempts, errors of failed documents.
* *Audit log*: logins, uploads, deletions, permission/policy changes, blocked or filtered requests, denied
  retrieval; filter and export CSV.
* Prometheus `/metrics` (API) and the Grafana dashboard (`--profile monitoring`); JSON logs with request ids
  (`LOG_FORMAT=json`); OpenTelemetry via `OTEL_EXPORTER_OTLP_ENDPOINT`.

## Operations
* Backups: `python -m scripts.backup create|verify|restore` — see `docs/DEPLOYMENT_GUIDE.md`.
* Migrations: automatic on start; `python -m database.migrations --status`.
* Post-deployment check: `python -m scripts.smoke_test --api … --ui … --user … --password …`.
* Evaluation: *Evaluation* page or `python -m evaluation.benchmark [--generation]`; runs are stored per tenant.
* Offline installs: pre-seed `data/models` (fastembed) and Ollama models; set `MODEL_DOWNLOAD=false` to never
  attempt downloads.
