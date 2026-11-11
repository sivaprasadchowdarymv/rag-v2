# Deployment Guide

Python 3.12 or 3.13. Tesseract for local OCR (`apt install tesseract-ocr` / `brew install tesseract` /
Windows installer from UB-Mannheim). Docker 24+ with Compose v2 for container profiles.

## Install (bare metal / VM)

```bash
git clone <your repo> rag-ultimate && cd rag-ultimate
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.lock                       # exact, audited versions
pip install -r requirements-optional.txt               # only if you use Bedrock, S3/MinIO or OpenTelemetry
cp .env.example .env                                   # then edit .env
streamlit run app.py                                   # UI  → http://localhost:8501
uvicorn api.main:app --host 127.0.0.1 --port 8000      # API → http://localhost:8000/docs
```

Database migrations run automatically on start; to run them explicitly: `python -m database.migrations`
(`--status` lists applied versions).

## PostgreSQL without Docker (Windows / macOS / Linux)

1. Install PostgreSQL 16 (Windows: EDB installer from postgresql.org; macOS: `brew install postgresql@16`;
   Linux: `apt install postgresql`). Remember the password you give the `postgres` superuser.
2. Create a database and user (Windows: *SQL Shell (psql)* from the Start menu; others: `psql -U postgres`):
   ```sql
   CREATE USER rag WITH PASSWORD 'choose-a-strong-password';
   CREATE DATABASE rag OWNER rag;
   ```
3. In `.env`:
   ```ini
   DATABASE_URL=postgresql://rag:choose-a-strong-password@localhost:5432/rag
   AUTH_MODE=users
   ADMIN_USERNAME=admin
   ADMIN_PASSWORD=<10+ characters, letters and digits>
   ```
4. `python -m database.migrations` (creates the tables; also happens automatically at start-up), then
   `streamlit run app.py` and, optionally, `uvicorn api.main:app --port 8000`.

## PROFILE D — cloud models + PostgreSQL (no Ollama) — 2.2.1

For Groq / OpenRouter (/ AnyRouter) users who want PostgreSQL and the REST API in containers without the large
Ollama image.

```ini
# .env
POSTGRES_PASSWORD=<strong>
AUTH_MODE=users
ADMIN_USERNAME=admin
ADMIN_PASSWORD=<10+ characters, letters and digits>
GROQ_API_KEY=...
OPENROUTER_API_KEY=...
```

```bash
docker compose --profile cloud up -d --build    # postgres, app-cloud (127.0.0.1:8501), api (127.0.0.1:8000)
docker compose run --rm backup                  # backup to ./backups
```

## PROFILE A — local offline

No external LLM APIs; everything stays on the machine.

```bash
# .env
LLM_MODE=local
OLLAMA_BASE_URL=http://localhost:11434      # docker compose: http://ollama:11434 (set automatically)
OLLAMA_MODEL=llama3.1:8b
EMBED_BACKEND=local                          # fastembed (downloads ~130 MB once) — or ollama + OLLAMA_EMBED_MODEL
```

```bash
ollama pull llama3.1:8b                      # once, while online (or copy ~/.ollama from another machine)
streamlit run app.py
# or with containers:
docker compose --profile local up -d && docker compose exec ollama ollama pull llama3.1:8b
```

Fully offline after the first download: fastembed models live in `data/models`, Ollama models in its volume. If a
model cannot be loaded, the app fails fast (MODEL_RETRY_S) and keeps working with keyword search.

## PROFILE B — hybrid

Local Ollama plus optional cloud providers, PostgreSQL, optional S3-compatible storage.

```bash
# .env
LLM_MODE=hybrid
GROQ_API_KEY=...            # and/or OPENAI_API_KEY / ANTHROPIC_API_KEY / BEDROCK_REGION
ALLOW_SENSITIVE_CLOUD=false # confidential/PII documents still answered locally only
POSTGRES_PASSWORD=<strong>
AUTH_MODE=users
ADMIN_USERNAME=admin
ADMIN_PASSWORD=<≥10 chars, letters+digits>
```

```bash
docker compose --profile hybrid up -d        # ollama, postgres, app-hybrid (8501), api (8000) on 127.0.0.1
```

## PROFILE C — production server

Containerised app + API, PostgreSQL, MinIO object storage, Caddy reverse proxy with automatic HTTPS,
authentication, monitoring and backups.

```bash
# .env (in addition to B)
DOMAIN=rag.example.com                        # DNS A record → this server; ports 80/443 open
MINIO_ROOT_USER=<user>
MINIO_ROOT_PASSWORD=<≥8 chars>
GRAFANA_ADMIN_PASSWORD=<strong>
LOG_FORMAT=json
```

```bash
docker compose --profile production up -d                  # app, api, postgres, minio, caddy
docker compose --profile production --profile monitoring up -d   # + prometheus (9090), grafana (3000) on localhost
```

* UI: `https://$DOMAIN/`, API: `https://$DOMAIN/api/…` (`/api/metrics` is blocked publicly; Prometheus scrapes
  the api container directly).
* Create more users in **Administration → Users & roles** or `python -m scripts.create_admin --username …`.
* Kubernetes is not required. Introduce it only when you need several app replicas; the app is stateless except
  for `DATA_DIR` (indexes) — put that on shared storage or rebuild indexes from object storage per replica.

## Backups

```bash
python -m scripts.backup create --dest backups               # bare metal
docker compose --profile production run --rm backup          # containers → ./backups/rag-backup-<time>/
python -m scripts.backup verify backups/rag-backup-<time>
python -m scripts.backup restore backups/rag-backup-<time>   # asks for confirmation; --yes to skip
```

Schedule `create` daily (cron / systemd timer) and copy `backups/` off the server. Contents: database
(SQLite backup API, or `pg_dump` custom format, or portable JSON), `data/` (indexes, figures, originals,
workspaces, evaluations, telemetry), SHA-256 manifest. MinIO data is backed up separately (`mc mirror`).
Restores were tested end to end on SQLite and PostgreSQL 16 (`tests/integration/test_database_storage.py`).

## Health and monitoring

| Endpoint | Purpose |
|---|---|
| `GET /healthz` (API) | liveness |
| `GET /readyz` (API) | database + provider reachability (503 if the DB is down) |
| `GET /metrics` (API) | Prometheus metrics (METRICS_ENABLED) |
| `GET /_stcore/health` (UI) | Streamlit liveness (Docker HEALTHCHECK) |

Grafana dashboard: `deployment/grafana/dashboards/rag-ultimate.json` (provisioned automatically).
Traces: set `OTEL_EXPORTER_OTLP_ENDPOINT` and install `requirements-optional.txt`.

## CI/CD

`.github/workflows/ci.yml`: ruff, bandit (fails on medium/high), pytest with a PostgreSQL service and tesseract,
retrieval benchmark artifact, pip-audit on the lock file, Docker build + smoke test + Trivy scan (fails on
fixable HIGH/CRITICAL). `.github/workflows/release.yml`: tag `v*` → image pushed to GHCR.

## Upgrading from 1.x

Nothing to migrate by hand: 1.x link workspaces keep working with `AUTH_MODE=none`; indexes are reused
(the cache key only changes for non-default chunk strategies). New tables are created on first start.
