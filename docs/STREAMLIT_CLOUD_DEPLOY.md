# Deploy to Streamlit Community Cloud (all features), then Docker

`launch.py` does the local work; Streamlit Community Cloud runs `app.py` from your GitHub repository.

| Feature | Streamlit Cloud | Docker (`--profile cloud`) |
|---|---|---|
| Chat, documents, metadata, cache, ❔ Help, dashboards, admin pages | ✅ | ✅ |
| Groq + backup keys, OpenRouter vision chain + backup keys | ✅ (secrets) | ✅ (.env) |
| Accounts, roles, audit log | ✅ (`AUTH_MODE=users`) | ✅ |
| PostgreSQL | ✅ hosted (Neon free tier) | ✅ container |
| Keep uploaded documents across restarts | ✅ private Hugging Face dataset | ✅ Docker volume |
| OCR (tesseract) | ✅ via `packages.txt` | ✅ in the image |
| REST API (`/v1/...`) | ❌ Streamlit Cloud runs one Streamlit app only | ✅ port 8000 |
| Local AI (Ollama) | ❌ | ✅ `--profile hybrid` / `local` |

## A. Prepare locally
```bash
python launch.py setup          # answer the questions (accounts: y)
python launch.py check --online # keys, database, OCR, providers
python launch.py run --api      # try it: http://localhost:8501 and http://localhost:8000/docs
```

## B. Free online services for Streamlit Cloud
1. **Neon** (neon.tech): create a project → copy the connection string (`postgresql://…?sslmode=require`).
2. **Hugging Face** (huggingface.co → Settings → Access Tokens): create a **write** token; choose a dataset name
   such as `your-name/rag-ultimate-data` (created private automatically).
3. Put both into `.env` (`DATABASE_URL=…`, `HF_TOKEN=…`, `HF_DATASET_REPO=…`), then
   `python launch.py check` (database must connect) and `python launch.py secrets`.

## C. GitHub
Push the project folder to a repository (private recommended). `.env`, `.streamlit/secrets.toml` and `data/` are
git-ignored — check with `git status` that they are not listed before committing.

## D. Streamlit Community Cloud
share.streamlit.io → **Create app** → your repository, branch `main`, main file `app.py` → **Advanced settings**:
Python **3.12** → **Secrets**: paste the contents of `.streamlit/secrets.toml` → **Deploy**. Sign in with your admin
account, open 🎓 Guide & Test Lab → Start here.

## E. Docker (same project, your own machine or server)
```bash
python launch.py docker up              # PostgreSQL + app + API (profile cloud)
python launch.py docker status | logs | backup | down
python launch.py docker up --profile production   # + MinIO + Caddy HTTPS (needs DOMAIN, MINIO_* in .env)
```
