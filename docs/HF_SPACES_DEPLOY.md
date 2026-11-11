# Deploy to Hugging Face Spaces (Docker) — beginner guide

This puts the full app (all 2.2 features: chat, library, OCR, metadata, answer cache, FAQ, dashboards,
Groq + OpenRouter fallback chains, backup keys) online at `https://huggingface.co/spaces/<you>/<space>`.

How it works, in one picture:

```
your PC: project folder + .env
   │  py launch.py spaces --push <you>/rag-ultimate
   ▼
Hugging Face Space (Docker)            ──►  Groq / OpenRouter (answers)
   builds Dockerfile → runs Streamlit  ──►  Neon PostgreSQL (accounts, chats)      [optional]
   on port 7860 as user 1000           ──►  HF dataset me/rag-data (document indexes) [optional]
```

* The Space builds **`Dockerfile.spaces`** (copied in as `Dockerfile`). Your normal `Dockerfile` and
  `docker-compose.yml` are unchanged.
* Your `.env` is **never uploaded**. Its values are sent to the Space's *Settings → Variables and
  secrets* (keys, passwords, tokens and `DATABASE_URL` as hidden **secrets**; the rest as **variables**).
* A Space's disk is wiped on every restart. Keep data with Neon (`DATABASE_URL`) and an HF dataset
  (`HF_TOKEN` + `HF_DATASET_REPO`), or attach a Storage Bucket (mounted at `/data`, used automatically).

---

## Before you start (5 minutes)

You need what you already used for Streamlit:

1. The project folder with a working `.env` (made by `py launch.py setup`). Check it:
   ```powershell
   cd D:\agents\RAG\rag-production-v1-2.2.1\rag-production-v1-main
   py launch.py check
   ```
2. A Hugging Face account → <https://huggingface.co/join>.
3. A Hugging Face **WRITE** token: <https://huggingface.co/settings/tokens> → *Create new token* →
   type **Write** → name it `rag-deploy` → *Create* → copy it (`hf_…`).
   *If the `HF_TOKEN` already in your `.env` is a Write token, the launcher uses it and you can skip this.*

Strongly recommended in `.env` (open it with Notepad: `notepad .env`):

| Line | Why |
|---|---|
| `AUTH_MODE=users` + `ADMIN_USERNAME=…` + `ADMIN_PASSWORD=…` | login screen; without it anyone with the link uses your API keys |
| `DATABASE_URL=postgresql://…neon.tech/neondb?sslmode=require` | accounts and chats survive restarts |
| `HF_TOKEN=hf_…` + `HF_DATASET_REPO=<you>/rag-data` | uploaded documents survive restarts |

---

## Step 1 — Put the new files in your project

Unzip `rag-production-v1-2.2.2.zip` and copy these into your project folder (replace when asked):

```
launch.py
Dockerfile.spaces
deploy\hf-space\start.sh
deploy\hf-space\db_probe.py
docs\HF_SPACES_DEPLOY.md
tests\unit\test_hf_space.py
.gitignore
CHANGELOG.md
```

## Step 2 — Build and upload the Space (one command)

Replace `MVSPC16` with **your** Hugging Face username:

```powershell
py launch.py spaces --push MVSPC16/rag-ultimate
```

What you will see:

```
[ OK ] Space folder ready: build\hf-space  (170 files; no .env, no data)
[ OK ] Signed in to Hugging Face as MVSPC16
[ OK ] Space MVSPC16/rag-ultimate ready (private)
[ OK ] Settings sent: 9 secrets (hidden: ADMIN_PASSWORD, DATABASE_URL, GROQ_API_KEY, …) and 14 variables
==> Uploading files (a few minutes the first time)...
[ OK ] Done. Build log + app: https://huggingface.co/spaces/MVSPC16/rag-ultimate
```

* If it asks for a token, paste the **Write** token (nothing shows while you paste — that is normal) and press Enter.
* `[WARN]` lines tell you what is missing (login, database, persistence). The app still works.
* New Spaces are **private** (only you, when signed in). Add `--public` the *first* time to make it public,
  or change it later in *Settings → Change Space visibility*. Use `AUTH_MODE=users` before going public.

## Step 3 — Watch it build (first time ≈ 5–10 minutes)

1. Open the link printed at the end.
2. The status badge goes **Building → Running**. Click **Logs** (top right) to watch:
   * *Build* tab: installing packages (the long part).
   * *Container* tab: you should see
     ```
     RAG Ultimate on Hugging Face Spaces: DATA_DIR=... LLM_MODE=cloud
     applied: [...]          (database tables ready)
     ```
3. When it says **Running**, the app appears on the Space page. Log in with `ADMIN_USERNAME` / `ADMIN_PASSWORD`.

## Step 4 — Check everything works

1. **Library** → upload a small PDF → it appears with its metadata.
2. **Chat** → ask a question about it → answer with citations.
3. Ask the same question again → the answer shows **cached**.
4. Admin → **Models** / **Model routing** → Groq and OpenRouter show *ready*, with the backup-key count.
5. Restart test (only if you set Neon + HF dataset): *Settings → Factory rebuild*, log in again → the
   account, chats and documents are still there.

## Step 5 — Change settings or update the code later

* **Change a key or setting:** *Space → Settings → Variables and secrets* → edit → the Space restarts.
  Or edit `.env` and run the Step 2 command again (it overwrites the settings).
* **New version of the code:** copy the new files into your project and run the Step 2 command again.
  Only changed files are uploaded.
* **Files only, keep the Space's settings as they are:** add `--no-secrets`.

---

## Alternative: do it by hand in the browser (no launcher)

1. `py launch.py spaces` (without `--push`) → creates the folder `build\hf-space`.
2. <https://huggingface.co/new-space> → name `rag-ultimate` → **SDK: Docker → Blank** → Private → *Create Space*.
3. In the Space: **Files → Add file → Upload files** → drag *everything inside* `build\hf-space`
   (including the `.streamlit` and `deploy` folders) → *Commit*. Check `README.md` and `Dockerfile` are at the top level.
4. **Settings → Variables and secrets** → add each line of your `.env` that has a value:
   *New secret* for names ending in `KEY`, `KEY_2`, `KEY_3`, `TOKEN`, `PASSWORD`, `SECRET`, and `DATABASE_URL`;
   *New variable* for the rest (`AUTH_MODE`, `ADMIN_USERNAME`, `MASTER_MODELS`, `VISION_MODELS`, …).
   Skip `DATA_DIR`, `OLLAMA_*`, `POSTGRES_PASSWORD`. Set `LLM_MODE` = `cloud`.

*(Pushing with `git` also works, but Hugging Face only accepts the image files in `assets/` through
Git Xet/LFS, so the launcher or the browser upload is easier.)*

---

## Troubleshooting

| You see | Fix |
|---|---|
| `Hugging Face rejected the token` | Use a **Write** token: `py launch.py spaces --push you/rag-ultimate --token hf_…` |
| Build fails at `pip wheel` | Open *Logs → Build*, copy the last 20 lines. Usually a temporary network error: *Settings → Factory rebuild*. |
| Container log: `WARNING: DATABASE_URL is not reachable from this Space -> using SQLite` | Hugging Face can block outgoing database ports. The app runs anyway, but accounts and chats are stored in the Space (lost on restart). Check `DATABASE_URL` is the Neon one ending in `?sslmode=require`; if it still fails, attach a Storage Bucket (*Settings → Storage*) so SQLite in `/data` survives restarts, or remove `DATABASE_URL`. |
| Upload of a document fails with `403` | The Space must run `deploy/hf-space/start.sh` (it turns off Streamlit's XSRF cookie, which browsers block inside the Space's iframe). Make sure the top-level `Dockerfile` is the one from `Dockerfile.spaces`, then hard-refresh (Ctrl+F5). |
| No answers / "no model available" | *Settings → Variables and secrets*: `GROQ_API_KEY` must be a **secret** and `LLM_MODE` = `cloud`. Admin → *Model routing* shows which keys are ready. |
| Documents gone after a restart | Set `HF_TOKEN` (Write) + `HF_DATASET_REPO`, or attach a Storage Bucket. |
| Space says **Sleeping** | Free Spaces sleep after a period without visitors. Open it; it wakes in about a minute. |
| Login screen missing | `AUTH_MODE` = `users` (variable) and `ADMIN_PASSWORD` (secret) — 10+ characters with letters and digits. |

## Security notes

* Secrets are encrypted by Hugging Face and are not shown in the Space files or logs; the launcher prints only their names.
* XSRF protection is off **only in the Space image** because Spaces run the app inside an iframe on another domain.
  Logins (`AUTH_MODE=users`), upload type/size checks, rate limits and guardrails stay on.
* Keep the Space **private** until `AUTH_MODE=users` is set. The REST API (`api/`) is not exposed on Spaces
  (one public port); run it with Docker Compose if you need it.
