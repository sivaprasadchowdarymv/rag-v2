# RAG∞ Ultimate: detailed installation, from zero to deployed

Text answers come from **Groq** (free). **Qwen / Gemma on OpenRouter** are used only for images: figure
questions and scanned pages. Every step ends with a check, so you know it worked before moving on.

**Time:** about 30 minutes. **Cost:** $0 with the default `free` profile.

---

## Step 0: Install the tools

| Tool | Version | Windows | macOS | Linux (Ubuntu/Debian) |
|---|---|---|---|---|
| Python | **3.12** (3.13 also works) | python.org installer: tick **"Add python.exe to PATH"** | `brew install python@3.12` | `sudo apt install python3.12 python3.12-venv` |
| Git | any | git-scm.com | `xcode-select --install` | `sudo apt install git` |

Check (new terminal):

```bash
python --version      # Windows; macOS/Linux: python3 --version   -> Python 3.12.x
git --version
```

> On macOS/Linux, type `python3` wherever this guide says `python` until the virtual environment is active.

---

## Step 1: Get the Groq key (text answers, required, free) — 3 min

1. Open **console.groq.com** and sign in (Google or GitHub works; no credit card).
2. **API Keys → Create API Key** → name it `rag-ultimate` → **Submit**.
3. Copy the key (`gsk_...`). It is shown only once.

Check: the key appears in the API Keys list.

Free-tier limits apply per model, per minute and per day (see console.groq.com → Settings → Limits). The app
stays inside them with small prompts, and falls back to another model when one is exhausted.

---

## Step 2: Get the OpenRouter key (images only, free) — 3 min

Without this key the app still works; figure questions are then answered from captions and nearby text, and
scanned pages stay unread.

1. Sign in at **openrouter.ai**.
2. **openrouter.ai/settings/keys → Create Key** → name `rag-ultimate`. Set a **credit limit** (e.g. $1): it is a
   hard spending cap, useful if you later switch to the `quality` profile.
3. Copy the key (`sk-or-v1-...`).
4. Free Gemma endpoints: if they later fail with "No endpoints found matching your data policy", open
   **openrouter.ai/settings/privacy** and allow the policy that free endpoints need.
5. Optional: buying $10 of credits raises the free-endpoint limit from about 50 to about 1,000 requests per day.

Check: the key is listed on the keys page.

---

## Step 3: Create your fresh repository — 5 min

1. Unzip `rag-ultimate.zip` and open a terminal **inside** the `rag-ultimate` folder
   (Windows: in File Explorer, type `cmd` in the address bar and press Enter).
2. On GitHub: **New repository** → name `rag-ultimate` → **Private** → do **not** add a README → **Create**.
3. Push the code:

   ```bash
   git init
   git add .
   git commit -m "RAG Ultimate: Groq for text, Qwen/Gemma for images"
   git branch -M main
   git remote add origin https://github.com/<your-user>/rag-ultimate.git
   git push -u origin main
   ```

Check: GitHub shows the files. `.env` and `data/` must **not** be there (`.gitignore` excludes them).

---

## Step 4: Create the virtual environment and install — 5 min

**Windows (Command Prompt):**

```bat
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

(PowerShell: `venv\Scripts\Activate.ps1`; if blocked, run once
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.)

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Your prompt now starts with `(venv)`. Every later command runs in this activated environment.

Check:

```bash
python -c "import streamlit, pymupdf, httpx, fastembed; print('ok')"
```

---

## Step 5: Configure the keys — 2 min

```bash
copy .env.example .env        # Windows
cp .env.example .env          # macOS / Linux
```

Open `.env` in any editor and set:

```ini
GROQ_API_KEY=gsk_...
OPENROUTER_API_KEY=sk-or-v1-...
MODEL_PROFILE=free
```

Everything else already has token-saving defaults. The settings you are most likely to change:

| Setting | Default | Change it when |
|---|---|---|
| `MODEL_PROFILE` | `free` | `quality` = paid Qwen 3.8 reads images first (better on dense graphs; cents) |
| `IMAGE_ANSWERS` | `true` | `false` = never send figure images; Groq answers from captions |
| `OCR_SCANNED` | `true` | `false` if your PDFs are never scanned |
| `MAX_CONTEXT_CHARS` | `5000` | raise to 8000 for long tables (more tokens) |
| `AGENT_MODE` | `agent` | `quick` = always exactly one call per question |
| `REASONING_EFFORT` | `low` | `medium` for hard calculations |
| `VERIFY_WITH_LLM` | `false` | `true` adds a Llama 70B check (one extra Groq call per answer) |
| `MASTER_MODELS` / `VISION_MODELS` | empty | pin your own chain, format `provider:model` |

---

## Step 6: Verify the setup against the live APIs — 1 min

```bash
python -m scripts.check_setup
```

It checks both keys, confirms every model in your chains exists on Groq / OpenRouter right now, then makes one
tiny call per role. Expected (abbreviated):

```text
Profile: free  (text -> Groq, images -> Qwen/Gemma on OpenRouter)
✓ Groq key works
✓ OpenRouter key works · used $0.0000 · limit $1.00
✓ master    groq:openai/gpt-oss-120b
✓ vision    openrouter:google/gemma-4-31b-it:free
...
✓ master    groq:openai/gpt-oss-120b -> tool call calculate({"expression": "7*1.5"}) · $0.00000
✓ vision    openrouter:google/gemma-4-31b-it:free -> "A black rectangle on a white background." · $0.00000
All good.
```

A ✗ line names the problem; see **Troubleshooting**. `python -m scripts.check_setup --no-calls` makes no model calls.

---

## Step 7: Run the tests (optional) — 1 min

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
```

Check: `96 passed`. The tests use a fake Groq + OpenRouter server: no keys, no cost.

---

## Step 8: Start the app and try it — 5 min

```bash
streamlit run app.py
```

Your browser opens http://localhost:8501. On the first upload the app downloads two small local models once
(embeddings ~130 MB, reranker ~80 MB); later uploads are fast.

1. **Sidebar → Status** shows 🟢 `Groq: text answers` and 🟢 `OpenRouter: images (figures, scanned pages)`.
2. **Upload a PDF.** Scanned pages, if any, are read by Gemma once ("Reading scanned page ...").
3. **Ask a text question**, e.g. a suggested one. Under the answer: `Groq: gpt-oss-120b · 1 LLM call · ~900 tokens`.
4. **Ask a figure question**, e.g. *"What does the output voltage vs input voltage graph show?"*. Under the answer:
   `OpenRouter: gemma-4-31b-it:free`, and **Agent activity** shows "Figure question: 1 image(s) sent to the vision
   model". Values read from a picture are marked ≈ Partly verified: compare them with the figure in **Evidence**.
5. **Ask a calculation**, e.g. *"What is the power dissipation at 12 V input and 1 A?"*: Groq uses the calculator
   tool (2 calls).
6. **Settings → AI models**: see each role's model chain, pick a different Groq model, switch profile.
7. **Validate**: generate questions → **Check retrieval** (free) → **Full test**.

Stop the app with `Ctrl+C` in the terminal.

---

## Step 9: Keep data across restarts (needed on Streamlit Cloud) — 5 min

1. huggingface.co → **Settings → Access Tokens → Create new token** → type **Write**.
2. Pick a dataset name: `<your-hf-username>/rag-ultimate-data` (created private automatically).
3. Add to `.env` (local) or the Streamlit secrets (cloud):

   ```ini
   HF_TOKEN=hf_...
   HF_DATASET_REPO=<your-hf-username>/rag-ultimate-data
   ```

Check: sidebar shows `Storage: saved to Hugging Face`.

---

## Step 10: Deploy

### Option A: Streamlit Community Cloud (free)

1. share.streamlit.io → **Create app** → repo `rag-ultimate`, branch `main`, file `app.py`.
2. **Advanced settings → Python 3.12.**
3. **Secrets** (from `.streamlit/secrets.toml.example`):

   ```toml
   GROQ_API_KEY = "gsk_..."
   OPENROUTER_API_KEY = "sk-or-v1-..."
   MODEL_PROFILE = "free"
   HF_TOKEN = "hf_..."
   HF_DATASET_REPO = "your-hf-username/rag-ultimate-data"
   APP_PASSWORD = "choose-one"
   ```
4. **Deploy.** Then bookmark your workspace link (**Settings → Workspace & storage**).

### Option B: Docker on your own server

```bash
cp .env.example .env            # fill in the keys
docker compose up -d --build    # app on 127.0.0.1:8501, data in the app_data volume
```

Put a reverse proxy (Caddy / nginx) in front for HTTPS.

---

## Step 11: Updating models later

* Groq or OpenRouter retire a model → `python -m scripts.check_setup` shows ✗ for it.
* Fix without code: `MASTER_MODELS=groq:<new-model>,groq:llama-3.3-70b-versatile` or
  `VISION_MODELS=openrouter:<new-model>,openrouter:google/gemma-4-31b-it:free`.
* Fix for everyone: edit the constants at the top of `llm/catalog.py`.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `'python' is not recognized` (Windows) | Python not on PATH | Reinstall with "Add python.exe to PATH", or use `py -3.12` |
| `No module named ...` | venv not active | Activate it (Step 4), then `pip install -r requirements.txt` |
| `Groq rejected the API key` | Wrong key | New key at console.groq.com/keys |
| `Groq's free limit was reached ...` | Per-minute / per-day quota | Wait a minute; the chain already falls back to Llama 70B and free Gemma; use Quick mode |
| `The question plus evidence is too long` | Prompt above Groq's per-request limit | Lower `MAX_CONTEXT_CHARS` |
| Figure question answered by Groq | No `OPENROUTER_API_KEY`, `IMAGE_ANSWERS=false`, or no figure matched | Add the key; name the figure ("Figure 3", "the graph of ...") |
| "figure images could not be read this time" | All vision models busy / rate limited | Retry later, or `MODEL_PROFILE=quality` |
| Free Gemma: `No endpoints found matching your data policy` | OpenRouter privacy settings | openrouter.ai/settings/privacy |
| `N image-only page(s) could not be read` | OCR off or no OpenRouter key | `OCR_SCANNED=true` + key, then re-upload |
| First upload slow | One-time local model download | Wait; later uploads use the cache |
| ⚠ Check sources often | Hard document | Thorough mode, `REASONING_EFFORT=medium`, check the Validate page |

---

## What is where

| Path | Purpose |
|---|---|
| `llm/catalog.py` | Which model does which job (`provider:model` chains, profiles) |
| `llm/providers.py` | Groq and OpenRouter clients (httpx), error handling |
| `llm/model_router.py` | Role → chain, fallback across providers |
| `agents/rag_pro.py` | The pipeline; the text-vs-image routing rule |
| `agents/master.py` | Answer writer (ReAct tools; images attached for figure questions) |
| `agents/verifier.py` | Claim-level checks; image-read values capped at "partly verified" |
| `rag/vision.py` | OCR of scanned pages (and optional figure descriptions) |
| `scripts/check_setup.py` | Live check of keys, models and roles |
| `.env.example` | Every setting, documented |
