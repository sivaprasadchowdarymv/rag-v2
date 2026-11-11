"""
Built-in help assistant (2.1): a compact FAQ chatbot available on every page.

* Deterministic: answers come from the maintained FAQ below (+ admin-added entries), the page help panels and
  docs/USER_GUIDE.md. No AI model is called, so it costs nothing, works offline and cannot leak documents,
  prompts, keys or admin data (it never reads any of them).
* Context-aware: questions about the page you are on are suggested first.
* Search: fuzzy keyword match (RapidFuzz, already a dependency) over questions, keywords and answers.
* Actions: each answer can open the relevant page.

    launcher(page_key)           floating "❔ Help" button with the chat panel (top right)
    panel(page_key, prefix)      the same panel inline (Guide → Help center)
    search(text, page_key)       ranked FAQ entries (also used by tests)
"""
from __future__ import annotations

import re
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

CATEGORIES = ["Getting started", "Uploading documents", "Supported file formats", "Asking questions",
              "Understanding citations", "How RAG works", "Agents and retrieval", "Cache and repeated questions",
              "Security and privacy", "Models and API configuration", "Metadata and document filters",
              "Troubleshooting"]

FAQ: List[Dict[str, object]] = [
    # --- Getting started
    {"id": "gs-what", "cat": "Getting started", "page": "guide", "kw": "what is this app purpose overview",
     "q": "What does this app do?",
     "a": "It answers questions **only from your own documents**. You upload files, ask in plain English, and "
          "every answer cites the exact page, table or equation it came from. Each answer is checked against "
          "those sources before you see it."},
    {"id": "gs-first", "cat": "Getting started", "page": "guide", "kw": "first time start begin new user steps",
     "q": "I'm new. Where do I start?",
     "a": "1. Open **🎓 Guide & Test Lab** → *Start here* and fix anything with a red dot.\n"
          "2. In the *Test Lab* tab press **Load sample documents**.\n"
          "3. Go to **💬 Chat** and ask *What is the dropout voltage?*\n"
          "4. Open **Evidence** under the answer to see where it came from."},
    {"id": "gs-samples", "cat": "Getting started", "page": "guide", "kw": "sample demo example documents try",
     "q": "Can I try it without my own files?",
     "a": "Yes. **Guide & Test Lab → Test Lab → Load sample documents** adds four small files (a datasheet, a "
          "pump manual, a stock list and an install guide) with known answers. The Chat page also offers this "
          "button while your library is empty."},
    {"id": "gs-pages", "cat": "Getting started", "page": "guide", "kw": "pages navigation sidebar menu",
     "q": "What are the pages in the sidebar for?",
     "a": "**Workspace**: Chat, Documents, Validate — daily use. **Operations**: Dashboard, Ingestion, Retrieval "
          "debugger, Agent traces, Evaluation, Analytics — to monitor and troubleshoot. **Administration**: "
          "Settings, plus Models, Security, Users, Audit log and Cache for admins. Every page has a "
          "*What is this page?* panel at the top."},
    {"id": "gs-workspace", "cat": "Getting started", "page": "settings", "kw": "workspace link bookmark save return",
     "q": "How do I get back to my chats and documents later?",
     "a": "With accounts (sign-in) everything is tied to your account. Without accounts, your private workspace "
          "is in the link (`?ws=…`): bookmark it — **Settings → Workspace & storage** shows it. Keep that link "
          "private: anyone with it can open the workspace."},
    # --- Uploading
    {"id": "up-how", "cat": "Uploading documents", "page": "library", "kw": "upload add file drag drop",
     "q": "How do I upload a document?",
     "a": "Open **📚 Documents** and drop files on the uploader (or use *Browse files*). Pick a **Sensitivity** "
          "first if the file is confidential. A progress bar shows reading, indexing and metadata extraction; "
          "a green message confirms when the file is ready."},
    {"id": "up-version", "cat": "Uploading documents", "page": "library", "kw": "new version update replace same name",
     "q": "What happens if I upload a file with the same name again?",
     "a": "If the content changed it becomes **version 2** (3, …): only the newest version is searched and cached "
          "answers based on the old version are discarded. Identical content is recognised and not stored twice."},
    {"id": "up-sensitivity", "cat": "Uploading documents", "page": "library",
     "kw": "sensitivity confidential restricted public internal label",
     "q": "What do the sensitivity labels mean?",
     "a": "**public / internal**: may be answered by any configured model. **confidential / restricted**: only "
          "local models (e.g. Ollama) unless an admin explicitly allows cloud use. Files containing personal data "
          "are classified confidential automatically."},
    {"id": "up-big", "cat": "Uploading documents", "page": "ingestion", "kw": "large many files background batch slow",
     "q": "How do I upload many or very large files?",
     "a": "Use **📥 Ingestion → Background upload**. Files are queued and indexed by background workers while you "
          "keep working; the page shows each job's progress and the reason if one fails."},
    {"id": "up-remove", "cat": "Uploading documents", "page": "library", "kw": "delete remove document hide chat",
     "q": "How do I stop a document from being used, or delete it?",
     "a": "On its card in **📚 Documents** switch off **Use in chat** (keeps it) or press **Remove**. Admins and "
          "owners can delete it permanently with all versions under *Access control & versions*."},
    # --- Formats
    {"id": "fmt-list", "cat": "Supported file formats", "page": "library", "kw": "formats types pdf word excel supported",
     "q": "Which file formats are supported?",
     "a": "PDF, Word (.docx), PowerPoint (.pptx), Excel (.xlsx), CSV, plain text, Markdown, HTML, JSON and images "
          "(PNG/JPG). Tables, equations and figures are recognised so you can ask about them directly."},
    {"id": "fmt-scan", "cat": "Supported file formats", "page": "guide", "kw": "scanned image ocr photo picture",
     "q": "Can it read scanned PDFs and photos?",
     "a": "Yes, with OCR. Local OCR uses *tesseract* (install it on the server; the Guide's *Start here* tab shows "
          "if it is available). Cards in Documents tell you when scanned pages could not be read."},
    {"id": "fmt-rejected", "cat": "Supported file formats", "page": "library",
     "kw": "rejected not accepted unsafe macro exe zip",
     "q": "Why was my file rejected?",
     "a": "Uploads are checked for safety: wrong or disguised file types (e.g. a program renamed to .pdf), macros, "
          "archive bombs and files over the size limit are refused. The message says which rule applied."},
    # --- Asking
    {"id": "ask-how", "cat": "Asking questions", "page": "chat", "kw": "ask question type chat how good",
     "q": "How do I ask good questions?",
     "a": "Be specific and use the words of the document: *What is the maximum input voltage of the LM7805?* "
          "works better than *Tell me about voltage*. Follow-up questions (*and the minimum?*) keep the context."},
    {"id": "ask-notfound", "cat": "Asking questions", "page": "chat", "kw": "not found answer missing no answer",
     "q": "Why do I get 'NOT FOUND IN DOCUMENT'?",
     "a": "The assistant never guesses: if the active documents do not contain the answer, it says so. Check that "
          "the right documents are switched on (**Use in chat**), rephrase with the document's terms, or open "
          "**🔎 Retrieval debugger** to see what the search finds."},
    {"id": "ask-docs", "cat": "Asking questions", "page": "library", "kw": "which documents choose select one document",
     "q": "How do I ask about just one document?",
     "a": "On its card in **📚 Documents** press **Ask →**: only that document is switched on for Chat. In Chat you "
          "can also narrow the search with **Filters** (category, tags, part number, element type)."},
    {"id": "ask-style", "cat": "Asking questions", "page": "settings", "kw": "answer length style short detailed",
     "q": "Can I make answers shorter or more detailed?",
     "a": "Yes: **⚙️ Settings → Answer style** (length, technical depth, citations, equations…). Feedback with 👍/👎 "
          "also tunes the style over time."},
    # --- Citations
    {"id": "cit-read", "cat": "Understanding citations", "page": "chat", "kw": "ref citation meaning label read",
     "q": "What does [REF-2: TABLE p3] mean?",
     "a": "It is a citation: *evidence 2, a table on page 3*. Open **Evidence** under the answer to read that "
          "exact passage (and the figure, if it is one)."},
    {"id": "cit-verified", "cat": "Understanding citations", "page": "chat",
     "kw": "verified partly check sources badge green",
     "q": "What do ✓ Verified and ≈ Partly verified mean?",
     "a": "After writing, every number and claim is compared with the cited passages. **✓ Verified**: all found. "
          "**≈ Partly verified**: some only weakly supported. **⚠ Check sources**: something could not be "
          "confirmed — read the evidence. Answers with unsupported claims are withheld instead of shown."},
    {"id": "cit-explain", "cat": "Understanding citations", "page": "chat",
     "kw": "how answer made explanation steps summary",
     "q": "How can I see how an answer was produced?",
     "a": "Under every answer: the **summary line** (documents searched, evidence found, method, model, cache, "
          "verification), **How this answer was made** (plain English) and **Agent activity**. These are factual "
          "execution records — the model's hidden reasoning is never shown."},
    # --- RAG
    {"id": "rag-what", "cat": "How RAG works", "page": "guide", "kw": "rag retrieval augmented generation meaning",
     "q": "What is RAG?",
     "a": "*Retrieval-Augmented Generation*: first **retrieve** the passages of your documents that match the "
          "question, then let an AI model **generate** an answer using only those passages. That is why answers "
          "can cite sources and why the assistant says 'not found' instead of inventing."},
    {"id": "rag-steps", "cat": "How RAG works", "page": "guide", "kw": "pipeline steps process flow happens",
     "q": "What happens when I press Enter?",
     "a": "1. Safety check of the question. 2. Permission check of the documents. 3. Cache lookup (repeated "
          "question?). 4. Search: meaning + keywords + fuzzy, merged and re-ranked. 5. An AI model writes the "
          "answer from the evidence. 6. Citations and numbers are verified. 7. Quality check, output filter, "
          "display. See **Guide → How it works** for a diagram."},
    {"id": "rag-offline", "cat": "How RAG works", "page": "models", "kw": "offline local internet no cloud ollama",
     "q": "Does it work offline?",
     "a": "Yes. Search, verification, OCR and this help assistant are local. For written answers connect a local "
          "model with Ollama (`OLLAMA_BASE_URL`). Without any model the app still shows the best matching passages."},
    # --- Agents
    {"id": "ag-what", "cat": "Agents and retrieval", "page": "traces", "kw": "agents what are agent workflow",
     "q": "What are the 'agents'?",
     "a": "Small specialised steps with fixed responsibilities: Query Understanding, Document Retrieval (text, "
          "tables, equations, figures), Reasoning (writes the answer), Citation Verification and Answer Quality. "
          "They run in a fixed order with time limits — see **🧭 Agent traces**."},
    {"id": "ag-hybrid", "cat": "Agents and retrieval", "page": "retrieval", "kw": "hybrid dense bm25 fuzzy rrf rerank",
     "q": "What are dense, BM25, fuzzy, RRF and rerank?",
     "a": "**dense** = meaning match (embeddings) · **BM25** = keyword match · **fuzzy** = approximate text match · "
          "**RRF** merges their rankings · **rerank** = a small model re-orders the best candidates. "
          "**🔎 Retrieval debugger** shows every score."},
    {"id": "ag-debug", "cat": "Agents and retrieval", "page": "retrieval", "kw": "wrong answer missed debug search",
     "q": "An answer missed something. How do I find out why?",
     "a": "Type the question in **🔎 Retrieval debugger**: if the right passage is not in the list, the search "
          "missed it (try other words or filters); if it is there, check **Agent traces** for that answer."},
    # --- Cache
    {"id": "ca-what", "cat": "Cache and repeated questions", "page": "chat", "kw": "cache cached repeated same question fast",
     "q": "Why was my answer instant ('⚡ From cache')?",
     "a": "You (or, if your admin enabled sharing, a colleague with the same access) asked the same or an "
          "equivalent question about the **same document versions with the same settings** earlier, and that "
          "answer passed verification. It is reused at zero AI cost."},
    {"id": "ca-safe", "cat": "Cache and repeated questions", "page": "chat",
     "kw": "cache safe stale wrong old permissions tenant",
     "q": "Can the cache give me an outdated or someone else's answer?",
     "a": "No. A cached answer is only reused for the same user/tenant boundary, the same document versions, "
          "filters, models and security policy; permissions are re-checked on every hit. Uploading a new version "
          "or deleting a document removes the related answers. Entries also expire (default 1 hour)."},
    {"id": "ca-not", "cat": "Cache and repeated questions", "page": "chat", "kw": "not cached follow up why not",
     "q": "Which answers are never cached?",
     "a": "Blocked questions, unverified or poorly grounded answers, 'not found' and withheld answers, failures, "
          "follow-up questions that depend on the conversation, and whole-document summaries."},
    {"id": "ca-fresh", "cat": "Cache and repeated questions", "page": "chat", "kw": "regenerate fresh force new answer",
     "q": "How do I get a fresh answer instead of the cached one?",
     "a": "Rephrase the question with different details, change a setting, or ask an admin to clear the cache "
          "(**⚡ Cache & help**). ↻ Regenerate re-runs the same question and may also be served from cache."},
    # --- Security
    {"id": "sec-blocked", "cat": "Security and privacy", "page": "chat", "kw": "blocked not processed red refused",
     "q": "Why was my question blocked?",
     "a": "The safety check stops instruction attacks (*ignore previous instructions…*), harmful requests, "
          "restricted topics and some personal data before any AI model is called. Rephrase it as a normal "
          "question about the documents. Admins see the event in the **Audit log**."},
    {"id": "sec-privacy", "cat": "Security and privacy", "page": "security", "kw": "privacy data where stored sent cloud",
     "q": "Where does my data go?",
     "a": "Documents and indexes stay on this server (or your configured storage). Only the few passages needed "
          "for an answer are sent to the chosen AI model; confidential documents go to local models only unless "
          "an admin allows otherwise. Telemetry never stores question text."},
    {"id": "sec-pii", "cat": "Security and privacy", "page": "security", "kw": "personal data pii card number redact",
     "q": "What happens to personal data in questions or answers?",
     "a": "It is detected (cards, IDs, IBANs, e-mails, phone numbers…) and handled by your organisation's policy: "
          "redacted, masked, blocked or shown only to authorised roles."},
    {"id": "sec-roles", "cat": "Security and privacy", "page": "users", "kw": "roles viewer editor admin permission",
     "q": "What can viewers, editors and admins do?",
     "a": "**viewer**: ask questions. **editor**: + upload and label own documents, run evaluations. **admin**: "
          "everything, including users, policies, models, audit log and the cache."},
    # --- Models
    {"id": "mod-config", "cat": "Models and API configuration", "page": "models",
     "kw": "configure model api key groq openai anthropic ollama",
     "q": "How do I connect an AI model?",
     "a": "Set it in `.env` (or Streamlit secrets) and restart: `OLLAMA_BASE_URL` for local Ollama, or a cloud "
          "key such as `GROQ_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`. "
          "**🤖 Models** shows which providers are reachable. Keys are never displayed in the app."},
    {"id": "mod-choose", "cat": "Models and API configuration", "page": "chat", "kw": "choose model select mode hybrid local cloud",
     "q": "Can I choose which model answers?",
     "a": "Yes: sidebar → **Model mode** (local, cloud or hybrid) and **Model** (automatic or a specific one). "
          "The server still enforces the rules — e.g. confidential documents stay on local models."},
    {"id": "mod-api", "cat": "Models and API configuration", "page": "guide", "kw": "rest api endpoint integrate token",
     "q": "Is there an API?",
     "a": "Yes, a REST API (`uvicorn api.main:app`): `/v1/auth/token`, `/v1/query`, `/v1/documents`, "
          "`/v1/documents/{id}/metadata`, `/v1/retrieval/debug`, `/v1/cache/stats` and more. It applies the same "
          "permissions and guardrails as the UI. See docs/USER_GUIDE.md."},
    {"id": "mod-free", "cat": "Models and API configuration", "page": "routing",
     "kw": "free models anyrouter groq fallback chain vision paid",
     "q": "Which models answer, and can the app ever use a paid model by accident?",
     "a": "With `ANYROUTER_API_KEY` set, text questions go to a Groq primary model with 3 Groq fallbacks, then 2 "
          "AnyRouter **verified-free** text models; image questions go to 3 AnyRouter verified-free vision models. "
          "A model counts as free only when AnyRouter reports it routable, free (not BYOK) and priced at zero; "
          "anything unknown is rejected. Paid providers are never used unless an admin sets "
          "`MODEL_ALLOW_PAID_FALLBACK=true` and `MODEL_FREE_ONLY=false`. See **🔀 Model routing**."},
    {"id": "mod-cost", "cat": "Models and API configuration", "page": "dashboard", "kw": "cost price budget spend tokens",
     "q": "How much does it cost and can I limit it?",
     "a": "Local models are free. Cloud cost per answer is shown under each answer and on the **Dashboard**. "
          "`DAILY_BUDGET_USD` stops AI answers once the daily budget is used; cached answers cost nothing."},
    # --- Metadata
    {"id": "md-what", "cat": "Metadata and document filters", "page": "library",
     "kw": "metadata tags category labels extracted",
     "q": "What is document metadata here?",
     "a": "Facts about a document: title, author, type, size, pages, category, tags, version, sensitivity, part "
          "numbers, technical parameters with units, tables, equations and figures. Most are **extracted "
          "automatically** (marked *auto*, with a confidence score and page) and you can confirm or correct them "
          "(marked *you*)."},
    {"id": "md-edit", "cat": "Metadata and document filters", "page": "library", "kw": "edit change correct metadata tag",
     "q": "How do I edit tags, category or title?",
     "a": "In **📚 Documents** open a card's **🏷️ Metadata** section → *Edit*. Your values always win over the "
          "automatic suggestions; leave a field empty to go back to the suggestion. Editing needs the editor or "
          "admin role."},
    {"id": "md-trust", "cat": "Metadata and document filters", "page": "library", "kw": "confidence accurate trust auto",
     "q": "Can I trust the automatically extracted values?",
     "a": "Treat them as suggestions: each shows a confidence (green ≥ 80 %, amber ≥ 50 %, red below) and the "
          "page it came from so you can check it. Answers in Chat always come from the document text itself, not "
          "from metadata."},
    {"id": "md-filter", "cat": "Metadata and document filters", "page": "chat",
     "kw": "filter narrow search part number category tag type",
     "q": "How do metadata filters improve answers?",
     "a": "In Chat, **Filters** limit the search to documents with a category, tag or part number, or to element "
          "types (tables, equations…). Fewer, more relevant passages give more precise answers. In Documents you "
          "can search and sort the library by the same fields."},
    # --- Troubleshooting
    {"id": "tr-slow", "cat": "Troubleshooting", "page": "dashboard", "kw": "slow latency wait long time",
     "q": "Answers are slow. What can I do?",
     "a": "Check the **Dashboard** (p95 latency) and **Agent traces** (which step is slow). Choose **⚡ Quick** "
          "answer mode in Settings, lower *Top-K*, or use a faster model. Repeated questions are served from cache."},
    {"id": "tr-nomodel", "cat": "Troubleshooting", "page": "guide", "kw": "no model connected evidence only no answer",
     "q": "I only get passages, no written answer.",
     "a": "No AI model is reachable. **Guide → Start here** shows the model status; add an API key or start Ollama "
          "and restart the app. Meanwhile the passages shown are still the best evidence for your question."},
    {"id": "tr-upload", "cat": "Troubleshooting", "page": "ingestion", "kw": "upload failed error stuck job",
     "q": "My upload failed or seems stuck.",
     "a": "Open **📥 Ingestion**: failed jobs show the reason. Damaged files: re-save and upload again. Temporary "
          "errors are retried automatically. Very large files: use background upload."},
    {"id": "tr-rate", "cat": "Troubleshooting", "page": "chat", "kw": "too many questions wait rate limit",
     "q": "It says 'Too many questions right now'.",
     "a": "A rate limit protects the service and your budget. Wait the number of seconds shown and ask again; "
          "admins can change `RATE_LIMIT_PER_MIN`."},
    {"id": "tr-error", "cat": "Troubleshooting", "page": "guide", "kw": "something went wrong error crash bug",
     "q": "I saw 'Something went wrong'.",
     "a": "Reload the page and try again. Details are written to the server log (never shown in the browser). "
          "If it repeats, note the time and the request id shown under the answer and tell your administrator."},
]

_STOP = {"the", "a", "an", "is", "are", "do", "does", "how", "what", "why", "can", "i", "my", "to", "of", "in", "it",
         "me", "this", "and", "or", "for", "on", "with", "be", "get", "you"}
_HELP_FILE = Path(__file__).resolve().parent.parent / "docs" / "USER_GUIDE.md"


def _words(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if w not in _STOP]


# ------------------------------------------------------------------ custom entries (admin managed)
def _custom_path(data_dir: Path) -> Path:
    return Path(data_dir) / "faq" / "custom.json"


def custom_entries(data_dir: Optional[Path]) -> List[Dict[str, object]]:
    if data_dir is None:
        return []
    from storage.cache import read_json

    data = read_json(_custom_path(data_dir), []) or []
    return [e for e in data if isinstance(e, dict) and e.get("q") and e.get("a")]


def add_custom(data_dir: Path, question: str, answer: str, category: str = "Getting started",
               page: str = "") -> Dict[str, object]:
    from storage.cache import write_json_atomic

    question, answer = " ".join(question.split())[:200], answer.strip()[:3000]
    if len(question) < 5 or len(answer) < 5:
        raise ValueError("Question and answer need at least 5 characters.")
    entry = {"id": "c-" + uuid.uuid4().hex[:8], "cat": category if category in CATEGORIES else CATEGORIES[0],
             "q": question, "a": answer, "page": page, "kw": "", "custom": True, "ts": round(time.time())}
    entries = custom_entries(data_dir) + [entry]
    write_json_atomic(_custom_path(data_dir), entries[-200:])
    return entry


def delete_custom(data_dir: Path, entry_id: str) -> bool:
    from storage.cache import write_json_atomic

    entries = custom_entries(data_dir)
    keep = [e for e in entries if e.get("id") != entry_id]
    if len(keep) != len(entries):
        write_json_atomic(_custom_path(data_dir), keep)
        return True
    return False


def all_entries(data_dir: Optional[Path] = None) -> List[Dict[str, object]]:
    return FAQ + custom_entries(data_dir)


# ------------------------------------------------------------------ search
def search(text: str, page_key: str = "", category: str = "", limit: int = 5,
           data_dir: Optional[Path] = None) -> List[Dict[str, object]]:
    """Ranked FAQ entries. Empty text → the entries for this page/category (context-aware suggestions)."""
    from rapidfuzz import fuzz

    entries = [e for e in all_entries(data_dir) if not category or e["cat"] == category]
    words = _words(text)
    if not words:
        here = [e for e in entries if e.get("page") == page_key]
        rest = [e for e in entries if e.get("page") != page_key]
        return (here + rest)[:limit]
    query = " ".join(words)
    scored = []
    for e in entries:
        hay_q = " ".join(_words(str(e["q"]) + " " + str(e.get("kw", ""))))
        hay_a = " ".join(_words(str(e["a"])))
        overlap = len(set(words) & set(hay_q.split())) / max(len(set(words)), 1)
        score = (0.55 * fuzz.token_set_ratio(query, hay_q) + 0.15 * fuzz.partial_ratio(query, hay_a)
                 + 30 * overlap + (6 if e.get("page") == page_key else 0))
        scored.append((score, e))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [e for s, e in scored if s >= 45][:limit]


def fallback(text: str, limit: int = 2) -> List[Dict[str, str]]:
    """When no FAQ entry fits: matching passages of the page help panels and docs/USER_GUIDE.md."""
    from rapidfuzz import fuzz

    from ui.help_content import PAGE_EXTRA, PAGE_HELP

    words = _words(text)
    if not words:
        return []
    query = " ".join(words)
    chunks: List[Dict[str, str]] = []
    for key, info in PAGE_HELP.items():
        body = f"{info['what']} " + " ".join(map(str, info["how"])) + " " + str(PAGE_EXTRA.get(key, {}).get("why", ""))
        chunks.append({"title": str(info["title"]), "text": body, "page": key})
    try:
        guide = _HELP_FILE.read_text(encoding="utf-8")
        for part in re.split(r"\n(?=#{1,3} )", guide):
            lines = part.strip().split("\n")
            if len(lines) > 1:
                chunks.append({"title": lines[0].lstrip("# ").strip(), "text": " ".join(lines[1:])[:1200], "page": ""})
    except OSError:
        pass
    scored = sorted(((fuzz.token_set_ratio(query, " ".join(_words(c["title"] + " " + c["text"]))), c) for c in chunks),
                    key=lambda t: t[0], reverse=True)
    return [c for s, c in scored if s >= 50][:limit]


# ------------------------------------------------------------------ UI
_PAGE_NAMES = {"guide": "🎓 Guide & Test Lab", "chat": "💬 Chat", "library": "📚 Documents", "validate": "✅ Validate",
               "dashboard": "🏠 Dashboard", "ingestion": "📥 Ingestion", "retrieval": "🔎 Retrieval debugger",
               "traces": "🧭 Agent traces", "evaluation": "🧪 Evaluation", "insights": "📊 Analytics",
               "settings": "⚙️ Settings", "models": "🤖 Models", "security": "🛡️ Security", "users": "👥 Users",
               "audit": "📜 Audit log", "cache": "⚡ Cache & help", "routing": "🔀 Model routing"}


def _answer(st, entry: Dict[str, object], prefix: str, page_key: str) -> None:
    st.markdown(f'<div class="faq-cat">{entry["cat"]}</div>', unsafe_allow_html=True)
    st.markdown(f"**{entry['q']}**")
    st.markdown(str(entry["a"]))
    target = str(entry.get("page") or "")
    pages = st.session_state.get("pages") or {}
    if target and target != page_key and pages.get(target) is not None:
        if st.button(f"Open {_PAGE_NAMES.get(target, target)} →", key=f"{prefix}_go_{entry['id']}"):
            st.switch_page(pages[target])


def _select(st, prefix: str, entry_id: object) -> None:
    """Quick-reply callback (runs before widgets render, so the search box may be cleared)."""
    st.session_state[f"{prefix}_sel"] = entry_id
    st.session_state[f"{prefix}_q"] = ""


def panel(page_key: str = "", prefix: str = "faq", data_dir: Optional[Path] = None) -> None:
    """The help chat: search box, categories, context suggestions (quick replies), answer, fallback."""
    import streamlit as st

    st.markdown("**👋 Hi! I answer questions about using this app.**")
    st.caption("Instant answers from the help guide — no AI model is used and nothing you type is stored.")
    text = st.text_input("Ask a question about the app", key=f"{prefix}_q", max_chars=200,
                         placeholder="e.g. how do I upload a file?")
    cat = st.selectbox("Category", ["All topics"] + CATEGORIES, key=f"{prefix}_cat")
    category = "" if cat == "All topics" else cat
    selected_id = st.session_state.get(f"{prefix}_sel")
    by_id = {str(e["id"]): e for e in all_entries(data_dir)}
    if text and text.strip():
        hits = search(text, page_key, category, limit=4, data_dir=data_dir)
        if hits:
            _answer(st, hits[0], prefix, page_key)
            if len(hits) > 1:
                st.caption("Related questions")
                for e in hits[1:]:
                    st.button(str(e["q"]), key=f"{prefix}_rel_{e['id']}", use_container_width=True,
                              on_click=_select, args=(st, prefix, e["id"]))
        else:
            docs = fallback(text)
            if docs:
                st.markdown("I don't have a ready answer, but the help guide says:")
                for d in docs:
                    st.markdown(f"**{d['title']}** — {d['text'][:400]}{'…' if len(d['text']) > 400 else ''}")
            else:
                st.info("I couldn't find that. Try other words or pick a category. Questions about the *content* of "
                        "your documents? Ask them in 💬 Chat.")
        return
    if selected_id and selected_id in by_id:
        _answer(st, by_id[selected_id], prefix, page_key)
        st.divider()
    st.caption("Suggested for this page" if not category else f"Questions about {category.lower()}")
    for e in search("", page_key, category, limit=5, data_dir=data_dir):
        if e["id"] == selected_id:
            continue
        st.button(str(e["q"]), key=f"{prefix}_s_{e['id']}", use_container_width=True,
                  on_click=_select, args=(st, prefix, e["id"]))


def launcher(page_key: str = "", data_dir: Optional[Path] = None) -> None:
    """Floating ❔ Help button (top right, every page) that opens the compact help chat."""
    import streamlit as st

    with st.container(key="faq_fab"):
        with st.popover("❔ Help", help="Questions about using the app? Ask here."):
            panel(page_key, "faq", data_dir)
