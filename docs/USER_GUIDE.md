# User Guide

> **New in 2.1:** ❔ Help assistant on every page · first-time checklist · ⚡ instant answers for repeated
> questions (verified-answer cache) · 🏷️ document metadata you can review, edit, search and filter by ·
> 📋 "How this answer was produced" summary under every answer.

## Your first 5 minutes
1. Open **🎓 Guide & Test Lab** → *Start here*. Anything with a red dot tells you exactly what to fix.
2. *Test Lab* tab → **Load sample documents** (or upload your own files in **📚 Documents**).
3. Go to **💬 Chat** and ask *What is the dropout voltage?*
4. Under the answer open **Evidence** (the exact passages) and **📋 How this answer was produced**.
5. Rate the answer 👍 / 👎. The checklist at the top of Chat ticks off each step and disappears when you are done.

Stuck at any point? Press **❔ Help** (top right of every page) and type your question, e.g. *how do I upload
a file?* It answers instantly from this guide — no AI model is used and nothing you type is stored.

## Sign in
With accounts enabled you sign in with the username and password your administrator gave you. Otherwise the
app opens directly (single-user mode); bookmark the link, it holds your private workspace.

## Add documents — *Documents*
Drop files on the uploader: **PDF, Word (.docx), PowerPoint (.pptx), Excel (.xlsx), CSV, text, Markdown, HTML,
JSON, PNG/JPG** (images and scanned pages are read with OCR). Choose a **sensitivity**:

| Label | Meaning |
|---|---|
| public / internal | may be answered by any enabled model |
| confidential / restricted | answered only by **local** models unless an administrator allows cloud use |

Documents containing personal data (IDs, account numbers, …) are classified confidential automatically.
Uploading a file with the same name again creates a **new version**; the old version is no longer searched.
Identical files are stored once. Large batches: *Ingestion* → *Background upload*; follow progress there.

## Ask — *Chat*
Type a question. The answer cites its evidence as `[REF-n: TYPE pX]`; open **Evidence** to see each passage
(page, section, table row, figure). A badge shows whether citations were **verified**. Follow-up questions keep
context ("and at 85 °C?"). Use **Copy**, **Regenerate**, and 👍 / 👎 feedback (the assistant adapts its style).

In the sidebar: **Model mode** (hybrid / local / cloud) and **Model** (automatic routing or a specific model).
**Why this model** under an answer explains the routing decision (e.g. "confidential data: local only").

What you may see instead of an answer:

| Message | Why |
|---|---|
| "…attempt to override the assistant's instructions…" | the question looked like a prompt-injection attempt |
| "…outside what this assistant can help with" / out-of-scope | content or topic policy of your organisation |
| "I could not find enough reliable evidence…" | the draft answer was not supported by the documents, so it was withheld |
| "NOT FOUND IN DOCUMENT" | the documents do not contain the answer |
| "Personal data in your question was redacted" | e-mails, IDs etc. are removed before any model sees them |

### Filters (Chat)
**🎚️ Filters** next to the document list narrows the search: document **category**, **tags**, a **part number /
identifier**, and **element types** (only tables, only equations…). A badge shows how many filters are on;
**Clear** removes them. Fewer, more relevant passages give more precise answers.

### Instant answers for repeated questions (⚡ From cache)
When you ask a question that you already asked — in the same or equivalent words (*What's the max. output
current?* = *what is the maximum output current*) — about the **same document versions with the same settings**,
the earlier **verified** answer is shown instantly at zero AI cost, marked **⚡ From cache**. It is never reused:
for other users (or, if your admin chose tenant sharing, for users with other roles) or other organisations;
after a document is updated or deleted; for questions that differ in numbers, identifiers, *min/max* or *not*
(*pin 3* never answers *pin 4*); for follow-up questions; or for answers that were blocked, unverified, withheld or
failed. Cached answers expire after one hour by default.

### How this answer was produced
The **📋 summary** under each answer lists: documents searched · evidence found · retrieval method · model ·
cache status · sources cited · verification result. It is a factual record of what ran; the model's internal
reasoning is never shown.

## Document metadata — *Documents → 🏷️ Metadata*
Each document card shows its category and tags. Open **🏷️ Metadata** to see:

| Shown | Meaning |
|---|---|
| **auto** | extracted automatically — a *suggestion*; check the **confidence** bar and the page it came from |
| **file** | stored in the file itself (size, type, page count, embedded title) |
| **you** | entered by a person — always wins over automatic values |

Automatically extracted: title, author, file type and size, pages, category, tags, version, sensitivity, part
numbers / identifiers, standards (ISO, IEC…), **technical parameters** (e.g. `VOUT = 5.0 V`, table rows with
Min/Typ/Max) with the original text, the unit **normalised to SI** (500 mA → 0.5 A) and the page, plus tables,
equations and figure references.

**Edit** (editor or admin role; editors only for their own documents): title, author, category, version label,
tags, part numbers, description. Leave a field empty to return to the automatic suggestion; **↺ Reset to
automatic** removes all your values. Admins can **⟳ Re-extract**. Labels are private to your organisation (or
your workspace without accounts).

**Find documents** at the top of the Documents page searches titles, tags, part numbers, parameters, authors and
custom fields, filters by category and tags, and sorts by name, category, pages or *active first*.

## Other pages
* **Validate** — generate questions from a document and check answers.
* **Dashboard / Analytics** — usage, latency, cost, feedback.
* **Retrieval debugger** — see how a query is rewritten and every candidate's dense, BM25, fuzzy, fusion and
  rerank scores, with filters by document, element type and page.
* **Agent traces** — the agents that ran for each answer, their timing, routing, verification and security
  decisions.
* **Evaluation** — run the benchmark (baseline vs upgraded) and download reports.
* **⚡ Cache & help** (admins) — cache hit rate, savings, why answers were not cached, **Clear cache** (your
  organisation only), your own help questions for the ❔ Help assistant, and a metadata overview of all documents.

## Backup API keys (2.2.1)
Add extra keys in `.env` as `GROQ_API_KEY_2=`, `GROQ_API_KEY_3=` … and `OPENROUTER_API_KEY_2=` … (up to 9 each).
When a key is rejected, out of credit or rate-limited, the app retries the same request with the next key, and only
then moves to the next model. **Administration → 🤖 Models** shows each key's state (ok / cooling / rejected) without
ever displaying the key.

## Free-model routing (2.2, admins: *🔀 Model routing*)
With `ANYROUTER_API_KEY` set, text questions are answered by a Groq model, falling back to three other Groq models
and then two verified-free AnyRouter models; figure questions go to three verified-free AnyRouter vision models.
The page shows each slot, whether its free access was verified, vision capability, last health check, successes,
failures and latency, and has **Test connection** and **Test vision model** buttons. A model is never used if it is
paid, BYOK-only or has unknown pricing. If no vision model can read a figure, the answer uses captions and nearby
text and says so.

## REST API additions (2.1, backwards compatible)
* `POST /v1/query` responses now also contain `cached` (bool) and `cache` (status, match, savings).
* `GET /v1/documents/{id}/metadata`, `PATCH /v1/documents/{id}/metadata` (editor/owner or admin),
  `GET /v1/metadata/search?q=&tag=&category=&part_number=`.
* `GET /v1/cache/stats`, `DELETE /v1/cache` (admin; affects only your tenant).
* `filters` of `/v1/query` and `/v1/retrieval/debug` accept `category`, `tags`, `part_number`, `query`,
  `file_type` (document level) and `part_numbers`, `parameter_kinds` (passage level) besides the existing keys.

## Troubleshooting
| Problem | What to do |
|---|---|
| Only passages, no written answer | No AI model is reachable — see *Guide → Start here*; add a key or start Ollama |
| "NOT FOUND IN DOCUMENT" | Check *Use in chat* on the right documents, remove filters, rephrase with document terms |
| Upload failed | *Ingestion* shows the reason; re-save damaged files; programs/macros are refused on purpose |
| Answers slow | *Dashboard* (p95) and *Agent traces* show the slow step; try ⚡ Quick mode in Settings |
| "Too many questions right now" | Wait the seconds shown (rate limit) |
| "Something went wrong" | Reload; details are in the server log — give your admin the request id under the answer |

## Tips
Ask specific questions with symbols and units ("max VIN at 25 °C"). For summaries say "Summarize this document".
Use the debugger when an answer misses: it shows whether the right passage was found.
