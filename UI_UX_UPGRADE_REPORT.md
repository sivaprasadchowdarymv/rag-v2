# UI/UX Upgrade Report — RAG∞ Ultimate 2.1

Design reference: `uxui-workflow` (requirements → UX → UI, "go wide then narrow", rule of three). That workflow is
interactive (the user picks at every checkpoint); per the master prompt's instruction to decide autonomously, the
decisions below record the option chosen and why. Screenshots: `docs/screenshots/`.

## 1. Problem framing (phase 1 — requirements)

| Who | Need | Pain in 2.0 |
|---|---|---|
| First-time user | Understand what the app does and get a first verified answer without help | Guide existed but was one page among 15; no progress feedback; no in-context answers to "how do I…?" |
| Daily user | Ask, trust, verify; ask the same things again | No signal that a repeated answer was instant/free; explanation spread over 4 expanders |
| Document owner | Organise many files, find them, narrow searches | Only file name + stats; no labels, search or filters in the library |
| Admin / operator | Health, cost, safety, cache | Dashboard of plain `st.metric`s; no cache visibility or control |

Success criteria: a new user reaches a verified answer and knows where it came from without outside help; every
page explains *what / why / how / what happens / what if it fails*; nothing that worked in 2.0 changes behaviour.

## 2. UX decisions (phase 2 — three concepts, one chosen)

| Concept | Idea | Decision |
|---|---|---|
| A. New single-page "assistant" shell | Replace navigation with one chat-centric screen | ✗ breaks existing navigation, URLs, role gating and tests |
| B. Wizard-first onboarding | Force a multi-step setup wizard before use | ✗ blocks experienced users; duplicates the Guide |
| **C. Progressive layer on the existing IA** | Keep the 15 pages and groups; add a consistent page header, help that answers 5 questions, a floating help assistant, a dismissible onboarding checklist driven by real state, richer answer and library cards | ✔ preserves everything, adds guidance where users need it |

Help assistant: three options — (1) LLM chatbot, (2) static FAQ page, (3) **deterministic, searchable, page-aware
FAQ in a floating panel** ✔ (free, offline, cannot leak data, always available).

## 3. UI decisions (phase 3 — directions and variations)

Directions considered: *vivid gradient SaaS*, *dense dark console*, **calm enterprise (indigo + neutral,
translucent surfaces)** ✔ — works unchanged in Streamlit light and dark themes, high contrast, no heavy gradients.
Variation chosen for the help launcher: bottom-right floating button (✗ — the visual review showed it covered the
chat box's send button) → **top-right pill next to the app menu** ✔.

Design system (`ui/styles.py`): one accent `#4F46E5` + semantic ok/warn/bad; 14 px radius; 4/8 px spacing; one
subtle shadow; Inter with system fallback (offline-safe); `:focus-visible` outlines on all controls;
`prefers-reduced-motion` honoured; mobile breakpoints; sidebar collapses automatically on phones. All 2.0 CSS class
names are kept, so no existing markup broke.

## 4. What changed, page by page (capabilities unchanged)

| Page | Additions |
|---|---|
| All pages | Uniform header (icon · group · title · one-line purpose) via `page_header()`; help panel now answers **what · why you need it · how to use it · what happens when you act · if something goes wrong · how to test it**; floating **❔ Help** assistant |
| 💬 Chat | Onboarding checklist (4 real-state steps with progress bar, *Hide*); **🎚️ Filters** popover (category, tags, part number, element types) with an "n filter(s) on" badge; **⚡ From cache** badge; **📋 How this answer was produced** grid (documents searched, evidence, retrieval method — honest when dense search is offline —, model, cache, sources, verification); live progress now actually renders from agent threads |
| 📚 Documents | Find bar (full-text over metadata, category, tags, sort); category/tag chips per card; **🏷️ Metadata** viewer (source *auto/file/you* + confidence bars + page provenance, parameter table with SI values, identifiers, standards, figures) and editor (role-gated, reset, re-extract); empty states |
| 🏠 Dashboard | KPI tiles grouped *Content & usage · Speed & cache · Quality, cost & safety* (documents, chunks, questions, active users, p50/p95, cache hit/miss, savings, verification rate, withheld, feedback, tokens, cost, blocked, ingestion failures, error rate) — all from real counters |
| 📊 Analytics | System tab: answered from cache, tokens/cost saved, verified-answer rate |
| ⚡ Cache & help (new, admin) | Cache status/hit rate/savings/expiry/evictions, *why not cached* table, retrieval-cache stats, per-tenant **Clear**; custom help questions (add/delete); metadata overview |
| 🎓 Guide | New **⑤ Help center** tab (same assistant inline) |
| Other pages | Header/help only; content untouched |

## 5. Accessibility & responsiveness
Headings exposed with `role="heading" aria-level="1"`; KPI/summary grids are lists for screen readers; all controls
are native Streamlit widgets (keyboard operable) with visible focus; colour is never the only signal (badges carry
text: "✓ Verified", "⚡ From cache", confidence as %); grids reflow with `auto-fit` down to 390 px (screenshot
`08_mobile.png`).

## 6. Evidence
* 18 AppTest UI tests (`tests/integration/test_ui_upgrade.py`): new page, launcher on 6 pages, FAQ search + quick
  reply + fallback, first-time onboarding/empty states, cache badge + summary on a repeated question, dashboard
  tiles, metadata view/search/edit, viewer cannot edit, chat filters, help center, page headers. The 21 original UI
  tests pass unchanged.
* Visual review in headless Chrome (desktop 1400 px and phone 390 px) → three fixes (launcher position, page title
  element, sidebar on mobile).

## 7. Known limitations
* Streamlit controls the base widget styling; deeper restyling would need a custom component (not added).
* The help launcher's popover closes when one of its buttons reruns the page in some Streamlit versions; the
  selected answer is kept in session state and the same assistant is available inline in Guide → Help center.
* Fonts load from Google Fonts when reachable; offline the system font stack is used.
