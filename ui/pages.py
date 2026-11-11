"""Pages: Chat · Library · Validate · Insights · Settings."""
from __future__ import annotations

import json
import secrets
import statistics
import time
from typing import Dict, List

import pandas as pd
import streamlit as st

from core.service import AskRequest
from ops import telemetry
from chat.conversations import result_from_dict
from learning.dpo_dataset import build_pairs
from learning.feedback import REASONS
from learning.preferences import DEPTHS, LENGTHS, Preferences
from llm.catalog import PROFILE_LABELS, ROLES, short_name
from metrics import validation
from metrics.rag_metrics import health_score
from rag.models import AgentStep, QueryResult
from rag.pdf_parser import PdfProcessingError
from ui.help_content import page_header, page_help
from ui import state, widgets

MAX_QUESTION_CHARS = 1000
GENERIC_SUGGESTIONS = ["Summarize this document", "What are the absolute maximum ratings?",
                       "What is the pin configuration?", "How do I calculate the power dissipation?"]


# =============================================================================
# Shared helpers
# =============================================================================
def ask(question: str, conv, on_step) -> QueryResult:
    """All checks (rate limits, guardrails, document permissions, routing) run server-side in RAGService."""
    session_key = st.session_state.setdefault("_rate_key", secrets.token_hex(8))
    req = AskRequest(question=question, indexes=state.active_indexes(), memory=conv.memory(state.settings().memory_turns),
                     previous_question=conv.last_question(), preferences=state.preferences().load().to_prompt(),
                     session_key=session_key, on_step=on_step,
                     model_override=st.session_state.get("model_override", ""),
                     llm_mode=st.session_state.get("llm_mode", ""),
                     metadata_filter=_chat_filter() or None,
                     extra={"workspace": state.workspace().id})  # cache + metadata scope without accounts
    return state.service().ask(req, state.principal())


def _meta_scope() -> str:
    from storage.metadata_store import scope_for

    return scope_for(state.principal(), state.workspace().id, state.settings().auth_mode)


def _chat_filter() -> Dict:
    """Active Chat filters as an AskRequest.metadata_filter dict (doc-level keys are resolved server-side)."""
    f = st.session_state.get("chat_filter") or {}
    return {k: v for k, v in f.items() if v}


def _run_with_status(question: str, conv) -> QueryResult:
    import threading

    try:
        from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
        ctx = get_script_run_ctx()
    except Exception:  # pragma: no cover - older Streamlit
        add_script_run_ctx, ctx = None, None
    with st.status("Thinking…", expanded=True) as box:
        def on_step(step: AgentStep) -> None:
            if step.kind not in ("stage", "action", "note"):
                return
            text = ("✓ " if step.kind == "stage" else "🔧 ") + widgets.describe_step(step)
            thread = threading.current_thread()
            prev = get_script_run_ctx(suppress_warning=True) if ctx is not None else None
            try:  # agents run in shared worker threads: bind THIS session only for the write, then restore
                if ctx is not None and prev is not ctx:
                    add_script_run_ctx(thread, ctx)
                box.write(text)
            except Exception:  # progress display is best effort; never break the answer
                pass
            finally:
                if ctx is not None and prev is not ctx:
                    _restore_ctx(thread, prev)
        result = ask(question, conv, on_step)
        box.update(label=f"Answered in {result.latency} s", state="complete", expanded=False)
    st.session_state["last_result"] = result
    return result


def _restore_ctx(thread, prev) -> None:
    try:
        from streamlit.runtime.scriptrunner_utils import script_run_context as _src

        setattr(thread, _src.SCRIPT_RUN_CONTEXT_ATTR_NAME, prev)
    except Exception:  # pragma: no cover
        pass


def _upload_box(key: str, label: str = "Drop documents here") -> None:
    from security.auth import has_permission

    if not has_permission(state.principal(), "upload"):
        st.caption("Your role can ask questions; uploading documents needs the editor or admin role.")
        return
    s = state.settings()
    cols = st.columns([3, 1])
    uploads = cols[0].file_uploader(label, type=list(s.allowed_extensions), accept_multiple_files=True, key=key,
                                    help=f"PDF, Word, PowerPoint, Excel, CSV, text, Markdown, HTML, JSON or images. "
                                         f"Up to {s.max_upload_mb} MB each.")
    sensitivity = cols[1].selectbox("Sensitivity", ["internal", "public", "confidential", "restricted"],
                                    key=f"{key}_sens", help="Confidential/restricted documents are only sent to "
                                    "local models unless an admin allows cloud use.")
    for up in uploads or []:
        marker = f"done_{key}_{up.name}_{up.size}"
        if st.session_state.get(marker):
            continue
        bar = st.progress(0.0, text=f"Reading {up.name}…")
        try:
            vision = state.vision()
            res = state.ingestion().ingest(up.getvalue(), up.name, state.principal(), sensitivity=sensitivity,
                                           progress=lambda f, m: bar.progress(min(max(f, 0.0), 1.0), text=m),
                                           vision=vision)
            if vision is not None and vision.calls:
                st.caption(f"{up.name}: {vision.calls} vision call(s) while indexing, ${vision.cost:.4f}")
            for old in res.superseded_hashes:
                state.remove_doc(old)
            state.add_doc(res.index)
            st.session_state[marker] = True
            label_ = {"new": "is ready", "new_version": f"updated to version {res.version}",
                      "duplicate": "was already indexed"}[res.status]
            st.toast(f"{res.index.filename} {label_}.", icon="✅")
            for w in res.warnings:
                st.warning(w)
        except PdfProcessingError as exc:
            st.error(f"{up.name}: {exc.user_message}")
        finally:
            bar.empty()


def _suggestions() -> List[str]:
    if "suggestions" not in st.session_state:
        qs: List[str] = []
        for index in state.active_indexes()[:2]:
            qs += [q["question"] for q in validation.generate_questions(index, limit=6)]
        picked = []
        for q in qs:  # mix question types: values, pins, equations
            if q not in picked and len(picked) < 3:
                picked.append(q)
        st.session_state["suggestions"] = (picked + [g for g in GENERIC_SUGGESTIONS if g not in picked])[:4]
    return st.session_state["suggestions"]


# =============================================================================
# Sidebar (all pages)
# =============================================================================
def sidebar() -> None:
    sync = state.sync_manager().status()
    with st.sidebar:
        st.caption("Verified answers from your documents")
        who = st.session_state.get("principal")
        if who is not None:
            c1, c2 = st.columns([3, 2])
            c1.caption(f"👤 {who.username} · {', '.join(who.roles)}")
            if c2.button("Sign out", key="logout"):
                state.logout()
                st.rerun()
        if st.button("✏️  New chat", use_container_width=True, type="primary"):
            conv = state.conversations().create([i.doc_id for i in state.active_indexes()])
            st.session_state["conv_id"] = conv.id
            st.session_state.pop("last_result", None)
            st.switch_page(st.session_state["pages"]["chat"])
        current_id = st.session_state.get("conv_id")
        convs = [c for c in state.conversations().list() if c.messages or c.id == current_id][:25]
        if any(c.messages for c in convs):
            widgets.section("Recent chats")
            current = st.session_state.get("conv_id")
            for c in convs:
                label = ("● " if c.id == current else "") + c.title
                if st.button(label, key=f"conv_{c.id}", use_container_width=True):
                    st.session_state["conv_id"] = c.id
                    st.session_state.pop("last_result", None)
                    st.switch_page(st.session_state["pages"]["chat"])
        st.session_state.setdefault("beginner_help", True)
        st.toggle("🎓 Beginner help", key="beginner_help",
                  help="Show 'What is this page?' panels and 'How this answer was made' explanations.")
        widgets.section("Answering")
        s = state.settings()
        modes = ["hybrid", "local", "cloud"]
        st.selectbox("Model mode", modes, index=modes.index(st.session_state.get("llm_mode") or s.llm_mode),
                     key="llm_mode", help="local = only on-premise models (Ollama); cloud = only cloud providers; "
                                          "hybrid = both. Confidential documents always stay local unless allowed.")
        import dataclasses as _dc
        from llm.model_router import ModelRouter
        options = [""] + [c[2] for c in ModelRouter(_dc.replace(s, llm_mode=st.session_state["llm_mode"],
                                                                manual_model="")).plan("master")[0]]
        current = st.session_state.get("model_override", "")
        st.selectbox("Model", options, index=options.index(current) if current in options else 0,
                     format_func=lambda e: "Automatic (router)" if not e else e, key="model_override")
        widgets.section("Status")
        lines = []
        for p in state.provider_status():
            if not p.configured:
                if p.name in ("groq", "ollama"):
                    need = "GROQ_API_KEY" if p.name == "groq" else "OLLAMA_BASE_URL"
                    lines.append(widgets.status_line(None, p.label, f"add {need} ({p.used_for})"))
                continue
            detail = "unavailable"
            if p.connected:
                k = p.key
                detail = f"{p.used_for}" + (f" · ${k.remaining:.2f} left" if k and k.remaining is not None else "")
            lines.append(widgets.status_line(p.connected, p.label, detail))
        if sync.enabled:
            ok = not sync.last_error
            lines.append(widgets.status_line(ok, "Storage", "saved to Hugging Face" if ok else sync.last_error))
        else:
            lines.append(widgets.status_line(None, "Storage", "temporary (resets on restart)"))
        st.markdown("".join(lines), unsafe_allow_html=True)


# =============================================================================
# Chat
# =============================================================================
def _feedback_widget(conv, msg, question: str) -> None:
    if msg.feedback:
        st.caption("Thanks for your feedback " + ("👍" if msg.feedback["rating"] > 0 else "👎"))
        return
    rating = st.feedback("thumbs", key=f"fb_{msg.id}")
    if rating is None:
        return
    reasons, comment = [], ""
    if rating == 0:
        reasons = st.pills("What went wrong?", REASONS, selection_mode="multi", key=f"fbr_{msg.id}") or []
        comment = st.text_input("Anything else? (optional)", key=f"fbc_{msg.id}", max_chars=500)
    if st.button("Send feedback", key=f"fbs_{msg.id}", type="primary"):
        r = msg.result or {}
        value = 1 if rating == 1 else -1
        state.feedback().record(conversation_id=conv.id, message_id=msg.id, question=question, answer=msg.content,
                                rating=value, reasons=list(reasons), comment=comment, provider=r.get("provider", ""),
                                model=r.get("model", ""), metrics=r.get("metrics", {}))
        msg.feedback = {"rating": value, "reasons": list(reasons), "comment": comment}
        state.conversations().save(conv)
        try:  # tenant-level feedback analytics (dashboard); the JSON store above keeps the learning loop
            from database.repository import get_repository
            pr = state.principal()
            get_repository(state.settings()).add_feedback(pr.tenant_id, pr.user_id, msg.id, value, list(reasons),
                                                          comment)
        except Exception:
            pass
        changes = state.preferences().apply_feedback(value, list(reasons))
        st.toast("Feedback saved." + (" I'll adapt: " + "; ".join(changes) if changes else ""), icon="🙏")
        st.rerun()


def chat_page() -> None:
    s = state.settings()
    conv = state.current_conversation()
    indexes = state.active_indexes()

    if not indexes:
        widgets.hero("Ask your documents anything",
                     "Upload a document, ask in plain English, and get an answer that cites the exact page, table "
                     "or equation — checked before you see it.",
                     [("Upload", "Add your files below, or load the sample documents."),
                      ("Ask", "Type a question in plain English."),
                      ("Verify", "Open the evidence and 'How this answer was made' under every answer.")])
        _onboarding()
        page_help("chat")
        from security.auth import has_permission

        c1, c2 = st.columns(2)
        if has_permission(state.principal(), "upload") and c1.button(
                "📥 First time? Load 4 sample documents", use_container_width=True, key="chat_samples",
                help="A datasheet (PDF), pump manual (Word), stock list (Excel) and install guide (Markdown) with "
                     "known answers, so you can see how everything works."):
            from ui.guide import _load_samples
            _load_samples()
            st.rerun()
        if c2.button("🎓 Open the Guide & Test Lab", use_container_width=True, key="chat_guide"):
            st.switch_page(st.session_state["pages"]["guide"])
        _upload_box("chat_upload")
        if state.active_indexes():
            st.rerun()
        return

    _onboarding()
    page_help("chat")
    chips = "".join(f'<span class="chip">📄 {widgets.esc(i.filename)}</span>' for i in indexes)
    fc1, fc2 = st.columns([5, 1.3])
    flt = _chat_filter()
    note = (f'<span class="badge info">🎚️ {len(flt)} filter(s) on</span>' if flt else "")
    fc1.markdown(f'<div class="meta" style="margin-bottom:.4rem">Answering from</div>{chips}{note}',
                 unsafe_allow_html=True)
    with fc2.popover("🎚️ Filters", use_container_width=True, help="Narrow the search with document metadata"):
        _filters_form(indexes)

    pending = st.session_state.pop("pending_question", None)
    if not conv.messages and not pending:
        widgets.hero("What would you like to know?",
                     "Pick a suggestion or type your own question. Follow-up questions keep the context.")
        cols = st.columns(2)
        for i, q in enumerate(_suggestions()):
            if cols[i % 2].button(q, key=f"sugg_{i}", use_container_width=True):
                st.session_state["pending_question"] = q
                st.rerun()

    last_q = ""
    assistants = [m for m in conv.messages if m.role == "assistant"]
    for msg in conv.messages:
        with st.chat_message(msg.role, avatar="🧑" if msg.role == "user" else "♾️"):
            if msg.role == "user":
                last_q = msg.content
                st.markdown(msg.content)
                continue
            result = result_from_dict(msg.result) if msg.result else QueryResult("answer", msg.content)
            widgets.render_result(result, s, stream=st.session_state.get("stream_id") == msg.id)
            if st.session_state.get("stream_id") == msg.id:
                st.session_state.pop("stream_id", None)
            c1, c2, _ = st.columns([1, 1.3, 4])
            with c1.popover("Copy"):
                st.code(msg.content, language="markdown")
            if msg is assistants[-1] and c2.button("↻ Regenerate", key=f"regen_{msg.id}"):
                new = _run_with_status(last_q, conv)
                state.conversations().replace_answer(conv, msg.id, new)
                st.rerun()
            _feedback_widget(conv, msg, last_q)

    question = st.chat_input("Ask about your documents…", max_chars=MAX_QUESTION_CHARS) or pending
    if question and question.strip():
        with st.chat_message("user", avatar="🧑"):
            st.markdown(question)
        with st.chat_message("assistant", avatar="♾️"):
            result = _run_with_status(question.strip(), conv)
        reply = state.conversations().add_exchange(conv, question.strip(), result)
        st.session_state["stream_id"] = reply.id
        st.rerun()


def _onboarding() -> None:
    """First-time checklist built from real state; hidden once complete or dismissed."""
    if st.session_state.get("onboard_dismissed") or not st.session_state.get("beginner_help", True):
        return
    convs = state.conversations().list()
    asked = any(m.role == "user" for c in convs for m in c.messages)
    rated = any(m.feedback for c in convs for m in c.messages if m.role == "assistant")
    helped = bool(st.session_state.get("faq_sel") or st.session_state.get("faq_q")
                  or "guide" in st.session_state.get("_help_seen", set()))
    items = [(bool(state.active_indexes()), "Add a document (or load the 4 sample documents)"),
             (asked, "Ask your first question in plain English"),
             (rated, "Open the evidence under an answer and rate it 👍 / 👎"),
             (helped, "Explore the ❔ Help assistant or the 🎓 Guide & Test Lab")]
    if all(d for d, _ in items):
        return
    c1, c2 = st.columns([6, 1])
    with c1:
        widgets.onboarding(items)
    if c2.button("Hide", key="onboard_hide", help="Hide this checklist for this session"):
        st.session_state["onboard_dismissed"] = True
        st.rerun()


def _filters_form(indexes) -> None:
    from storage.metadata_store import MetadataStore

    store, scope = MetadataStore(state.settings()), _meta_scope()
    cats, tags = set(), set()
    for ix in indexes:
        e = store.view(ix, scope)["effective"]
        cats.add(str(e["category"]["value"] or "other"))
        tags.update(str(t) for t in e["tags"]["value"] or [])
    cur = st.session_state.get("chat_filter") or {}
    st.caption("Search only documents / passages that match. Leave empty for no filter.")
    cat_opts = [""] + sorted(cats)
    cat = st.selectbox("Document category", cat_opts, index=cat_opts.index(cur.get("category", ""))
                       if cur.get("category", "") in cat_opts else 0, format_func=lambda c: c or "any",
                       key="cf_cat")
    tg = st.multiselect("Document tags (all must match)", sorted(tags),
                        default=[t for t in cur.get("tags", []) if t in tags], key="cf_tags")
    pn = st.text_input("Part number / identifier", value=cur.get("part_number", ""), max_chars=60, key="cf_pn",
                       help="Only documents whose metadata mentions this identifier (e.g. LM7805)")
    types = st.multiselect("Element types", ["text", "table", "row", "pin", "equation", "figure"],
                           default=cur.get("types", []), key="cf_types",
                           help="e.g. only tables for questions about values")
    a, b = st.columns(2)
    if a.button("Apply", type="primary", key="cf_apply", use_container_width=True):
        st.session_state["chat_filter"] = {"category": cat, "tags": tg, "part_number": pn.strip(), "types": types}
        st.rerun()
    if b.button("Clear", key="cf_clear", use_container_width=True):
        st.session_state["chat_filter"] = {}
        st.rerun()


# =============================================================================
# Library
# =============================================================================
def library_page() -> None:
    page_header("library")
    st.caption("Documents in your workspace. They are reopened automatically next time"
               + (" (saved to Hugging Face)." if state.sync_manager().enabled else "; storage is temporary on this server."))
    _upload_box("library_upload", "Add documents")
    docs = state.docs()
    if not docs:
        widgets.empty_state("📭", "Your library is empty",
                            "Drop files on the uploader above (PDF, Word, Excel, PowerPoint, text, images…). "
                            "New here? Load the sample documents from the 🎓 Guide & Test Lab.")
        return
    from storage.metadata_store import MetadataStore

    store, scope = MetadataStore(state.settings()), _meta_scope()
    views = {doc_id: store.view(d["index"], scope, _file_info(d)) for doc_id, d in docs.items() if d["index"] is not None}

    # --- search, filter, sort (metadata-aware) ---
    widgets.section("Find documents")
    f1, f2, f3, f4 = st.columns([3, 1.4, 1.6, 1.4])
    q = f1.text_input("Search", key="lib_search", placeholder="title, tag, part number, parameter, author…",
                      label_visibility="collapsed")
    cats = sorted({str(v["effective"]["category"]["value"] or "other") for v in views.values()})
    cat = f2.selectbox("Category", [""] + cats, format_func=lambda c: c or "All categories", key="lib_cat",
                       label_visibility="collapsed")
    all_tags = sorted({str(t) for v in views.values() for t in v["effective"]["tags"]["value"] or []})
    tags = f3.multiselect("Tags", all_tags, key="lib_tags", placeholder="Tags", label_visibility="collapsed")
    sort = f4.selectbox("Sort", ["Name", "Category", "Pages", "Active first"], key="lib_sort",
                        label_visibility="collapsed")
    from rag.metadata_intel import matches

    shown = []
    for doc_id, d in docs.items():
        v = views.get(doc_id)
        if v is None:
            shown.append((doc_id, d))  # missing index: always listed so it can be removed
            continue
        if matches(v["profile"], v["user"], query=q.strip(), tags=tags, category=cat):
            shown.append((doc_id, d))

    def sort_key(item):
        doc_id, d = item
        e = (views.get(doc_id) or {}).get("effective") or {}
        if sort == "Category":
            return (str((e.get("category") or {}).get("value", "")), d["filename"].lower())
        if sort == "Pages":
            return (-(e.get("pages") or 0), d["filename"].lower())
        if sort == "Active first":
            return (not d["active"], d["filename"].lower())
        return (d["filename"].lower(),)

    shown.sort(key=sort_key)
    st.caption(f"{len(shown)} of {len(docs)} document(s)" + (" match your search" if (q or cat or tags) else ""))
    if not shown:
        widgets.empty_state("🔍", "No document matches", "Clear the search box, category or tags to see everything.")
    cols = st.columns(2)
    for i, (doc_id, d) in enumerate(shown):
        with cols[i % 2]:
            index = d["index"]
            if index is None:
                st.markdown(widgets.missing_card(d["filename"]), unsafe_allow_html=True)
                if st.button("Remove", key=f"rm_{doc_id}"):
                    state.remove_doc(doc_id)
                    st.rerun()
                continue
            v = views[doc_id]
            st.markdown(widgets.doc_card(index.filename, index.stats(), index.doc_meta or {}, d["active"]),
                        unsafe_allow_html=True)
            e = v["effective"]
            chips = f'<span class="chip">🗂 {widgets.esc(e["category"]["value"])}</span>' + "".join(
                f'<span class="chip auto">#{widgets.esc(t)}</span>' for t in (e["tags"]["value"] or [])[:6])
            st.markdown(f'<div style="margin:.35rem 0 .2rem">{chips}</div>', unsafe_allow_html=True)
            a, b, c = st.columns([1.3, 1, 1])
            active = a.toggle("Use in chat", value=d["active"], key=f"act_{doc_id}")
            state.set_active(doc_id, active)
            if b.button("Ask →", key=f"ask_{doc_id}"):
                for other in docs:
                    state.set_active(other, other == doc_id)
                st.switch_page(st.session_state["pages"]["chat"])
            if c.button("Remove", key=f"rm_{doc_id}"):
                state.remove_doc(doc_id)
                st.rerun()
            sections = (index.doc_meta or {}).get("sections") or []
            if sections:
                with st.expander("Sections"):
                    st.markdown("\n".join(f"- {t} · p{p}" for t, p in sections[:60]))
            with st.expander("🏷️ Metadata"):
                _metadata_panel(doc_id, d, v, store, scope)
            st.write("")
    _library_admin()


def _file_info(d: Dict) -> Dict:
    """File facts from the database (accounts mode); empty without accounts."""
    if not d.get("document_id"):
        return {}
    try:
        from database.repository import get_repository

        repo = get_repository(state.settings())
        ver, doc = repo.current_version(d["document_id"]), repo.get_document(d["document_id"])
        return {"size_bytes": getattr(ver, "size_bytes", 0), "uploaded_at": getattr(ver, "created_at", 0),
                "sensitivity": getattr(doc, "sensitivity", ""), "owner": getattr(doc, "owner_id", "")}
    except Exception:
        return {}


def _can_edit_metadata(d: Dict) -> bool:
    from security.auth import has_permission

    p = state.principal()
    if not has_permission(p, "relabel"):
        return False
    if p.has("admin") or not d.get("document_id"):
        return True
    return _file_info(d).get("owner") == p.user_id


def _metadata_panel(doc_id: str, d: Dict, v: Dict, store, scope: str) -> None:
    prof, e = v["profile"], v["effective"]
    doc = prof.get("document", {})
    st.caption("**auto** = extracted automatically (a suggestion, check the confidence and page) · "
               "**file** = stored in the file · **you** = entered by a person (always wins).")
    rows = []
    for label, key in (("Title", "title"), ("Author", "author"), ("Category", "category"), ("Tags", "tags"),
                       ("Version", "version_label"), ("Part numbers", "part_numbers"), ("Description", "description")):
        f = e[key]
        val = ", ".join(map(str, f["value"])) if isinstance(f["value"], list) else str(f["value"] or "")
        rows.append({"field": label, "value": val or "—", "source": {"user": "you"}.get(f["source"], f["source"]),
                     "confidence": float(f.get("confidence") or 0.0)})
    for label, key in (("File type", "file_type"), ("Pages", "pages"), ("Size (bytes)", "file_size_bytes"),
                       ("Sensitivity", "sensitivity")):
        f = doc.get(key) or {}
        if f.get("value") not in (None, "", 0):
            rows.append({"field": label, "value": str(f["value"]), "source": f.get("source", "file"),
                         "confidence": float(f.get("confidence") or 0.0)})
    for k, val in (e["custom"]["value"] or {}).items():
        rows.append({"field": k, "value": val, "source": "you", "confidence": 1.0})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                 column_config={"confidence": st.column_config.ProgressColumn("confidence", min_value=0.0,
                                                                              max_value=1.0, format="%.2f")})
    counts = prof.get("counts") or {}
    refs = prof.get("references") or {}
    st.markdown(" ".join(f'<span class="stat"><b>{counts.get(k, 0)}</b> {lbl}</span>' for k, lbl in
                         (("chunks", "chunks"), ("tables", "tables"), ("equations", "equations"),
                          ("figures", "figures"))), unsafe_allow_html=True)
    params = prof.get("parameters") or []
    if params:
        widgets.section(f"Technical parameters ({len(params)}, auto-extracted)")
        st.dataframe(pd.DataFrame([{"name": p_["name"], "kind": p_["kind"], "qualifier": p_["qualifier"],
                                    "value": p_["value"], "unit": p_["unit"],
                                    "SI": "" if p_["si_value"] is None else f"{p_['si_value']:g} {p_['si_unit']}",
                                    "pages": ", ".join(map(str, p_["pages"])), "from": p_["origin"],
                                    "confidence": p_["confidence"]} for p_ in params[:150]]),
                     hide_index=True, use_container_width=True,
                     column_config={"confidence": st.column_config.ProgressColumn("confidence", min_value=0.0,
                                                                                  max_value=1.0, format="%.2f")})
    parts = prof.get("part_numbers") or []
    if parts:
        st.markdown("**Identifiers / part numbers** " + "".join(
            f'<span class="chip auto">{widgets.esc(x["value"])} {widgets.conf_badge(x["confidence"])}</span>'
            for x in parts[:15]), unsafe_allow_html=True)
    if prof.get("standards"):
        st.markdown("**Standards** " + ", ".join(prof["standards"][:10]))
    if refs.get("figures"):
        st.caption("Figures: " + " · ".join(f"{f['label']} (p{f['page']})" for f in refs["figures"][:8]))
    if not _can_edit_metadata(d):
        st.caption("Editing metadata needs the editor or admin role (and, for editors, owning the document).")
        return
    with st.form(f"md_form_{doc_id}"):
        st.markdown("**Edit** — leave a field empty to use the automatic suggestion.")
        usr = v["user"]
        c1, c2 = st.columns(2)
        title = c1.text_input("Title", usr.get("title", ""), max_chars=300, placeholder=str(e["title"]["value"])[:80])
        author = c2.text_input("Author", usr.get("author", ""), max_chars=200)
        c3, c4 = st.columns(2)
        category = c3.text_input("Category", usr.get("category", ""), max_chars=80,
                                 placeholder=f"suggested: {doc.get('category', {}).get('value', 'other')}")
        version = c4.text_input("Version label", usr.get("version_label", ""), max_chars=60)
        tags = st.text_input("Tags (comma separated)", ", ".join(usr.get("tags", [])),
                             placeholder=", ".join(prof.get("suggested_tags", [])[:6]))
        parts_in = st.text_input("Part numbers (comma separated)", ", ".join(usr.get("part_numbers", [])))
        desc = st.text_area("Description", usr.get("description", ""), max_chars=2000, height=80)
        if st.form_submit_button("💾 Save metadata", type="primary"):
            from storage.metadata_store import MetadataError

            try:
                store.update_user(doc_id, scope, {"title": title, "author": author, "category": category,
                                                  "version_label": version, "tags": tags,
                                                  "part_numbers": parts_in, "description": desc},
                                  actor=getattr(state.principal(), "user_id", ""))
                _audit("metadata_changed", d.get("document_id") or doc_id)
                st.toast("Metadata saved. Filters and search use it immediately.", icon="🏷️")
                st.rerun()
            except MetadataError as exc:
                st.error(str(exc))
    a, b = st.columns(2)
    if a.button("↺ Reset to automatic", key=f"md_reset_{doc_id}", help="Remove your values for this document"):
        store.reset_user(doc_id, scope)
        st.rerun()
    if state.principal().has("admin") and b.button("⟳ Re-extract", key=f"md_refresh_{doc_id}",
                                                   help="Run the automatic extraction again"):
        store.profile(d["index"], _file_info(d), refresh=True)
        st.toast("Metadata re-extracted.")
        st.rerun()


def _audit(action: str, target: str) -> None:
    try:
        from database.repository import get_repository

        p = state.principal()
        get_repository(state.settings()).audit(p.tenant_id, p.user_id, action, target, {})
    except Exception:
        pass


# =============================================================================
# Validate
# =============================================================================
def _library_admin() -> None:
    from ui.admin_pages import document_permissions_section

    document_permissions_section()


def validate_page() -> None:
    page_header("validate")
    st.caption("Check accuracy on YOUR documents. Questions are generated from your tables and equations with "
               "known answers; review them, then run the test.")
    indexes = state.active_indexes()
    if not indexes:
        st.info("Add a document in the Library first.")
        return
    ws = state.workspace()
    rows = st.session_state.get("val_rows") or ws.load_validation()
    c1, c2 = st.columns([2, 1])
    by_id = {i.doc_id: i for i in indexes}  # options must be plain values, not index objects
    target = by_id[c1.selectbox("Document", list(by_id), format_func=lambda d: by_id[d].filename)]
    count = c2.number_input("Questions", 3, 30, 10)
    if st.button("✨ Generate test questions", type="primary") or (not rows and st.session_state.get("val_auto") is None):
        st.session_state["val_auto"] = True
        rows = validation.generate_questions(target, limit=int(count))
        st.session_state["val_rows"] = rows
    if not rows:
        st.warning("No questions could be generated (no tables or formulas found). Add your own rows below.")
        rows = [{"question": "", "expected_pages": [1], "expected_answer_contains": [""]}]

    table = pd.DataFrame([{"question": r["question"],
                           "expected pages": ", ".join(map(str, r.get("expected_pages", []))),
                           "expected answer contains": ", ".join(map(str, r.get("expected_answer_contains", [])))}
                          for r in rows])
    edited = st.data_editor(table, num_rows="dynamic", use_container_width=True, hide_index=True, key="val_editor")
    questions = []
    for _, r in edited.iterrows():
        if str(r["question"]).strip():
            pages = [int(p) for p in str(r["expected pages"]).replace(" ", "").split(",") if p.isdigit()]
            expect = [x.strip() for x in str(r["expected answer contains"]).split(",") if x.strip()]
            questions.append({"question": str(r["question"]).strip(), "expected_pages": pages,
                              "expected_answer_contains": expect})
    if st.button("💾 Save this question set"):
        ws.save_validation(questions)
        st.toast("Saved to your workspace.")

    a, b = st.columns(2)
    run_retrieval = a.button("Check retrieval (free, no AI calls)", use_container_width=True)
    run_full = b.button(f"Full test ({len(questions)} AI answers)", use_container_width=True,
                        disabled=not state.router().enabled)
    if (run_retrieval or run_full) and questions:
        bar = st.progress(0.0)
        rep = validation.evaluate(questions, indexes, state.settings(), state.router(), with_answers=run_full,
                                  pause_s=2.0 if run_full else 0.0,
                                  progress=lambda f, m: bar.progress(min(max(f, 0.0), 1.0), text=m))
        bar.empty()
        st.session_state["val_report"] = rep
    rep = st.session_state.get("val_report")
    if rep:
        s = rep["summary"]
        widgets.metric_cards({"Retrieval hit rate": s["retrieval_hit_rate"], "MRR": s["mrr"], "NDCG@5": s.get("ndcg@5"),
                              "Answer accuracy": s["answer_accuracy"], "Cites correct page": s["cited_correct_page"],
                              "Verified": s["verified_rate"]})
        view = pd.DataFrame([{"question": r["question"], "expected": r["expected"],
                              "retrieval": "✅" if r["retrieval_hit"] else "❌",
                              "answer": "-" if "answer_correct" not in r else ("✅" if r["answer_correct"] else "❌"),
                              "verdict": r.get("verdict", "-")} for r in rep["rows"]])
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.download_button("Download report", validation.to_markdown(rep), "VALIDATION_REPORT.md")


# =============================================================================
# Insights
# =============================================================================
def _all_results() -> List[Dict]:
    return [{"conversation": c.title, **m.result} for c in state.conversations().list()
            for m in c.messages if m.role == "assistant" and m.result and m.result.get("metrics")]


def _system_tab() -> None:
    sm = telemetry.summary()
    if not sm["requests"]:
        st.info("System metrics (latency percentiles, tokens, cost, cache hits, errors) appear after a few questions.")
        return
    fmt = lambda v, f: "—" if v is None else f.format(v)  # noqa: E731
    c = st.columns(4)
    c[0].metric("P50 latency", fmt(sm["p50_latency_s"], "{:.2f} s"))
    c[1].metric("P95 latency", fmt(sm["p95_latency_s"], "{:.2f} s"))
    c[2].metric("Cache hit rate", fmt(sm["cache_hit_rate"], "{:.0%}"))
    c[3].metric("Error rate", fmt(sm["error_rate"], "{:.0%}"))
    c = st.columns(4)
    c[0].metric("Requests", sm["requests"])
    c[1].metric("Avg tokens", fmt(sm["avg_tokens"], "{:.0f}"))
    c[2].metric("Cost / request", fmt(sm["cost_per_request"], "${:.5f}"))
    c[3].metric("Rate-limited", sm["rate_limited"])
    c = st.columns(4)
    c[0].metric("Answered from cache", sm.get("cache_hits", 0))
    c[1].metric("Tokens saved by cache", f"{sm.get('cache_saved_tokens', 0):,}")
    c[2].metric("Cost saved by cache", f"${sm.get('cache_saved_cost', 0.0):.4f}")
    c[3].metric("Verified answers", fmt(sm.get("verified_rate"), "{:.0%}"),
                help="Share of answers whose citations passed verification (verified or partly verified).")
    st.caption("Since the server started. Full log: data/telemetry/requests.jsonl (no question text stored).")


def insights_page() -> None:
    page_header("insights")
    tab1, tab2, tab3, tab4 = st.tabs(["Quality", "Agent activity", "Feedback", "System"])
    with tab4:
        _system_tab()
    s = state.settings()
    results = _all_results()
    fb = state.feedback().stats()
    with tab1:
        if not results:
            st.info("Ask a few questions; quality metrics appear here.")
        else:
            df = pd.DataFrame([{**r["metrics"], "question": r.get("question", "")} for r in results])
            mean = {k: float(df[k].mean()) for k in df.columns if df[k].dtype.kind in "fi"}
            st.metric("RAG health score", f"{health_score(mean, s.health_weights, fb['ratio'])} / 100",
                      help="Application-level composite of faithfulness, relevance, citation accuracy, grounding and "
                           "user feedback. A monitoring aid, not a scientific measure of correctness.")
            widgets.metric_cards({"Faithfulness": mean.get("faithfulness"), "Relevance": mean.get("answer_relevance"),
                                  "Citation precision": mean.get("citation_precision"),
                                  "Citation accuracy": mean.get("citation_accuracy"),
                                  "Citation support": mean.get("citation_support"),
                                  "Numbers grounded": mean.get("numerical_score"),
                                  "Grounded answers": mean.get("grounded"), "User 👍": fb["ratio"]})
            c1, c2, c3 = st.columns(3)
            c1.metric("Answers", len(results))
            c2.metric("Avg latency", f"{mean.get('latency_s', 0):.2f} s")
            c3.metric("Avg tokens", f"{mean.get('tokens', 0):.0f}")
            stages: Dict[str, List[float]] = {}
            for r in results:
                for k, v in (r.get("stage_latency") or {}).items():
                    if k != "total":
                        stages.setdefault(k, []).append(v)
            if stages:
                widgets.section("Average latency by stage (ms)")
                st.bar_chart(pd.Series({k: statistics.mean(v) for k, v in stages.items()}, name="ms"))
            with st.expander("All answers"):
                cols = [c for c in ("question", "health_score", "faithfulness", "citation_accuracy", "citation_support",
                                    "numerical_score", "grounded", "latency_s", "tokens") if c in df.columns]
                st.dataframe(df[cols], hide_index=True, use_container_width=True)
    with tab2:
        result = st.session_state.get("last_result")
        if result is None:
            conv = state.current_conversation()
            last = next((m for m in reversed(conv.messages) if m.role == "assistant" and m.result), None)
            result = result_from_dict(last.result) if last else None
        if result is None:
            st.info("Ask a question to see how the agents handled it.")
        else:
            st.markdown(f"**Question:** {result.question or '(legacy mode)'}")
            if result.plan:
                st.markdown(" ".join(f'<span class="chip">{widgets.esc(i)}</span>' for i in result.plan.get("intents", [])),
                            unsafe_allow_html=True)
                for name in ("text_agent", "table_agent", "equation_agent", "figure_agent", "document_agent"):
                    ran = name in result.plan.get("agents", [])
                    st_ = (result.agent_stats or {}).get(name, {})
                    st.markdown(f"{'✅' if ran else '⏭️'} **{name.replace('_', ' ').title()}**"
                                + (f": {int(st_.get('results', 0))} candidates · {st_.get('latency_ms', 0):.1f} ms"
                                   if ran else " · skipped (not needed)"))
                cl = (result.agent_stats or {}).get("concept_linker")
                if cl:
                    st.markdown(f"🔗 **Concept Linker**: {int(cl.get('concepts', 0))} concept(s) · "
                                f"{int(cl.get('results', 0))} linked item(s) added · {int(cl.get('kept', 0))} kept · "
                                f"~{int(cl.get('extra_tokens', 0))} extra tokens · {cl.get('latency_ms', 0):.1f} ms")
            if result.stage_latency:
                widgets.section("Latency by stage")
                st.dataframe(pd.DataFrame([{"stage": k, "ms": round(v, 1)} for k, v in result.stage_latency.items()]),
                             hide_index=True)
            widgets.section("Steps (actions only)")
            widgets.render_trace(result.steps)
            if result.verification:
                with st.expander("Verification report"):
                    st.json(result.verification)
    with tab3:
        c1, c2, c3 = st.columns(3)
        c1.metric("👍 Helpful", fb["up"])
        c2.metric("👎 Not helpful", fb["down"])
        c3.metric("Helpful rate", "-" if fb["ratio"] is None else f"{fb['ratio'] * 100:.0f}%")
        if fb["reasons"]:
            st.bar_chart(pd.Series(fb["reasons"], name="count"))


# =============================================================================
# Settings
# =============================================================================
def settings_page() -> None:
    page_header("settings")
    t1, t2, t3, t4 = st.tabs(["Workspace & storage", "AI models", "Answer style", "Advanced"])
    s = state.settings()
    with t1:
        ws = state.workspace()
        st.markdown("**Your private workspace link.** Bookmark it to come back to your chats and documents. "
                    "Anyone with this link can open the workspace, so keep it private.")
        st.code(f"?ws={ws.id}", language=None)
        st.caption("Add it to the end of the app address, e.g. https://your-app.streamlit.app/?ws=…")
        with st.form("switch_ws"):
            code = st.text_input("Open another workspace (paste its code)")
            if st.form_submit_button("Open"):
                if state.switch_workspace(code.strip().split("ws=")[-1]):
                    st.rerun()
                st.error("That doesn't look like a workspace code.")
        sync = state.sync_manager().status()
        widgets.section("Storage")
        if sync.enabled:
            ago = f"{int(time.time() - sync.last_push)} s ago" if sync.last_push else "not yet"
            st.markdown(f'<div class="store">☁️ <b>Saved to your private Hugging Face dataset</b><br>'
                        f'Last upload: {ago} · pending changes: {sync.pending}'
                        f'{"<br>⚠ " + widgets.esc(sync.last_error) if sync.last_error else ""}</div>',
                        unsafe_allow_html=True)
            if st.button("Save now"):
                ok = state.sync_manager().flush("manual")
                st.toast("Saved." if ok else "Upload failed; it will be retried automatically.")
        else:
            st.markdown('<div class="store">⚠️ <b>Temporary storage.</b> Chats, feedback and indexes are lost when the '
                        'server restarts. Add HF_TOKEN and HF_DATASET_REPO to the secrets to keep them (see '
                        'DEPLOYMENT.md).</div>', unsafe_allow_html=True)
        conv = state.current_conversation()
        st.download_button("Export current chat (Markdown)", state.conversations().export_markdown(conv),
                           f"chat_{conv.id}.md")
        if st.button("Delete current chat"):
            state.conversations().delete(conv.id)
            st.session_state.pop("conv_id", None)
            st.rerun()
    with t2:
        st.caption("Text questions are answered by Groq (free). Only figure questions and scanned pages go to "
                   "Qwen / Gemma on OpenRouter.")
        statuses = state.provider_status()
        st.markdown("".join(widgets.status_line(p.connected if p.configured else None, p.label,
                                                (f"connected · {p.used_for}" if p.connected else p.message or "unavailable")
                                                if p.configured else f"not configured · {p.used_for}")
                            for p in statuses), unsafe_allow_html=True)
        by_name = {p.name: p for p in statuses}
        orr = by_name.get("openrouter")
        key = orr.key if orr is not None and orr.connected else None
        if key is not None:
            c = st.columns(3)
            c[0].metric("OpenRouter spent (this key)", f"${key.usage:.4f}")
            c[1].metric("OpenRouter key limit", "none" if key.limit is None else f"${key.limit:.2f}")
            c[2].metric("Spent today (answers)", f"${telemetry.spent_today(s.data_dir):.4f}")
            if key.limit is None:
                st.caption("Tip: set a credit limit on this key at openrouter.ai/settings/keys - it is the hard "
                           "spending cap, enforced by OpenRouter itself.")
        profiles = list(PROFILE_LABELS)
        prof = st.radio("Model profile", profiles, index=profiles.index(s.model_profile),
                        format_func=PROFILE_LABELS.get)
        state.set_override(model_profile=prof)
        s = state.settings()
        rows = [{"Role": r, "Model chain (tried in order)": " → ".join(s.models_for(r))} for r in ROLES]
        rows.append({"Role": "embeddings", "Model chain (tried in order)":
                     s.embed_model_remote if s.remote_embeddings else f"local: {s.embed_model}"})
        rows.append({"Role": "reranker", "Model chain (tried in order)":
                     (s.rerank_model_remote + " → local fallback") if s.remote_rerank else f"local: {s.rerank_model}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        groq = by_name.get("groq")
        available = [f"groq:{m}" for m in groq.models] if groq is not None and groq.connected else []
        if available:
            current = s.models_for("master")[0] if s.models_for("master") else ""
            options = list(dict.fromkeys([current] + available))
            pick = st.selectbox("Text answer model (Groq)", options, index=0, format_func=short_name,
                                help="Models on your Groq account. The rest of the chain stays as fallback.")
            if pick != current:
                state.set_override(master_models=(pick,) + tuple(m for m in s.models_for("master") if m != pick))
        mode = st.radio("Answer mode", ["agent", "quick"], index=0 if s.agent_mode == "agent" else 1, horizontal=True,
                        format_func=lambda x: "🧠 Thorough (agent uses tools when needed)" if x == "agent"
                        else "⚡ Quick (one AI call, cheapest)")
        efforts = ["none", "low", "medium", "high"]
        effort = st.select_slider("Hidden reasoning effort (master)", efforts, value=s.reasoning_effort,
                                  help="More reasoning helps calculations and comparisons; it costs more tokens. "
                                       "The reasoning itself is never shown or stored.")
        state.set_override(agent_mode=mode, reasoning_effort=effort)
    with t3:
        store = state.preferences()
        p = store.load()
        with st.form("prefs"):
            length = st.select_slider("Answer length", LENGTHS, value=p.response_length)
            depth = st.select_slider("Technical depth", DEPTHS, value=p.technical_depth)
            c = st.columns(4)
            cit, eq = c[0].toggle("Citations", p.citations), c[1].toggle("Equations", p.equations)
            fig, tab = c[2].toggle("Figures", p.figures), c[3].toggle("Tables", p.tables)
            if st.form_submit_button("Save", type="primary"):
                store.save(Preferences(length, depth, cit, eq, fig, tab, p.history))
                st.toast("Saved.")
        if p.history:
            with st.expander(f"Learned from your feedback ({len(p.history)})"):
                for h in reversed(p.history[-20:]):
                    st.write("• " + "; ".join(h["changes"]))
        pairs = build_pairs(state.feedback().all())
        st.download_button(f"Download training pairs (DPO, {len(pairs)})",
                           "\n".join(json.dumps(x, ensure_ascii=False) for x in pairs), "dpo_dataset.jsonl",
                           disabled=not pairs)
    with t4:
        c1, c2, c0 = st.columns(3)
        rerank = c1.toggle("Reranking (local cross-encoder)", s.rerank)
        imgs = c2.toggle("Figure questions: send images to Qwen/Gemma", s.image_answers)
        ocr = c0.toggle("OCR scanned pages (Qwen/Gemma)", s.ocr_scanned)
        c3, c4, c5 = st.columns(3)
        vllm = c3.toggle("Extra LLM cross-check (more tokens)", s.verify_with_llm)
        vis = st.toggle("Describe ALL figures while indexing (many vision calls)", s.vision_index)
        regen = c4.number_input("Max corrections", 0, 2, s.max_regenerations)
        steps = c5.slider("Agent tool rounds", 1, 6, s.agent_max_steps)
        c6, c7 = st.columns(2)
        memory = c6.slider("Conversation memory (turns)", 0, 10, s.memory_turns)
        top_k = c7.slider("Top-K per retriever", 1, 10, s.top_k)
        widgets.section("Health score weights")
        names = ["Faithfulness", "Relevance", "Citations", "Grounding", "Feedback"]
        wc = st.columns(5)
        weights = [wc[i].number_input(n, 0.0, 1.0, float(round(s.health_weights[i], 2)), 0.05) for i, n in enumerate(names)]
        total = sum(weights) or 1.0
        state.set_override(rerank=rerank, vision_index=vis, ocr_scanned=ocr, verify_with_llm=vllm, image_answers=imgs, max_regenerations=int(regen),
                           agent_max_steps=steps, memory_turns=memory, top_k=top_k,
                           health_weights=tuple(w / total for w in weights))
