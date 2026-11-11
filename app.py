"""
RAG∞ Ultimate — verified answers from your documents, Groq for text, Qwen 3.8 / Gemma 4 for images.

    streamlit run app.py

Pages: Start here (Guide & Test Lab) · Workspace (Chat, Documents, Validate) · Operations (Dashboard, Ingestion, Retrieval debugger,
Agent traces, Evaluation, Analytics) · Administration (Settings, Models, Security, Users & roles, Audit log,
Cache & help). A floating ❔ Help assistant (ui/faq.py) is available on every page.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="RAG∞ Ultimate", page_icon="♾️", layout="wide", initial_sidebar_state="auto")

from config.settings import get_logger, setup_logging  # noqa: E402
from rag.embeddings import warm_up  # noqa: E402
from ui import pages, state  # noqa: E402
from ui.styles import CSS  # noqa: E402

setup_logging(state.base_settings().log_level, state.base_settings().log_format)
log = get_logger("app")


@st.cache_resource(show_spinner=False)
def _warm_up() -> bool:
    warm_up(state.base_settings())  # load local fallback models in the background
    if state.base_settings().otel_endpoint:
        from monitoring.tracing import configure
        configure(state.base_settings().otel_endpoint, "rag-ultimate-ui")
    state.sync_manager()  # start background sync (if Hugging Face storage is configured)
    return True


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.logo("assets/logo.png", size="large", icon_image="assets/icon.png")
    if not state.password_ok():
        return
    _warm_up()
    state.workspace()  # pick up / create the private workspace from the link
    from security.auth import has_permission
    from ui import admin_pages

    who = state.principal()
    from ui.guide import guide_page

    nav = {
        "guide": st.Page(guide_page, title="Guide & Test Lab", icon="🎓", url_path="guide"),
        "chat": st.Page(pages.chat_page, title="Chat", icon="💬", url_path="chat", default=True),
        "library": st.Page(pages.library_page, title="Documents", icon="📚", url_path="library"),
        "validate": st.Page(pages.validate_page, title="Validate", icon="✅", url_path="validate"),
        "dashboard": st.Page(admin_pages.dashboard_page, title="Dashboard", icon="🏠", url_path="dashboard"),
        "ingestion": st.Page(admin_pages.ingestion_page, title="Ingestion", icon="📥", url_path="ingestion"),
        "retrieval": st.Page(admin_pages.retrieval_page, title="Retrieval debugger", icon="🔎", url_path="retrieval"),
        "traces": st.Page(admin_pages.traces_page, title="Agent traces", icon="🧭", url_path="traces"),
        "evaluation": st.Page(admin_pages.evaluation_page, title="Evaluation", icon="🧪", url_path="evaluation"),
        "insights": st.Page(pages.insights_page, title="Analytics", icon="📊", url_path="insights"),
        "settings": st.Page(pages.settings_page, title="Settings", icon="⚙️", url_path="settings"),
    }
    sections = {"Start here": ["guide"], "Workspace": ["chat", "library", "validate"],
                "Operations": ["dashboard", "ingestion", "retrieval", "traces", "evaluation", "insights"],
                "Administration": ["settings"]}
    for key, fn, title, icon, perm in [("models", admin_pages.models_page, "Models", "🤖", "manage_models"),
                                       ("security", admin_pages.security_page, "Security", "🛡️", "manage_policy"),
                                       ("users", admin_pages.users_page, "Users & roles", "👥", "manage_users"),
                                       ("audit", admin_pages.audit_page, "Audit log", "📜", "view_audit"),
                                       ("cache", admin_pages.cache_page, "Cache & help", "⚡", "manage_policy"),
                                       ("routing", admin_pages.routing_page, "Model routing", "🔀", "manage_models")]:
        if has_permission(who, perm):  # admin pages are not even listed for other roles
            nav[key] = st.Page(fn, title=title, icon=icon, url_path=key)
            sections["Administration"].append(key)
    st.session_state["pages"] = nav
    current = st.navigation({sec: [nav[k] for k in keys] for sec, keys in sections.items()})
    pages.sidebar()
    current.run()
    from ui import faq  # floating ❔ Help assistant on every page (deterministic FAQ, no AI call)

    key = next((k for k, pg in nav.items() if pg is current or getattr(pg, "url_path", None) ==
                getattr(current, "url_path", "")), "")
    faq.launcher(key, state.settings().data_dir)


try:
    main()
except Exception:  # last safety net: log details, show a plain message (never a stack trace)
    log.exception("Unhandled error in the app")
    st.error("Something went wrong. Please try again; details are in the server logs.")
