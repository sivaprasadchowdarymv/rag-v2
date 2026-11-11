"""AppTest harness: renders ONE page (query param ?page=) for a pre-authenticated principal."""
import streamlit as st

from security.auth import AuthService
from ui import admin_pages, guide, pages, state
from ui.styles import CSS

st.markdown(CSS, unsafe_allow_html=True)
if "principal" not in st.session_state:
    st.session_state["principal"] = AuthService(state.base_settings()).authenticate(
        st.query_params.get("user", "admin"), "Adm1nPassw0rd")
st.session_state.setdefault("pages", {"chat": None, "guide": None})
PAGES = {"guide": guide.guide_page, "chat": pages.chat_page, "library": pages.library_page, "validate": pages.validate_page,
         "insights": pages.insights_page, "settings": pages.settings_page, "dashboard": admin_pages.dashboard_page,
         "ingestion": admin_pages.ingestion_page, "retrieval": admin_pages.retrieval_page,
         "traces": admin_pages.traces_page, "evaluation": admin_pages.evaluation_page,
         "models": admin_pages.models_page, "security": admin_pages.security_page, "users": admin_pages.users_page,
         "audit": admin_pages.audit_page, "cache": admin_pages.cache_page,
         "routing": admin_pages.routing_page}
pages.sidebar()
PAGES[st.query_params.get("page", "chat")]()
from ui import faq  # noqa: E402  (rendered after the page, as in app.py)

faq.launcher(st.query_params.get("page", "chat"), state.settings().data_dir)
