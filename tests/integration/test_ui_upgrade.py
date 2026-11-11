"""2.1 UI: new pages render, floating help assistant, onboarding, cache badge, answer summary, metadata editor,
library search, chat filters, role gating."""
from __future__ import annotations

import os

import pytest
from streamlit.testing.v1 import AppTest

PW = "Adm1nPassw0rd"


@pytest.fixture(scope="module")
def ui2_env(tmp_path_factory, fake_llm):
    d = tmp_path_factory.mktemp("ui2")
    env = {"DATA_DIR": str(d), "AUTH_MODE": "users", "ADMIN_USERNAME": "admin", "ADMIN_PASSWORD": PW,
           "GROQ_API_KEY": "gsk_test", "GROQ_BASE_URL": fake_llm + "/openai/v1", "OPENROUTER_API_KEY": "sk-or-test",
           "OPENROUTER_BASE_URL": fake_llm + "/api/v1", "LLM_MAX_RETRIES": "0", "RATE_LIMIT_PER_MIN": "100",
           "GLOBAL_RATE_LIMIT_PER_MIN": "1000"}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    _reset_process_caches()
    from config.settings import load_settings
    from ingestion.jobs import IngestionService
    from security.auth import AuthService
    from tests.conftest import make_datasheet
    s = load_settings()
    auth = AuthService(s)
    auth.bootstrap_admin()
    auth.create_user("viewer2", PW, ["viewer"], "default")
    auth.create_user("newbie", PW, ["editor"], "fresh")       # empty tenant: first-time experience
    admin = auth.authenticate("admin", PW)
    IngestionService(s).ingest(make_datasheet(), "LM7805X.pdf", admin)
    yield s
    _reset_process_caches()
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _reset_process_caches() -> None:
    """Streamlit resource caches (settings, sync) and DB repositories are per process: start clean so this module
    never reuses another module's DATA_DIR / database (and vice versa)."""
    import streamlit as st

    from database.repository import reset_repositories

    st.cache_resource.clear()
    st.cache_data.clear()
    reset_repositories()


def _page(name, user="admin"):
    at = AppTest.from_file("../ui_harness.py", default_timeout=90)
    at.query_params["page"] = name
    at.query_params["user"] = user
    return at


def _text(at) -> str:
    return " ".join(m.value for m in at.markdown)


def test_cache_page_renders_for_admin(ui2_env):
    at = _page("cache").run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Cached answers" in _text(at) and at.button(key="cache_clear")


def test_viewer_cannot_open_cache_page(ui2_env):
    at = _page("cache", user="viewer2").run()
    assert not at.exception and any("permission" in e.value for e in at.error)


@pytest.mark.parametrize("name", ["chat", "library", "dashboard", "guide", "retrieval", "settings"])
def test_help_launcher_on_every_page(ui2_env, name):
    at = _page(name).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.text_input(key="faq_q") is not None
    labels = " ".join(b.label for b in at.button)
    assert labels  # quick replies rendered


def test_faq_search_and_quick_reply(ui2_env):
    at = _page("library").run()
    at.text_input(key="faq_q").set_value("how do I upload a file").run()
    assert not at.exception
    assert "How do I upload a document?" in _text(at)
    # context-aware suggestions on the library page include an uploading question; clicking it shows the answer
    at = _page("library").run()
    at.button(key="faq_s_up-how").click().run()
    assert not at.exception and "Sensitivity" in _text(at)
    assert at.session_state["faq_sel"] == "up-how"


def test_faq_unknown_question_falls_back(ui2_env):
    at = _page("chat").run()
    at.text_input(key="faq_q").set_value("zzqx wibble").run()
    assert not at.exception and any("couldn't find" in i.value for i in at.info)


def test_first_time_user_sees_onboarding_and_empty_states(ui2_env):
    at = _page("chat", user="newbie").run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Get started in 4 steps" in _text(at) and "0 of 4 done" in _text(at)
    lib = _page("library", user="newbie").run()
    assert not lib.exception and "Your library is empty" in _text(lib)


def test_repeated_question_shows_cache_badge_and_summary(ui2_env, server):
    at = _page("chat").run()
    at.chat_input[0].set_value("What is the peak output current?").run()
    assert not at.exception, [e.value for e in at.exception]
    at = _page("chat").run()
    at.chat_input[0].set_value("what is the peak output current").run()
    assert not at.exception, [e.value for e in at.exception]
    text = _text(at)
    assert "From cache" in text and "2.2" in text
    assert any("How this answer was produced" in e.label for e in at.expander)
    assert "Documents searched" in text and "Verification" in text


def test_dashboard_shows_real_cache_and_quality_tiles(ui2_env):
    at = _page("dashboard").run()
    assert not at.exception
    text = _text(at)
    for label in ("Cache hit / miss", "Citation verification", "Indexed chunks", "Ingestion failures",
                  "Blocked requests", "Active users", "AI cost"):
        assert label in text, label


def test_library_metadata_view_search_and_edit(ui2_env):
    at = _page("library").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(e.label == "🏷️ Metadata" for e in at.expander)
    assert "Technical parameters" in _text(at)
    at.text_input(key="lib_search").set_value("no-such-thing-xyz").run()
    assert "No document matches" in _text(at)
    at = _page("library").run()
    tags = next(t for t in at.text_input if t.label == "Tags (comma separated)")
    tags.set_value("Power, Regulators")
    next(b for b in at.button if "Save metadata" in b.label).click().run()
    assert not at.exception, [e.value for e in at.exception]
    from storage.metadata_store import MetadataStore

    s = ui2_env
    doc_id = next(p.parent.name for p in (s.data_dir / "documents").glob("*/metadata_user.json"))
    assert MetadataStore(s).user(doc_id, "tenant:default")["tags"] == ["power", "regulators"]
    at = _page("library").run()
    at.multiselect(key="lib_tags").set_value(["power"]).run()
    assert not at.exception and "1 of 1 document(s)" in " ".join(c.value for c in at.caption)


def test_viewer_sees_metadata_but_cannot_edit(ui2_env):
    at = _page("library", user="viewer2").run()
    assert not at.exception
    assert not any("Save metadata" in b.label for b in at.button)
    assert any("needs the editor or admin role" in c.value for c in at.caption)


def test_chat_filters_apply_and_answer(ui2_env, server):
    at = _page("chat").run()
    at.selectbox(key="cf_cat").set_value("datasheet")
    at.multiselect(key="cf_types").set_value(["table", "row"])
    at.button(key="cf_apply").click().run()
    assert not at.exception
    assert at.session_state["chat_filter"]["category"] == "datasheet"
    at.chat_input[0].set_value("What is the quiescent current?").run()
    assert not at.exception, [e.value for e in at.exception]
    assert "filter(s) on" in _text(at)
    at.button(key="cf_clear").click().run()
    assert at.session_state["chat_filter"] == {}


def test_help_center_tab_in_guide(ui2_env):
    at = _page("guide").run()
    assert not at.exception
    assert at.text_input(key="hc_q") is not None
    at.text_input(key="hc_q").set_value("what is rag").run()
    assert "Retrieval-Augmented Generation" in _text(at)


def test_page_headers_explain_purpose(ui2_env):
    at = _page("retrieval").run()
    text = _text(at)
    assert "page-title" in " ".join(m.value for m in at.markdown) or "Retrieval debugger" in text
    assert "Why you need it" in text and "If something goes wrong" in text
