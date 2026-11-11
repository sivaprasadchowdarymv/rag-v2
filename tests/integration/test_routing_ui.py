"""2.2 Model routing admin page (mocked providers; no real API calls)."""
from __future__ import annotations

import os

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from tests.integration.test_ui_upgrade import _reset_process_caches
from tests.unit.test_free_model_routing import ANY, GROQ, FakeAPI

PW = "Adm1nPassw0rd"


@pytest.fixture(scope="module")
def routing_env(tmp_path_factory):
    d = tmp_path_factory.mktemp("routing_ui")
    env = {"DATA_DIR": str(d), "AUTH_MODE": "users", "ADMIN_USERNAME": "admin", "ADMIN_PASSWORD": PW,
           "GROQ_API_KEY": "gsk_UIKEY_SECRET", "GROQ_BASE_URL": GROQ, "ANYROUTER_API_KEY": "sk-ar-UIKEY_SECRET",
           "ANYROUTER_BASE_URL": ANY, "OPENROUTER_API_KEY": "", "MODEL_MAX_RETRIES": "0"}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    _reset_process_caches()
    from config.settings import load_settings
    from security.auth import AuthService
    s = load_settings()
    AuthService(s).bootstrap_admin()
    AuthService(s).create_user("viewer9", PW, ["viewer"], "default")
    yield s
    _reset_process_caches()
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture
def mocked(monkeypatch):
    from llm import discovery, providers
    from llm import model_router as mr

    fake = FakeAPI()
    transport = httpx.MockTransport(fake.handler)
    monkeypatch.setattr(providers, "_client", lambda base, timeout: httpx.Client(base_url=base, transport=transport))
    discovery.reset()
    mr.reset_state()
    return fake


def _page(user="admin"):
    at = AppTest.from_file("../ui_harness.py", default_timeout=90)
    at.query_params["page"] = "routing"
    at.query_params["user"] = user
    return at


def _all_text(at) -> str:
    return " ".join([m.value for m in at.markdown] + [c.value for c in at.caption]
                    + [str(df.value.to_dict()) for df in at.dataframe])


def test_routing_dashboard_shows_both_chains(routing_env, mocked):
    at = _page().run()
    assert not at.exception, [e.value for e in at.exception]
    text = _all_text(at)
    for needle in ("Groq primary", "Groq fallback 3", "AnyRouter text fallback 2", "AnyRouter vision fallback 3",
                   "acme/vision-1", "openai/gpt-oss-120b", "Catalogue refreshed", "Cost guard"):
        assert needle in text, needle
    assert "SECRET" not in text


def test_connection_and_vision_test_buttons(routing_env, mocked):
    at = _page().run()
    at.button(key="rt_conn").click().run()
    assert not at.exception
    ok = " ".join(s.value for s in at.success)
    assert "Groq: connected" in ok and "AnyRouter: connected" in ok and "SECRET" not in ok
    at = _page().run()
    at.button(key="rt_vis_run").click().run()
    assert not at.exception
    assert any("acme/vision-1 answered" in s.value for s in at.success)
    assert any(c["image"] and c["model"] == "acme/vision-1" for c in mocked.calls)


def test_viewer_cannot_open_routing(routing_env, mocked):
    at = _page("viewer9").run()
    assert not at.exception and any("permission" in e.value for e in at.error)


def test_page_explains_when_discovery_is_off(routing_env, mocked, monkeypatch):
    monkeypatch.setenv("MODEL_DISCOVERY", "off")
    _reset_process_caches()
    at = _page().run()
    assert not at.exception and "routing is off" in _all_text(at)
    monkeypatch.delenv("MODEL_DISCOVERY")
    _reset_process_caches()
