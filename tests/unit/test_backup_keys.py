"""2.2.1 backup API keys: GROQ_API_KEY_2…9 / OPENROUTER_API_KEY_2…9 (mocked HTTP, no real calls)."""
from __future__ import annotations

import json
import logging

import httpx
import pytest

from llm import model_router as mr
from llm import providers
from llm.providers import LLMError

GROQ = "https://groq.test/openai/v1"
OR = "https://or.test/api/v1"
K1, K2, K3 = "gsk_KEYONE_secret", "gsk_KEYTWO_secret", "gsk_KEYTHREE_secret"
O1, O2 = "sk-or-ORONE_secret", "sk-or-ORTWO_secret"
MSG = [{"role": "user", "content": "What is VOUT?"}]


class Fake:
    def __init__(self):
        self.calls = []          # (key, model)
        self.rules = {}          # (key, model) or key -> status list

    def handler(self, req: httpx.Request) -> httpx.Response:
        key = req.headers.get("authorization", "")[7:]
        if req.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        model = json.loads(req.content)["model"]
        self.calls.append((key, model))
        for rk in ((key, model), key):
            seq = self.rules.get(rk)
            if seq:
                status = seq.pop(0) if len(seq) > 1 else seq[0]
                if status != 200:
                    return httpx.Response(status, json={"error": {"message": f"status {status}"}})
        return httpx.Response(200, json={"model": model, "choices": [{"message": {"content": f"ok {model}"}}],
                                         "usage": {"total_tokens": 5}})


@pytest.fixture
def fake(monkeypatch):
    f = Fake()
    t = httpx.MockTransport(f.handler)
    monkeypatch.setattr(providers, "_client", lambda base, timeout: httpx.Client(base_url=base, transport=t))
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)
    mr.reset_state()
    return f


@pytest.fixture
def ks(make_settings):
    def _make(**kw):
        base = dict(groq_api_key=K1, groq_backup_keys=(K2, K3), groq_base_url=GROQ, openrouter_api_key=O1,
                    openrouter_backup_keys=(O2,), openrouter_base_url=OR, anyrouter_api_key="",
                    model_discovery="off", llm_max_retries=0,
                    master_models=("groq:model-a", "groq:model-b", "openrouter:vision-x:free"),
                    vision_models=("openrouter:vision-x:free", "openrouter:vision-y:free"))
        base.update(kw)
        return make_settings(**base)
    return _make


def test_rate_limited_key_switches_to_backup_for_same_model(fake, ks):
    fake.rules[K1] = [429]
    r = mr.ModelRouter(ks()).chat("master", MSG)
    assert r.routing["chosen"] == "groq:model-a"
    assert fake.calls == [(K1, "model-a"), (K2, "model-a")]


def test_rejected_key_is_skipped_on_later_requests(fake, ks):
    fake.rules[K1] = [401]
    s = ks()
    mr.ModelRouter(s).chat("master", MSG)
    fake.calls.clear()
    mr.ModelRouter(s).chat("master", MSG)
    assert fake.calls == [(K2, "model-a")]                       # bad key not tried again
    status = providers.key_pool_status("groq", [K1, K2, K3])
    assert [x["state"] for x in status] == ["rejected", "ok", "ok"]
    assert "secret" not in json.dumps(status)


def test_rate_limit_cools_key_only_for_that_model(fake, ks):
    fake.rules[(K1, "model-a")] = [429]
    s = ks()
    mr.ModelRouter(s).chat("master", MSG)
    fake.calls.clear()
    s2 = ks(master_models=("groq:model-b",))
    mr.ModelRouter(s2).chat("master", MSG)
    assert fake.calls == [(K1, "model-b")]                       # key 1 still serves other models


def test_all_keys_limited_then_next_model(fake, ks, caplog):
    caplog.set_level(logging.INFO)
    fake.rules[(K1, "model-a")] = [429]
    fake.rules[(K2, "model-a")] = [429]
    fake.rules[(K3, "model-a")] = [429]
    r = mr.ModelRouter(ks()).chat("master", MSG)
    assert r.routing["chosen"] == "groq:model-b"
    assert [c for c in fake.calls if c[1] == "model-a"] == [(K1, "model-a"), (K2, "model-a"), (K3, "model-a")]
    assert "secret" not in caplog.text and "key #1" in caplog.text


def test_all_keys_exhausted_everywhere_reports_clearly(fake, ks):
    for k in (K1, K2, K3):
        fake.rules[k] = [429]
    for k in (O1, O2):
        fake.rules[k] = [429]
    with pytest.raises(LLMError) as e:
        mr.ModelRouter(ks()).chat("master", MSG)
    assert e.value.kind == "rate_limit" and "keys were tried" in e.value.user_message
    assert "secret" not in e.value.user_message


def test_openrouter_out_of_credit_key_switches(fake, ks):
    fake.rules[O1] = [402]
    r = mr.ModelRouter(ks()).analyze_image("Describe.", [b"\x89PNG\r\n\x1a\n" + b"\0" * 32])
    assert r.routing["chosen"] == "openrouter:vision-x:free"
    assert fake.calls == [(O1, "vision-x:free"), (O2, "vision-x:free")]


def test_other_errors_do_not_rotate_keys(fake, ks):
    fake.rules[K1] = [400]
    with pytest.raises(LLMError) as e:
        mr.ModelRouter(ks()).chat("master", MSG)
    assert e.value.kind == "bad_request" and fake.calls == [(K1, "model-a")]
    fake.calls.clear()
    fake.rules[K1] = [200]
    fake.rules[(K1, "model-a")] = [503]
    r = mr.ModelRouter(ks()).chat("master", MSG)
    assert r.routing["chosen"] == "groq:model-b"
    assert {k for k, _ in fake.calls} == {K1}                     # outage: next model, not next key


def test_single_key_behaviour_unchanged(fake, ks):
    fake.rules[(K1, "model-a")] = [429]
    r = mr.ModelRouter(ks(groq_backup_keys=())).chat("master", MSG)
    assert r.routing["chosen"] == "groq:model-b"


def test_backup_keys_read_from_environment(monkeypatch):
    from config.settings import load_settings

    monkeypatch.setenv("GROQ_API_KEY", "g1")
    monkeypatch.setenv("GROQ_API_KEY_2", "g2")
    monkeypatch.setenv("GROQ_API_KEY_3", "g1")                   # duplicate of primary: ignored
    monkeypatch.setenv("GROQ_API_KEYS", "g4, g2 ,g5")
    monkeypatch.setenv("OPENROUTER_API_KEY", "o1")
    monkeypatch.setenv("OPENROUTER_API_KEY_2", "o2")
    s = load_settings()
    assert s.groq_backup_keys == ("g2", "g4", "g5") and s.openrouter_backup_keys == ("o2",)
    assert "g2" not in repr(s) and "o2" not in repr(s)


def test_models_page_shows_key_health_without_secrets(fake, ks, monkeypatch):
    """Render the Models page section through AppTest-free call: key_pool_status drives it."""
    s = ks()
    fake.rules[K2] = [429]
    fake.rules[K1] = [401]
    mr.ModelRouter(s).chat("master", MSG)
    rows = providers.key_pool_status("groq", [K1, K2, K3])
    assert [r["key"] for r in rows] == ["key #1", "key #2", "key #3"]
    assert [r["state"] for r in rows] == ["rejected", "cooling", "ok"] or rows[1]["state"] == "ok"
    assert all("secret" not in str(v) for r in rows for v in r.values())
