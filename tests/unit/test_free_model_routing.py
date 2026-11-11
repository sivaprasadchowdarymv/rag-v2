"""2.2 free-model discovery routing: Groq ×4 + AnyRouter free text ×2 (text), AnyRouter free vision ×3 (vision).

Everything runs against mocked HTTP APIs (httpx.MockTransport) — no real provider is called, no credits used.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

import httpx
import pytest

from llm import discovery
from llm import model_router as mr
from llm.providers import LLMError

GROQ = "https://groq.test/openai/v1"
ANY = "https://any.test/api/v1"
OR = "https://or.test/api/v1"
PNG = (b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
JPG = b"\xff\xd8\xff\xe0" + b"\x00" * 64

GROQ_MODELS = [
    {"id": "openai/gpt-oss-120b", "active": True, "context_window": 131072},
    {"id": "llama-3.3-70b-versatile", "active": True, "context_window": 131072},
    {"id": "openai/gpt-oss-20b", "active": True, "context_window": 131072},
    {"id": "qwen/qwen3-32b", "active": True, "context_window": 131072},
    {"id": "llama-3.1-8b-instant", "active": True, "context_window": 131072},
    {"id": "whisper-large-v3", "active": True},                       # speech: excluded
    {"id": "meta-llama/llama-guard-4-12b", "active": True},           # safety classifier: excluded
    {"id": "groq/compound", "active": True},                          # agentic web system: excluded
    {"id": "old-llama-2-70b", "active": False, "context_window": 4096},  # deprecated: excluded
]


def _free(mid, *, vision=False, access="free", ctx=128000, tools=True, **extra):
    m = {"id": mid, "access": access, "can_route": True, "context_length": ctx,
         "pricing": {"prompt": "0", "completion": "0"},
         "architecture": {"input_modalities": ["text", "image"] if vision else ["text"],
                          "output_modalities": ["text"]},
         "supported_parameters": ["tools", "temperature"] if tools else ["temperature"]}
    m.update(extra)
    return m


ANY_MODELS = [
    _free("acme/text-a"), _free("acme/text-b", access="Pool", ctx=64000),
    _free("acme/vision-1", vision=True, ctx=262000), _free("acme/vision-2", vision=True, ctx=200000),
    _free("acme/vision-3", vision=True, ctx=100000),
    _free("acme/paid-x", access="$0.30 · $1.00", pricing={"prompt": "0.0000003", "completion": "0.000001"}),
    _free("acme/paid-zero-label", access="free", pricing={"prompt": "0", "completion": "0.000002"}),
    _free("acme/byok-y", access="BYOK"),                                     # $0 gateway fee, but BYOK
    _free("acme/unknown-price", pricing=None),
    _free("acme/half-price", pricing={"prompt": "0"}),                       # output price missing
    _free("acme/no-route", can_route=False),
    _free("acme/route-unknown", can_route=None),
    _free("acme/retired-vision", vision=True, status="deprecated"),
    {"id": "anyrouter/free", "access": "free", "can_route": True, "pricing": {"prompt": "0", "completion": "0"},
     "members": [{"id": "acme/text-a"}]},                                    # virtual alias
    _free("acme/claims-nothing", vision=False, tools=False, capabilities=["reasoning"]),  # free text, no vision
]


def _or(mid, *, vision=False, prompt="0", completion="0", ctx=131072):
    return {"id": mid, "context_length": ctx, "pricing": {"prompt": prompt, "completion": completion},
            "architecture": {"input_modalities": ["text", "image"] if vision else ["text"],
                             "output_modalities": ["text"]}, "supported_parameters": ["tools"]}


OR_MODELS = [
    _or("google/gemma-x:free", vision=True, ctx=262144), _or("qwen/qwen-vl-y:free", vision=True, ctx=131072),
    _or("meta/llama-vision-z:free", vision=True, ctx=65536), _or("deepseek/text-t:free"),
    _or("mistral/text-u:free", ctx=32768),
    _or("openai/gpt-paid", vision=True, prompt="0.000002", completion="0.000008"),     # paid
    _or("weird/free-but-priced:free", vision=True, prompt="0.000001"),                 # ':free' id but priced
    _or("openrouter/free"),                                                            # router alias, no ':free'
]


class FakeAPI:
    """Scripted Groq + AnyRouter. `script[model]` = list of (status, body|None) consumed per call."""

    def __init__(self):
        self.script: Dict[str, List] = {}
        self.calls: List[Dict[str, Any]] = []
        self.model_list_calls = {"groq": 0, "any": 0, "or": 0}
        self.any_models = list(ANY_MODELS)
        self.groq_models = list(GROQ_MODELS)
        self.or_models = list(OR_MODELS)
        self.fail_lists = set()
        self.cost: Dict[str, float] = {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        host = "groq" if "groq.test" in url else "or" if "or.test" in url else "any"
        auth = request.headers.get("authorization", "")
        if request.method == "GET" and request.url.path.endswith("/models"):
            self.model_list_calls[host] += 1
            if host in self.fail_lists:
                return httpx.Response(503, json={"error": {"message": "down"}})
            return httpx.Response(200, json={"data": {"groq": self.groq_models, "or": self.or_models,
                                                      "any": self.any_models}[host]})
        body = json.loads(request.content)
        model = body["model"]
        has_image = any(isinstance(m.get("content"), list) and any(p.get("type") == "image_url" for p in m["content"])
                        for m in body["messages"])
        self.calls.append({"host": host, "model": model, "image": has_image, "auth": auth})
        steps = self.script.get(model) or []
        status, payload = steps.pop(0) if steps else (200, None)
        if status != 200:
            return httpx.Response(status, json=payload or {"error": {"message": f"status {status}"}})
        text = payload or f"Answer from {model} [REF-1: TABLE p2]."
        return httpx.Response(200, json={"model": model, "choices": [{"message": {"content": text},
                                                                        "finish_reason": "stop"}],
                                         "usage": {"total_tokens": 42, "cost": self.cost.get(model, 0)}})

    def models_called(self, host=None):
        return [c["model"] for c in self.calls if host is None or c["host"] == host]


@pytest.fixture
def api(monkeypatch):
    fake = FakeAPI()
    transport = httpx.MockTransport(fake.handler)
    from llm import providers

    monkeypatch.setattr(providers, "_client", lambda base, timeout: httpx.Client(base_url=base, transport=transport))
    monkeypatch.setattr(mr, "_sleep", lambda s: None)
    discovery.reset()
    mr.reset_state()
    return fake


@pytest.fixture
def rs(make_settings):
    """Discovery settings: Groq + AnyRouter mocked; OpenRouter off; no retries unless a test asks."""
    def _make(**kw):
        base = dict(groq_api_key="gsk_SECRETKEY123", groq_base_url=GROQ, anyrouter_api_key="sk-ar-SECRETKEY456",
                    anyrouter_base_url=ANY, openrouter_api_key="", model_discovery="on", model_max_retries=0,
                    circuit_failures=100)
        base.update(kw)
        return make_settings(**base)
    return _make


MSG = [{"role": "user", "content": "What is the output voltage?"}]
TEXT_CHAIN = ["groq:openai/gpt-oss-120b", "groq:llama-3.3-70b-versatile", "groq:openai/gpt-oss-20b",
              "groq:qwen/qwen3-32b", "anyrouter:acme/text-a", "anyrouter:acme/text-b"]
VISION_CHAIN = ["anyrouter:acme/vision-1", "anyrouter:acme/vision-2", "anyrouter:acme/vision-3"]


def _down(api, *models, status=503):
    for m in models:
        api.script[m.split(":", 1)[1]] = [(status, None)] * 10


# ----------------------------------------------------------------- selection
def test_discovered_chains_have_required_order_and_are_independent(api, rs):
    s = rs()
    assert list(s.models_for("master")) == TEXT_CHAIN
    assert list(s.models_for("vision")) == VISION_CHAIN
    assert not set(s.models_for("vision")) & {e for e in TEXT_CHAIN if e.startswith("groq:")}
    sel = discovery.select(s)
    assert [lbl for lbl, _ in sel.text] == ["Groq primary", "Groq fallback 1", "Groq fallback 2", "Groq fallback 3",
                                            "AnyRouter text fallback 1", "AnyRouter text fallback 2"]
    assert not sel.shortages


# ----------------------------------------------------------------- 1-6 text fallbacks
@pytest.mark.parametrize("failing,expected", [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)],
                         ids=["groq-primary", "groq-fallback-1", "groq-fallback-2", "groq-fallback-3",
                              "anyrouter-text-1", "anyrouter-text-2"])
def test_text_chain_falls_back_in_order(api, rs, failing, expected):
    s = rs()
    _down(api, *TEXT_CHAIN[:failing])
    r = mr.ModelRouter(s).generate_text(MSG)
    assert r.routing["chosen"] == TEXT_CHAIN[expected]
    assert [a[0] + ":" + a[1] for a in r.attempts] == TEXT_CHAIN[:expected + 1]
    assert not any(c["image"] for c in api.calls)            # text pipeline never carries images
    assert set(api.models_called()) <= {e.split(":", 1)[1] for e in TEXT_CHAIN}


# ----------------------------------------------------------------- 7-9 vision fallbacks
@pytest.mark.parametrize("failing", [0, 1, 2], ids=["vision-1", "vision-2", "vision-3"])
def test_vision_chain_falls_back_in_order(api, rs, failing):
    s = rs()
    _down(api, *VISION_CHAIN[:failing])
    r = mr.ModelRouter(s).analyze_image("Describe the figure.", [PNG])
    assert r.routing["chosen"] == VISION_CHAIN[failing]
    assert all(c["host"] == "any" and c["image"] for c in api.calls)   # only vision models, always with the image
    assert not api.models_called("groq")


# ----------------------------------------------------------------- 10-11 everything down
def test_all_text_models_unavailable(api, rs):
    s = rs()
    _down(api, *TEXT_CHAIN)
    with pytest.raises(LLMError) as e:
        mr.ModelRouter(s).generate_text(MSG)
    log = e.value.routing["attempt_log"]
    assert [f"{r['provider']}:{r['model']}" for r in log] == TEXT_CHAIN
    assert all(r["category"] == "unavailable" for r in log) and "No other model" in e.value.user_message


def test_all_vision_models_unavailable_keeps_text_answer(api, rs):
    s = rs()
    _down(api, *VISION_CHAIN)
    router = mr.ModelRouter(s)
    with pytest.raises(LLMError):
        router.analyze_image("Describe.", [PNG])
    out = router.generate_multimodal_response("What does Figure 2 show?", "[REF-1: FIGURE p4] block diagram",
                                              images=[PNG])
    assert out["visual_interpretation"] is False and out["pipeline"] == "text"
    assert out["result"].routing["chosen"] == TEXT_CHAIN[0]
    assert any("text evidence" in w for w in out["warnings"])
    text_calls = [c for c in api.calls if c["host"] == "groq"]
    assert text_calls and not any(c["image"] for c in text_calls)   # image never forwarded to a text model


# ----------------------------------------------------------------- 12-14 cost guard
@pytest.mark.parametrize("mid,reason", [
    ("acme/paid-x", "paid"), ("acme/paid-zero-label", "paid"),                  # 12 paid
    ("acme/byok-y", "BYOK"),                                                    # 13 BYOK with $0 pricing
    ("acme/unknown-price", "unknown pricing"), ("acme/half-price", "unknown pricing"),  # 14 unknown pricing
    ("acme/no-route", "cannot be routed"), ("acme/route-unknown", "can_route unknown"),
    ("anyrouter/free", "alias"), ("acme/retired-vision", "deprecated"),
])
def test_non_free_models_are_rejected(api, rs, mid, reason):
    s = rs()
    cat = discovery.catalog(s)
    assert all(m.id != mid for m in cat.any_text + cat.any_vision)
    assert reason in cat.rejected[f"anyrouter:{mid}"]
    assert discovery.verified_free(f"anyrouter:{mid}", s) is False


def test_paid_model_never_called_even_when_configured_explicitly(api, rs):
    s = rs(master_models=("anyrouter:acme/paid-x", "anyrouter:acme/byok-y", "groq:openai/gpt-oss-120b"))
    r = mr.ModelRouter(s).generate_text(MSG)
    assert r.routing["chosen"] == "groq:openai/gpt-oss-120b"
    assert "cost guard" in r.routing["excluded"]["anyrouter:acme/paid-x"]
    assert "cost guard" in r.routing["excluded"]["anyrouter:acme/byok-y"]
    assert "paid-x" not in api.models_called() and "byok-y" not in api.models_called()


def test_paid_providers_need_explicit_approval(api, rs):
    s = rs(openai_api_key="sk-openai", anthropic_api_key="sk-ant")
    assert not any(e.split(":")[0] in ("openai", "anthropic") for e in s.models_for("master"))
    approved = rs(openai_api_key="sk-openai", model_allow_paid_fallback=True, model_free_only=False)
    assert any(e.startswith("openai:") for e in approved.models_for("master"))
    still_free_only = rs(openai_api_key="sk-openai", model_allow_paid_fallback=True)  # free-only wins
    assert not any(e.startswith("openai:") for e in still_free_only.models_for("master"))


def test_reported_cost_excludes_model_immediately(api, rs):
    s = rs()
    _down(api, *TEXT_CHAIN[:4])
    api.cost["acme/text-a"] = 0.0012
    r = mr.ModelRouter(s).generate_text(MSG)
    assert r.routing["chosen"] == "anyrouter:acme/text-a" and "cost_alert" in r.routing
    assert "anyrouter:acme/text-a" not in s.models_for("master")       # replaced from now on
    assert discovery.verified_free("anyrouter:acme/text-a", s) is False


def test_402_credits_request_marks_model_paid(api, rs):
    s = rs()
    _down(api, *TEXT_CHAIN[:4])
    api.script["acme/text-a"] = [(402, {"error": {"code": "insufficient_balance", "message": "no balance"}})]
    r = mr.ModelRouter(s).generate_text(MSG)
    assert r.routing["chosen"] == "anyrouter:acme/text-b"
    assert "credits" in discovery.excluded("anyrouter:acme/text-a")


# ----------------------------------------------------------------- 15 vision capability
def test_vision_capability_is_verified(api, rs):
    s = rs()
    cat = discovery.catalog(s)
    assert {m.id for m in cat.any_vision} == {"acme/vision-1", "acme/vision-2", "acme/vision-3"}
    assert "acme/claims-nothing" in {m.id for m in cat.any_text}
    assert "acme/claims-nothing" not in {m.id for m in cat.any_vision}
    router = mr.ModelRouter(s)
    with pytest.raises(LLMError) as e:
        router.chat("master", MSG, images=[PNG])                    # images never reach the text chain
    assert e.value.kind == "unsupported" and not api.calls
    s2 = rs(vision_models=("anyrouter:acme/claims-nothing", "anyrouter:acme/vision-1"))
    r = mr.ModelRouter(s2).analyze_image("Describe.", [PNG])
    assert r.routing["chosen"] == "anyrouter:acme/vision-1"
    assert r.routing["excluded"]["anyrouter:acme/claims-nothing"] == "no verified image input"


def test_model_rejecting_image_format_is_replaced(api, rs):
    s = rs()
    api.script["acme/vision-1"] = [(400, {"error": {"message": "image input is not supported by this model"}})]
    r = mr.ModelRouter(s).analyze_image("Describe.", [JPG])
    assert r.routing["chosen"] == "anyrouter:acme/vision-2"
    assert "image" in discovery.excluded("anyrouter:acme/vision-1")
    assert discovery.select(s).vision[0][1] == "anyrouter:acme/vision-2"   # slot refilled automatically


# ----------------------------------------------------------------- 16 image validation
@pytest.mark.parametrize("img,msg", [(b"", "empty"), (b"%PDF-1.7 not an image", "not a PNG"),
                                     (b"\x89PNG\r\n\x1a\n" + b"\x00" * (2 * 1024 * 1024), "larger than"),
                                     ("not-bytes", "empty or not binary")])
def test_image_encoding_errors_fail_before_any_call(api, rs, img, msg):
    s = rs(model_max_image_mb=1.0)
    with pytest.raises(LLMError) as e:
        mr.ModelRouter(s).analyze_image("Describe.", [img])
    assert e.value.kind == "invalid_image" and msg in e.value.user_message
    assert not api.calls


def test_supported_image_types():
    webp = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 16
    assert len(mr.validate_images([PNG, JPG, b"GIF89a" + b"\x00" * 8, webp], 1.0)) == 4


# ----------------------------------------------------------------- 17 429 retries
def test_429_is_retried_with_backoff_then_succeeds(api, rs, monkeypatch):
    waits = []
    monkeypatch.setattr(mr, "_sleep", lambda s: waits.append(s))
    s = rs(model_max_retries=2)
    api.script["openai/gpt-oss-120b"] = [(429, {"error": {"code": "rate_limit_exceeded"}})] * 2
    r = mr.ModelRouter(s).generate_text(MSG)
    assert r.routing["chosen"] == TEXT_CHAIN[0]
    assert [x["result"] for x in r.routing["attempt_log"]] == ["retry", "retry", "success"]
    assert waits == [0.5, 1.0]                                        # exponential backoff


def test_429_retries_are_bounded_then_next_model(api, rs):
    s = rs(model_max_retries=2)
    _down(api, TEXT_CHAIN[0], status=429)
    r = mr.ModelRouter(s).generate_text(MSG)
    assert r.routing["chosen"] == TEXT_CHAIN[1]
    assert api.models_called().count("openai/gpt-oss-120b") == 3      # 1 + MODEL_MAX_RETRIES


# ----------------------------------------------------------------- 18 invalid keys
def test_invalid_groq_key_skips_groq_and_uses_anyrouter_without_retry(api, rs, caplog):
    caplog.set_level(logging.INFO)
    s = rs(model_max_retries=2)
    for m in TEXT_CHAIN[:4]:
        api.script[m.split(":", 1)[1]] = [(401, {"error": {"message": "Invalid API Key"}})] * 5
    r = mr.ModelRouter(s).generate_text(MSG)
    assert r.routing["chosen"] == "anyrouter:acme/text-a"
    assert api.models_called("groq") == ["openai/gpt-oss-120b"]       # one call: no retries, no other Groq model
    assert "SECRETKEY" not in caplog.text and "SECRETKEY" not in json.dumps(r.routing)


def test_invalid_anyrouter_key_for_vision(api, rs):
    s = rs(model_max_retries=2)
    for m in VISION_CHAIN:
        api.script[m.split(":", 1)[1]] = [(401, None)] * 5
    with pytest.raises(LLMError) as e:
        mr.ModelRouter(s).analyze_image("Describe.", [PNG])
    assert e.value.kind == "auth" and len(api.calls) == 1 and "SECRETKEY" not in e.value.user_message


def test_free_tier_entitlement_missing_is_reported(api, rs):
    s = rs()
    for m in VISION_CHAIN:
        api.script[m.split(":", 1)[1]] = [(403, {"error": {"code": "free_tier_requires_subscription"}})]
    with pytest.raises(LLMError) as e:
        mr.ModelRouter(s).analyze_image("Describe.", [PNG])
    assert e.value.kind == "entitlement" and "eligible plan" in e.value.user_message


def test_bad_request_and_moderation_stop_immediately(api, rs):
    s = rs()
    api.script["openai/gpt-oss-120b"] = [(400, {"error": {"message": "invalid messages"}})]
    with pytest.raises(LLMError) as e:
        mr.ModelRouter(s).generate_text(MSG)
    assert e.value.kind == "bad_request" and len(api.calls) == 1


# ----------------------------------------------------------------- catalogue management
def test_catalogue_is_cached_not_refreshed_per_query(api, rs):
    s = rs()
    router = mr.ModelRouter(s)
    for _ in range(5):
        router.generate_text(MSG)
    assert api.model_list_calls == {"groq": 1, "any": 1, "or": 0}
    assert (s.data_dir / "model_catalog.json").exists()


def test_stale_selection_is_revalidated_and_fails_closed(api, rs):
    s = rs()
    discovery.catalog(s)
    discovery._CATALOG.refreshed_at -= 3600                  # older than MODEL_CATALOG_REFRESH_MINUTES
    discovery._CATALOG.anyrouter_validated_at -= 3600
    api.fail_lists.add("any")                                # AnyRouter cannot be re-validated now
    chain = s.models_for("master")
    assert api.model_list_calls["any"] >= 2                  # re-validated before use (1 transport retry on 503)
    assert not any(e.startswith("anyrouter:") for e in chain) and chain[0] == TEXT_CHAIN[0]
    assert s.models_for("vision") == ()


def test_repeatedly_failing_model_is_replaced(api, rs):
    s = rs(model_failure_exclude=2)
    _down(api, TEXT_CHAIN[0])
    router = mr.ModelRouter(s)
    router.generate_text(MSG)
    router.generate_text(MSG)
    assert "failed 2 times" in discovery.excluded(TEXT_CHAIN[0])
    chain = s.models_for("master")
    assert chain[0] == TEXT_CHAIN[1] and "groq:llama-3.1-8b-instant" in chain   # next ranked Groq model moved up


def test_shortage_is_reported_not_invented(api, rs):
    api.any_models = [_free("acme/vision-1", vision=True)]
    s = rs()
    sel = discovery.select(s)
    assert [e for _, e in sel.vision] == ["anyrouter:acme/vision-1"]
    assert any("1 of 3" in x for x in sel.shortages)


def test_discovery_off_keeps_legacy_routing(make_settings):
    s = make_settings(anyrouter_api_key="")
    assert not s.discovery_enabled
    assert s.models_for("master")[0].startswith("groq:")
    assert mr.ModelRouter(s).discovery is False


# ----------------------------------------------------------------- 19 cache isolation
def test_answer_cache_isolated_between_users_and_documents(api, rs, indexed, sample_pdf):
    from agents.rag_pro import answer_pro
    from ops import answer_cache

    index, _ = indexed
    s = rs(data_dir=index.figures_dir.parents[2])
    q = "What is the maximum output current?"
    first = answer_pro(q, [index], s, mr.ModelRouter(s), cache_scope="acme|u=1")
    assert first.kind == "answer"
    assert answer_pro(q, [index], s, mr.ModelRouter(s), cache_scope="acme|u=2").cached is False   # other user
    import dataclasses
    other_doc = dataclasses.replace(index, doc_id="0" * 16)
    assert answer_pro(q, [other_doc], s, mr.ModelRouter(s), cache_scope="acme|u=1").cached is False  # other doc
    c = answer_cache.AnswerCache(10, 60)
    c.store("t1|docA", q, "A", ["docA"], tenant="t1")
    assert c.lookup("t1|docB", q)[0] is None and c.lookup("t2|docA", q)[0] is None


# ----------------------------------------------------------------- 20 multimodal citations
def test_multimodal_citations_preserved_when_vision_fails(api, rs, indexed):
    from agents.rag_pro import answer_pro

    index, _ = indexed
    s = rs(data_dir=index.figures_dir.parents[2], image_answers=True)
    _down(api, *VISION_CHAIN)
    figs_before = sorted(p.name for p in index.figures_dir.iterdir())
    r = answer_pro("What does the functional block diagram figure show?", [index], s, mr.ModelRouter(s))
    assert r.kind in ("answer", "not_found")
    assert any("figure images could not be read" in w for w in r.warnings)   # no false visual claim
    assert all(isinstance(src.page, int) and src.page > 0 and src.doc_name == "LM7805X.pdf" for src in r.sources)
    assert sorted(p.name for p in index.figures_dir.iterdir()) == figs_before   # originals kept
    groq_calls = [c for c in api.calls if c["host"] == "groq"]
    assert groq_calls and not any(c["image"] for c in groq_calls)


def test_multimodal_answer_uses_vision_chain_and_keeps_figure_page(api, rs, indexed):
    from agents.rag_pro import answer_pro

    index, _ = indexed
    s = rs(data_dir=index.figures_dir.parents[2], image_answers=True)
    r = answer_pro("What does the functional block diagram figure show?", [index], s, mr.ModelRouter(s))
    vision_calls = [c for c in api.calls if c["image"]]
    assert vision_calls and all(c["host"] == "any" and c["model"] == "acme/vision-1" for c in vision_calls)
    figs = [e for e in r.retrieval_debug["selected"] if e["type"] == "figure"]
    assert figs and all(e["page"] in (3, 4) and e["doc"] == "LM7805X.pdf" for e in figs)   # figure page kept
    assert any("image(s) sent to the vision model" in st.text for st in r.steps)
    assert r.sources and all(src.page and src.doc_name == "LM7805X.pdf" for src in r.sources)
    assert r.provider == "anyrouter" and r.model == "acme/vision-1"


# ----------------------------------------------------------------- 2.2.1 OpenRouter + chain placeholders
OR_VISION = ["openrouter:google/gemma-x:free", "openrouter:qwen/qwen-vl-y:free", "openrouter:meta/llama-vision-z:free"]


@pytest.fixture
def rs_or(rs):
    def _make(**kw):
        base = dict(openrouter_api_key="sk-or-SECRET789", openrouter_base_url=OR,
                    master_models=("@groq", "@anyrouter_text", "@openrouter_text"),
                    fast_models=("@groq", "@anyrouter_text", "@openrouter_text"),
                    verifier_models=("@groq", "@anyrouter_text", "@openrouter_text"),
                    vision_models=("@openrouter_vision", "@anyrouter_vision"))
        base.update(kw)
        return rs(**base)
    return _make


def test_placeholders_build_combined_chains(api, rs_or):
    s = rs_or()
    assert list(s.models_for("master")) == TEXT_CHAIN + ["openrouter:deepseek/text-t:free",
                                                         "openrouter:mistral/text-u:free"]
    assert list(s.models_for("vision")) == OR_VISION + VISION_CHAIN
    assert not any(e.startswith("openai") for e in s.models_for("master"))


def test_openrouter_non_free_models_rejected(api, rs_or):
    s = rs_or()
    cat = discovery.catalog(s)
    ids = {m.id for m in cat.or_text + cat.or_vision}
    assert "openai/gpt-paid" not in ids and "openrouter/free" not in ids
    assert "paid" in cat.rejected["openrouter:weird/free-but-priced:free"]


def test_vision_falls_from_openrouter_to_anyrouter(api, rs_or):
    s = rs_or()
    _down(api, *OR_VISION)
    r = mr.ModelRouter(s).analyze_image("Describe.", [PNG])
    assert r.routing["chosen"] == VISION_CHAIN[0]
    hosts = [c["host"] for c in api.calls]
    assert set(hosts[:-1]) == {"or"} and hosts[-1] == "any" and all(c["image"] for c in api.calls)
    assert [m for m in dict.fromkeys(c["model"] for c in api.calls)] == [e.split(":", 1)[1] for e in OR_VISION] + [
        "acme/vision-1"]


def test_text_reaches_openrouter_after_groq_and_anyrouter(api, rs_or):
    s = rs_or()
    _down(api, *TEXT_CHAIN)
    r = mr.ModelRouter(s).generate_text(MSG)
    assert r.routing["chosen"] == "openrouter:deepseek/text-t:free"
    assert not any(c["image"] for c in api.calls)


def test_placeholders_mixed_with_explicit_entries_and_unknown_dropped(api, rs_or):
    s = rs_or(vision_models=("@openrouter_vision", "@nonsense", "ollama:llava"),
              ollama_base_url="http://ollama.test", ollama_vision_model="llava")
    assert list(s.models_for("vision")) == OR_VISION + ["ollama:llava"]


def test_placeholders_ignored_when_discovery_off(make_settings):
    s = make_settings(anyrouter_api_key="", master_models=("@groq", "groq:llama-3.3-70b-versatile"))
    assert s.models_for("master") == ("groq:llama-3.3-70b-versatile",)


def test_openrouter_free_rows_on_dashboard(api, rs_or):
    s = rs_or()
    summ = discovery.summary(s)
    slots = [r["slot"] for r in summ["vision"]]
    assert slots[:3] == ["OpenRouter free vision 1", "OpenRouter free vision 2", "OpenRouter free vision 3"]
    assert slots[3:] == ["AnyRouter vision fallback 1", "AnyRouter vision fallback 2", "AnyRouter vision fallback 3"]
    assert all(r["free verified"] for r in summ["vision"])
