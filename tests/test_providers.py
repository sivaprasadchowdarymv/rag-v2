"""Provider layer: Groq for text, OpenRouter (Qwen / Gemma) for images; chains, fallback, errors."""
import pytest

from llm.catalog import GEMMA_31B_FREE, GROQ_120B, GROQ_LLAMA70, QWEN_FLASH, chain_for, split
from llm.model_router import ModelRouter
from llm.providers import LLMError
from tests.conftest import control, server_log

MSG = [{"role": "user", "content": "Question: What is the peak output current?\n\n[REF-1: TEXT p2] peak 2.2 A"}]
PNG = b"\x89PNG\r\n\x1a\nfake"


def test_text_roles_are_groq_and_vision_is_qwen_gemma():
    for profile in ("free", "quality"):
        for role in ("master", "fast", "verifier"):
            assert split(chain_for(profile, role)[0])[0] == "groq"
        assert all(split(e)[0] == "openrouter" and ("gemma" in e or "qwen" in e) for e in chain_for(profile, "vision"))
    assert chain_for("free", "vision")[0] == GEMMA_31B_FREE and chain_for("quality", "vision")[0] == QWEN_FLASH
    assert chain_for("free", "master")[-1] == GEMMA_31B_FREE  # last-resort text fallback when Groq's quota is gone


def test_no_keys_give_clear_errors(make_settings):
    r = ModelRouter(make_settings(groq_api_key="", openrouter_api_key=""))
    assert not r.enabled
    with pytest.raises(LLMError) as err:
        r.chat("master", MSG)
    assert err.value.kind == "no_provider" and "GROQ_API_KEY" in err.value.user_message
    with pytest.raises(LLMError) as err:
        r.chat("vision", MSG, images=[PNG])
    assert "OPENROUTER_API_KEY" in err.value.user_message


def test_chains_skip_providers_without_key(make_settings):
    only_groq = ModelRouter(make_settings(openrouter_api_key=""))
    assert [p.name for p, _ in only_groq.chain("master")] == ["groq", "groq"] and not only_groq.has_role("vision")
    only_or = ModelRouter(make_settings(groq_api_key=""))
    assert [m for _, m in only_or.chain("master")] == ["google/gemma-4-31b-it:free"]


def test_text_goes_to_groq_with_clean_body(server, make_settings):
    out = ModelRouter(make_settings()).chat("master", MSG)
    assert "2.2 A" in out.result.content and out.result.provider == "groq"
    assert out.result.model == "openai/gpt-oss-120b" and out.result.tokens > 0 and out.result.cost == 0.0
    assert server_log(server)["errors"] == []  # no OpenRouter-only fields, reasoning hidden


def test_groq_rate_limit_falls_back_to_llama_then_gemma(server, make_settings):
    control(server, GROQ_120B)
    out = ModelRouter(make_settings()).chat("master", MSG)
    assert out.result.model == "llama-3.3-70b-versatile" and [a[2] for a in out.attempts] == ["rate_limit", "ok"]
    control(server, f"{GROQ_120B},{GROQ_LLAMA70}")
    out = ModelRouter(make_settings()).chat("master", MSG)
    assert out.result.provider == "openrouter" and out.result.model == "google/gemma-4-31b-it:free"


def test_whole_chain_failing_raises(server, make_settings):
    control(server, "all")
    with pytest.raises(LLMError) as err:
        ModelRouter(make_settings()).chat("master", MSG)
    assert err.value.kind == "rate_limit" and "No other model" in err.value.user_message


def test_unknown_model_falls_through(server, make_settings):
    s = make_settings(master_models=("groq:does-not-exist", GROQ_120B))
    out = ModelRouter(s).chat("master", MSG)
    assert out.result.model == "openai/gpt-oss-120b" and out.attempts[0][2] == "model"


def test_wrong_keys_are_reported_not_leaked(server, make_settings):
    with pytest.raises(LLMError) as err:
        ModelRouter(make_settings(groq_api_key="gsk_WRONG")).chat("master", MSG)
    assert err.value.kind == "auth" and "gsk_WRONG" not in err.value.user_message
    with pytest.raises(LLMError) as err:
        ModelRouter(make_settings(openrouter_api_key="sk-or-WRONG")).chat("vision", MSG, images=[PNG])
    assert err.value.kind == "auth" and "sk-or-WRONG" not in err.value.user_message


def test_tool_calling_roundtrip_on_groq(server, make_settings):
    r = ModelRouter(make_settings())
    q = [{"role": "user", "content": "Question: calculate the dissipation\n\n[REF-1: TEXT p3] PD"}]
    from rag.tools import TOOL_SCHEMAS
    first = r.chat("master", q, tools=TOOL_SCHEMAS).result
    assert first.provider == "groq" and [c.name for c in first.tool_calls] == ["search_datasheet", "calculate"]
    msgs = q + [{"role": "assistant", "tool_calls": [{"id": c.id, "type": "function",
                 "function": {"name": c.name, "arguments": c.arguments}} for c in first.tool_calls]}]
    msgs += [{"role": "tool", "tool_call_id": c.id, "content": "obs [REF-1: TEXT p3]"} for c in first.tool_calls]
    assert "P_D" in r.chat("master", msgs, tools=TOOL_SCHEMAS).result.content
    assert server_log(server)["errors"] == []


def test_images_go_to_openrouter_gemma(server, make_settings):
    out = ModelRouter(make_settings()).chat("vision", [{"role": "user", "content": "Describe this figure"}],
                                            images=[PNG])
    assert out.result.provider == "openrouter" and out.result.model == "google/gemma-4-31b-it:free"
    assert "output voltage" in out.result.content and server_log(server)["errors"] == []


def test_status_reports_both_providers(server, make_settings):
    st = {p.name: p for p in ModelRouter(make_settings()).status()}
    assert st["groq"].connected and "openai/gpt-oss-120b" in st["groq"].models
    assert not any("whisper" in m for m in st["groq"].models)
    assert st["openrouter"].connected and st["openrouter"].key.remaining == pytest.approx(4.9877)


def test_openrouter_embeddings_and_rerank_endpoints(server, make_settings):
    from llm.providers import OpenRouterProvider
    p = OpenRouterProvider(make_settings())
    assert len(p.embed("qwen/qwen3-embedding-8b", ["a b", "c"], dimensions=256).vectors[0]) == 256
    scores = p.rerank("qwen/qwen3-reranker-8b", "peak output current", ["thermal", "peak output current 2.2 A"])
    assert scores[1] > scores[0]
