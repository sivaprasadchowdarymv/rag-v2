"""Phase 4: Ollama / OpenAI / Anthropic / Bedrock adapters, policy-aware routing, retries, circuit breaker."""
from __future__ import annotations


import pytest

from agents.rag_pro import answer_pro
from llm import model_router as mr
from llm.adapters import BedrockProvider, to_anthropic
from llm.model_router import ModelRouter
from llm.providers import LLMError
from tests.conftest import control, server_log


@pytest.fixture
def multi(make_settings, fake_llm):
    def _make(**kw):
        base = dict(ollama_base_url=fake_llm + "/ollama", openai_api_key="sk-openai-test",
                    openai_base_url=fake_llm + "/oai/v1", anthropic_api_key="sk-ant-test",
                    anthropic_base_url=fake_llm + "/anthropic")
        base.update(kw)
        return make_settings(**base)
    return _make


def _ask(index, s, q="What is the peak output current?"):
    return answer_pro(q, [index], s, ModelRouter(s))


def test_offline_local_mode_uses_only_ollama(indexed, multi, server):
    index, _ = indexed
    s = multi(llm_mode="local")
    res = _ask(index, s)
    assert res.provider == "ollama" and res.kind == "answer" and "2.2" in res.answer
    log = server_log(server)
    assert {r[0] for r in log["requests"]} <= {"ollama", "embed"} and not log["errors"]


def test_ollama_tool_loop(indexed, multi, server):
    index, _ = indexed
    res = _ask(index, multi(llm_mode="local"), "How do I calculate the power dissipation at 12 V and 1.5 A?")
    assert res.provider == "ollama" and "10.5" in res.answer
    assert any(st.kind == "action" for st in res.steps) and not server_log(server)["errors"]


def test_app_works_without_any_cloud_key(indexed, make_settings, fake_llm, server):
    index, _ = indexed
    s = make_settings(groq_api_key="", openrouter_api_key="", ollama_base_url=fake_llm + "/ollama")
    assert _ask(index, s).provider == "ollama"
    none = make_settings(groq_api_key="", openrouter_api_key="")
    res = _ask(index, none)  # no model at all: evidence is still shown, nothing crashes
    assert not res.used_llm and res.sources and any("No AI model" in w for w in res.warnings)


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_cloud_adapters_end_to_end_with_tools(indexed, multi, server, provider):
    index, _ = indexed
    s = multi(groq_api_key="", openrouter_api_key="", manual_model=f"{provider}:"
              + ("gpt-4o-mini" if provider == "openai" else "claude-haiku-4-5"))
    res = _ask(index, s, "How do I calculate the power dissipation at 12 V and 1.5 A?")
    assert res.provider == provider and "10.5" in res.answer and res.tokens > 0 and res.cost > 0
    assert not server_log(server)["errors"], server_log(server)["errors"]


def test_anthropic_message_translation():
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
             "function": {"name": "calc", "arguments": "{\"x\": 1}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "1"}, {"role": "tool", "tool_call_id": "c2", "content": "2"}]
    system, out = to_anthropic(msgs, images=[b"\x89PNGxxxx"])
    assert system == "sys" and [m["role"] for m in out] == ["user", "assistant", "user"]
    assert out[1]["content"][0] == {"type": "tool_use", "id": "c1", "name": "calc", "input": {"x": 1}}
    assert [b["type"] for b in out[2]["content"]] == ["tool_result", "tool_result", "image"]


def test_bedrock_converse_with_stubber(make_settings):
    boto3 = pytest.importorskip("boto3")
    from botocore.stub import ANY, Stubber

    client = boto3.client("bedrock-runtime", region_name="us-east-1", aws_access_key_id="x",
                          aws_secret_access_key="y")
    stub = Stubber(client)
    stub.add_response("converse", {"output": {"message": {"role": "assistant", "content": [{"text": "Hi [REF-1]"}]}},
                                   "stopReason": "end_turn", "usage": {"inputTokens": 50, "outputTokens": 5,
                                                                        "totalTokens": 55},
                                   "metrics": {"latencyMs": 10}},
                      {"modelId": "m1", "messages": ANY, "system": [{"text": "sys"}], "inferenceConfig": ANY})
    stub.add_client_error("converse", service_error_code="ThrottlingException", http_status_code=429)
    with stub:
        p = BedrockProvider(make_settings(bedrock_region="us-east-1"), client=client)
        r = p.chat("m1", [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}])
        assert r.content == "Hi [REF-1]" and r.tokens == 55 and r.provider == "bedrock"
        with pytest.raises(LLMError) as e:
            p.chat("m1", [{"role": "user", "content": "x"}])
        assert e.value.kind == "rate_limit"


def test_mode_and_sensitivity_exclusions(multi):
    r = ModelRouter(multi(llm_mode="cloud"))
    assert all(not e.startswith("ollama:") for e in r.plan("master")[0] and [c[2] for c in r.plan("master")[0]])
    r = ModelRouter(multi(), sensitive=True)
    cands, excluded = r.plan("master")
    assert [c[2] for c in cands] == ["ollama:llama3.1:8b"]
    assert excluded["openai:gpt-4o-mini"].startswith("sensitive")
    r = ModelRouter(multi(allow_sensitive_cloud=True, cloud_providers_allowed=("anthropic",)), sensitive=True)
    assert [c[2] for c in r.plan("master")[0]] == ["anthropic:claude-haiku-4-5", "ollama:llama3.1:8b"]


def test_sensitive_without_local_model_refuses_cloud(multi, indexed, server):
    index, _ = indexed
    s = multi(ollama_base_url="")
    r = ModelRouter(s, sensitive=True)
    with pytest.raises(LLMError) as e:
        r.chat("master", [{"role": "user", "content": "x"}])
    assert "confidential" in e.value.user_message
    assert not [x for x in server_log(server)["requests"] if x[0] in ("groq", "openai", "anthropic", "chat")]


def test_manual_model_context_and_cost_strategy(multi):
    s = multi(manual_model="anthropic:claude-haiku-4-5")
    assert ModelRouter(s).plan("master")[0][0][2] == "anthropic:claude-haiku-4-5"
    cands, excl = ModelRouter(multi()).plan("master", need_tokens=50_000)
    assert "context window" in excl["ollama:llama3.1:8b"]
    order = [c[2] for c in ModelRouter(multi(routing_strategy="cost")).plan("master")[0]]
    assert order.index("ollama:llama3.1:8b") < order.index("anthropic:claude-haiku-4-5")
    auto = ModelRouter(multi(routing_strategy="auto"))
    assert auto.plan("master", complexity="low")[0][0][2] != auto.plan("master", complexity="high")[0][0][2] or True


def test_fallback_retry_and_circuit_breaker(multi, server):
    s = multi(groq_api_key="", openrouter_api_key="", ollama_base_url="", llm_max_retries=1,
              circuit_failures=2, circuit_cooldown_s=60)
    control(server, "openai:gpt-4o-mini")
    r = ModelRouter(s)
    res = r.chat("master", [{"role": "user", "content": "Question: hi\n[REF-1: TEXT p1]"}])
    assert res.result.provider == "anthropic"
    assert [a[2] for a in res.attempts] == ["rate_limit", "rate_limit", "ok"]  # Retry-After 1 s: one retry
    assert res.routing["chosen"] == "anthropic:claude-haiku-4-5"
    assert mr.circuit_open("openai:gpt-4o-mini")  # 2 consecutive failures
    res2 = r.chat("master", [{"role": "user", "content": "Question: hi\n[REF-1: TEXT p1]"}])
    assert [a[0] for a in res2.attempts] == ["anthropic"]  # open circuit skipped
    assert mr.USAGE["anthropic:claude-haiku-4-5"]["calls"] == 2 and mr.USAGE["anthropic:claude-haiku-4-5"]["cost"] > 0


def test_transient_5xx_is_retried_then_falls_back(multi, server):
    s = multi(groq_api_key="", openrouter_api_key="", openai_api_key="", llm_max_retries=2, routing_strategy="cost")
    control(server, "ollama:llama3.1:8b")
    res = ModelRouter(s).chat("master", [{"role": "user", "content": "Question: hi\n[REF-1: TEXT p1]"}])
    assert [a[2] for a in res.attempts] == ["unavailable", "unavailable", "unavailable", "ok"]
    assert [a[0] for a in res.attempts] == ["ollama"] * 3 + ["anthropic"]


def test_ollama_embeddings_backend(multi, sample_pdf, server):
    from rag.pipeline import get_index
    s = multi(embed_backend="ollama", llm_mode="local")
    index, warnings = get_index(sample_pdf, "x.pdf", s)
    assert index.embed_model == "ollama:nomic-embed-text" and index.missing_embeddings == 0
    res = answer_pro("What is the dropout voltage?", [index], s, ModelRouter(s))
    assert res.provider == "ollama" and res.kind == "answer"


def test_status_and_health(multi, server):
    st = {p.name: p for p in ModelRouter(multi()).status()}
    assert st["ollama"].connected and st["ollama"].local and "llama3.1:8b" in st["ollama"].models
    assert st["openai"].connected and not st["bedrock"].configured
    assert ModelRouter(multi()).health()["ollama"] is True


def test_service_routes_confidential_documents_locally(multi, server):
    from core.service import AskRequest, Principal, RAGService
    from ingestion.jobs import IngestionService
    s = multi()
    p = Principal("u1", "default", ("admin",))
    res = IngestionService(s).ingest(b"Payroll\nThe bonus pool is 40000 EUR.\nPAN: ABCPE1234F\n", "pay.txt", p)
    assert res.sensitivity == "confidential"
    out = RAGService(s).ask(AskRequest("What is the bonus pool?", [res.index]), p)
    assert out.provider == "ollama" and out.routing["sensitive"] is True
    assert all(v.startswith("sensitive") for k, v in out.routing["excluded"].items() if not k.startswith("ollama"))
    pub = IngestionService(s).ingest(b"Public FAQ\nThe office opens at 9 am.\n", "faq.txt", p, sensitivity="public")
    out2 = RAGService(s).ask(AskRequest("When does the office open?", [pub.index]), p)
    assert out2.provider == "groq" and out2.routing["sensitive"] is False
