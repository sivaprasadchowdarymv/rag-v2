"""
Offline fake of the OpenRouter API (/api/v1) and the Groq API (/openai/v1), for tests only.

    POST /openai/v1/chat/completions  Groq: strict parameter whitelist (400 on OpenRouter-only fields),
                                      images are an error (text must stay on Groq, images go to OpenRouter)
    GET  /openai/v1/models

    POST /api/v1/chat/completions   scripted answers, strict tool-call protocol checks, vision replies
    POST /api/v1/embeddings         deterministic hashing vectors (honours `dimensions`)
    POST /api/v1/rerank             word-overlap relevance in 0..1
    GET  /api/v1/key | /api/v1/models

Control: GET /_ctl?fail=<model>[,<model>]|all|none  (429 for those models),
         GET /_ctl?verifier=reject                   (verifier says "not grounded"),
         GET /_log, GET /_reset.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

STATE = {"fail": [], "verifier": "ok", "requests": [], "errors": [], "embed_calls": 0, "rerank_calls": 0}
KEY = "sk-or-test"
GROQ_KEY = "gsk_test"
GROQ_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
GROQ_FIELDS = {"model", "messages", "temperature", "max_completion_tokens", "max_tokens", "tools", "tool_choice",
               "reasoning_effort", "include_reasoning", "reasoning_format", "response_format", "stream", "top_p",
               "stop", "seed"}
MODELS = ["qwen/qwen3.8-27b", "qwen/qwen3.8-flash", "qwen/qwen3.8-max", "google/gemma-4-31b-it",
          "google/gemma-4-31b-it:free", "google/gemma-4-26b-a4b-it", "google/gemma-4-26b-a4b-it:free",
          "openai/gpt-5-mini"]
EMBED_MODELS = ["qwen/qwen3-embedding-8b"]
OLLAMA_MODELS = ["llama3.1:8b", "qwen2.5:3b", "llava:7b", "nomic-embed-text"]
OPENAI_KEY, OPENAI_MODELS = "sk-openai-test", ["gpt-4o-mini", "gpt-4.1"]
ANTHROPIC_KEY, ANTHROPIC_MODELS = "sk-ant-test", ["claude-haiku-4-5", "claude-sonnet-4-5"]
RERANK_MODELS = ["qwen/qwen3-reranker-8b"]
LABEL = re.compile(r"\[REF-\d+: [A-Z]+ p\d+\]")
OCR_TEXT = ("SCANNED APPENDIX\nThermal resistance RthJA = 65 °C/W for the TO-220 package.\n"
            "Maximum junction temperature TJ = 150°C")
FIG_TEXT = "Line graph: output voltage (V, 0-6) versus input voltage (V, 0-35); output rises then regulates at 5 V."
_STOP = {"what", "is", "the", "a", "an", "of", "in", "to", "and", "or", "for", "on", "at", "how", "do", "i"}


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def _has_image(messages) -> bool:
    return any(isinstance(m.get("content"), list) and any(p.get("type") == "image_url" for p in m["content"])
               for m in messages)


def scripted(messages, tools_on):
    """Return (content, tool_calls[list of (name, args)]) like a cooperative model."""
    text = "\n".join(_text(m.get("content")) for m in messages)
    if _has_image(messages):
        if "Attached images" in text:  # a figure QUESTION answered by looking at the image
            fig = re.search(r"Attached images \(in order\): \S+ (\[REF-\d+: FIGURE p\d+\])", text)
            return (f"**Answer:** The graph shows the output voltage rising with input voltage and regulating "
                    f"at about 5.1 V above roughly 7 V input {fig.group(1) if fig else ''}."), []
        return (OCR_TEXT if "Transcribe" in text else FIG_TEXT), []
    if "strict fact-checker" in text:
        if STATE["verifier"] == "reject" and "VERIFICATION FEEDBACK" not in text:
            return json.dumps({"grounded": False, "unsupported_claims": ["maximum current the regulator"]}), []
        return json.dumps({"grounded": True, "unsupported_claims": []}), []
    if "abbreviation" in text and "number only" in text:
        return "1", []
    user = next((_text(m["content"]) for m in messages if m["role"] == "user"), "")
    q = (re.search(r"Question:\s*(.*)", user) or re.search(r"(.*)", user)).group(1).lower()
    labels = LABEL.findall(text)
    n_tool = sum(1 for m in messages if m["role"] == "tool")
    if ("dissipation" in q or "calculate" in q) and tools_on and n_tool == 0:
        return "", [("search_datasheet", {"query": "dropout voltage", "source": "equation"}),
                    ("calculate", {"expression": "(12-5)*1.5"})]
    if "loop" in q and tools_on:
        return "", [("read_page", {"page": 1})]
    if "nothing" in q or not labels:
        return "NOT FOUND IN DOCUMENT", []
    a, b = labels[0], labels[min(1, len(labels) - 1)]
    if "unsupported number" in q and "VERIFICATION FEEDBACK" not in text:
        return f"**Answer:** The output current is 9.87 A {a}.", []
    if "unsupported number" in q:  # corrected answer: no invented value
        return f"**Answer:** The document lists the output current conditions {a}.", []
    if "invent" in q:
        return "**Answer:** It is 3.3 V [REF-99: TEXT p1].", []
    if "thermal resistance" in q:
        lab = next((x for x in labels if "TEXT" in x), a)
        return f"**Answer:** The thermal resistance RthJA is 65 °C/W {lab}.", []
    if n_tool:  # answer the calculation question from the tool observations
        return (f"**Answer:** The power dissipation is 10.5 W {a}.\n\n**Equations:**\n\n"
                "$$P_D = (V_{IN} - V_{OUT}) \\times I_{OUT}$$\n\n"
                f"Using the calculator: $P_D = (12-5) \\times 1.5 = 10.5$ W {a}"), []
    return (f"**Answer:** The peak output current is 2.2 A {a}.\n\n"
            f"**Explanation:** This is the maximum current the regulator delivers {b}.\n\n"
            "**Equations:**\n\n$$P_D = (V_{IN} - V_{OUT}) \\times I_{OUT}$$\n\nwhere $P_D$ is the power dissipation "
            + a), []


def _vec(text: str, dim: int):
    v = [0.0] * dim
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
    return v


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj, headers=None):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _err(self, code, message):
        self._send(code, {"error": {"code": code, "message": message}})

    def _groq_auth(self):
        if self.headers.get("Authorization") != f"Bearer {GROQ_KEY}":
            self._send(401, {"error": {"message": "Invalid API Key", "type": "invalid_request_error",
                                       "code": "invalid_api_key"}})
            return False
        return True

    def _auth(self):
        if self.headers.get("Authorization") != f"Bearer {KEY}":
            self._err(401, "No auth credentials found")
            return False
        if not self.headers.get("X-Title") or not self.headers.get("HTTP-Referer"):
            STATE["errors"].append("missing app attribution headers")
        return True

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/_ctl":
            qs = parse_qs(url.query)
            if "fail" in qs:
                v = qs["fail"][0]
                STATE["fail"] = [] if v == "none" else v.split(",")
            if "verifier" in qs:
                STATE["verifier"] = qs["verifier"][0]
            return self._send(200, {"fail": STATE["fail"]})
        if url.path == "/_log":
            return self._send(200, STATE)
        if url.path == "/_reset":
            STATE.update(fail=[], verifier="ok", embed_calls=0, rerank_calls=0)
            STATE["requests"].clear(); STATE["errors"].clear()
            return self._send(200, {"ok": True})
        if url.path == "/ollama/api/tags":
            return self._send(200, {"models": [{"name": m, "model": m} for m in OLLAMA_MODELS]})
        if url.path == "/oai/v1/models":
            if self.headers.get("Authorization") != f"Bearer {OPENAI_KEY}":
                return self._send(401, {"error": {"message": "bad key"}})
            return self._send(200, {"data": [{"id": m} for m in OPENAI_MODELS]})
        if url.path == "/openai/v1/models":
            if not self._groq_auth():
                return
            return self._send(200, {"object": "list", "data": [{"id": m, "object": "model"}
                                                               for m in GROQ_MODELS + ["whisper-large-v3"]]})
        if not self._auth():
            return
        if url.path == "/api/v1/key":
            return self._send(200, {"data": {"label": "sk-or-v1-tes...", "usage": 0.0123, "limit": 5.0,
                                             "limit_remaining": 4.9877, "is_free_tier": False}})
        if url.path == "/api/v1/models":
            return self._send(200, {"data": [{"id": m} for m in MODELS + EMBED_MODELS + RERANK_MODELS]})
        self._err(404, "Not Found")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/ollama/api/chat":
            return self._ollama_chat(body)
        if self.path == "/ollama/api/embed":
            return self._ollama_embed(body)
        if self.path == "/oai/v1/chat/completions":
            return self._openai(body)
        if self.path == "/anthropic/v1/messages":
            return self._anthropic(body)
        if self.path == "/openai/v1/chat/completions":
            if not self._groq_auth():
                return
            return self._groq(body)
        if not self._auth():
            return
        if self.path == "/api/v1/chat/completions":
            return self._chat(body)
        if self.path == "/api/v1/embeddings":
            return self._embeddings(body)
        if self.path == "/api/v1/rerank":
            return self._rerank(body)
        self._err(404, "Not Found")

    def _blocked(self, model):
        return "all" in STATE["fail"] or model in STATE["fail"]

    def _chat(self, body):
        model = body.get("model", "")
        STATE["requests"].append(("chat", model))
        if self._blocked(model):
            return self._err(429, f"{model} is temporarily rate-limited upstream")
        if model not in MODELS:
            return self._err(404, f"No endpoints found for {model}.")
        if "reasoning" not in body:
            STATE["errors"].append("reasoning control not sent")
        if body.get("tools") and not (body.get("provider") or {}).get("require_parameters"):
            STATE["errors"].append("tools sent without provider.require_parameters")
        pending = set()
        for m in body["messages"]:
            if m["role"] == "assistant" and m.get("tool_calls"):
                pending |= {t["id"] for t in m["tool_calls"]}
            elif m["role"] == "tool":
                if m["tool_call_id"] not in pending:
                    STATE["errors"].append("tool reply without call")
                pending.discard(m["tool_call_id"])
            elif pending:
                STATE["errors"].append("message before tool calls answered")
        tools_on = bool(body.get("tools")) and body.get("tool_choice") != "none"
        content, calls = scripted(body["messages"], tools_on)
        msg = {"role": "assistant", "content": content or None}
        if calls:
            msg["tool_calls"] = [{"id": f"call_{i}_{len(STATE['requests'])}", "type": "function",
                                  "function": {"name": n, "arguments": json.dumps(a)}} for i, (n, a) in enumerate(calls)]
        prompt_tokens = len(json.dumps(body["messages"])) // 4
        price = 0.0 if model.endswith(":free") else 0.000001
        self._send(200, {"id": "gen-x", "object": "chat.completion", "created": 0, "model": model,
                         "provider": "FakeUpstream",
                         "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if calls else "stop"}],
                         "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 100,
                                   "total_tokens": prompt_tokens + 100,
                                   "cost": round((prompt_tokens + 100) * price, 8)}})

    def _groq(self, body):
        model = body.get("model", "")
        STATE["requests"].append(("groq", model))
        extra = set(body) - GROQ_FIELDS
        if extra:
            STATE["errors"].append(f"groq: unsupported fields {sorted(extra)}")
            return self._send(400, {"error": {"message": f"property '{sorted(extra)[0]}' is unsupported",
                                              "type": "invalid_request_error"}})
        if _has_image(body["messages"]):
            STATE["errors"].append("groq: image sent to a text model")
        if "all" in STATE["fail"] or f"groq:{model}" in STATE["fail"]:
            return self._send(429, {"error": {"message": "Rate limit reached", "type": "tokens",
                                              "code": "rate_limit_exceeded"}})
        if model not in GROQ_MODELS:
            return self._send(404, {"error": {"message": f"The model `{model}` does not exist",
                                              "code": "model_not_found"}})
        if model.startswith("openai/gpt-oss") and body.get("include_reasoning") is not False:
            STATE["errors"].append("groq: reasoning not hidden")
        pending = set()
        for m in body["messages"]:
            if m["role"] == "assistant" and m.get("tool_calls"):
                pending |= {t["id"] for t in m["tool_calls"]}
            elif m["role"] == "tool":
                pending.discard(m["tool_call_id"])
            elif pending:
                STATE["errors"].append("groq: message before tool calls answered")
        tools_on = bool(body.get("tools")) and body.get("tool_choice") != "none"
        content, calls = scripted(body["messages"], tools_on)
        msg = {"role": "assistant", "content": content or None}
        if calls:
            msg["tool_calls"] = [{"id": f"call_g{i}_{len(STATE['requests'])}", "type": "function",
                                  "function": {"name": n, "arguments": json.dumps(a)}} for i, (n, a) in enumerate(calls)]
        tokens = len(json.dumps(body["messages"])) // 4
        self._send(200, {"id": "chatcmpl-x", "object": "chat.completion", "created": 0, "model": model,
                         "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if calls else "stop"}],
                         "usage": {"prompt_tokens": tokens, "completion_tokens": 100, "total_tokens": tokens + 100}})

    def _embeddings(self, body):
        model = body.get("model", "")
        STATE["embed_calls"] += 1
        STATE["requests"].append(("embed", model))
        if self._blocked(model):
            return self._err(429, "rate limited")
        if model not in EMBED_MODELS:
            return self._err(404, f"No endpoints found for {model}.")
        texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
        dim = int(body.get("dimensions") or 4096)
        data = [{"object": "embedding", "index": i, "embedding": _vec(t.replace("Instruct:", ""), dim)}
                for i, t in enumerate(texts)]
        tokens = sum(len(t) // 4 for t in texts)
        self._send(200, {"object": "list", "model": model, "data": data,
                         "usage": {"prompt_tokens": tokens, "total_tokens": tokens, "cost": tokens * 1e-8}})

    def _rerank(self, body):
        model = body.get("model", "")
        STATE["rerank_calls"] += 1
        if self._blocked(model):
            return self._err(429, "rate limited")
        if model not in RERANK_MODELS:
            return self._err(404, f"No endpoints found for {model}.")
        q = set(re.findall(r"[a-z0-9]+", body["query"].lower())) - _STOP
        results = []
        for i, d in enumerate(body["documents"]):
            words = set(re.findall(r"[a-z0-9]+", d.lower()))
            results.append({"index": i, "relevance_score": len(q & words) / (len(q) or 1) - 0.0001 * len(words),
                            "document": {"text": d}})
        results.sort(key=lambda r: -r["relevance_score"])
        self._send(200, {"id": "rr-x", "model": model, "provider": "FakeUpstream",
                         "results": results[: body.get("top_n") or len(results)],
                         "usage": {"cost": 0.00001, "search_units": 1, "total_tokens": 50}})


def _flt(provider, model):
    return "all" in STATE["fail"] or f"{provider}:{model}" in STATE["fail"]


def _ollama_chat(self, body):
    model = body.get("model", "")
    STATE["requests"].append(("ollama", model))
    if body.get("stream") is not False:
        STATE["errors"].append("ollama: stream must be false")
    if _flt("ollama", model):
        return self._send(503, {"error": "model is loading"})
    if model not in OLLAMA_MODELS:
        return self._send(404, {"error": f"model '{model}' not found, try pulling it first"})
    msgs = []
    for m in body["messages"]:
        if not isinstance(m.get("content", ""), str):
            STATE["errors"].append("ollama: content must be a string")
        if m["role"] == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                if not isinstance(tc["function"].get("arguments"), dict):
                    STATE["errors"].append("ollama: tool arguments must be an object")
        mm = dict(m)
        if m.get("images"):
            mm["content"] = [{"type": "text", "text": m["content"]}, {"type": "image_url"}]
        msgs.append(mm)
    tools_on = bool(body.get("tools"))
    content, calls = scripted(msgs, tools_on)
    msg = {"role": "assistant", "content": content or ""}
    if calls:
        msg["tool_calls"] = [{"function": {"name": n, "arguments": a}} for n, a in calls]
    self._send(200, {"model": model, "message": msg, "done": True, "prompt_eval_count": 120, "eval_count": 80})


def _ollama_embed(self, body):
    model = body.get("model", "")
    STATE["embed_calls"] += 1
    if model not in OLLAMA_MODELS:
        return self._send(404, {"error": f"model '{model}' not found"})
    texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
    self._send(200, {"model": model, "embeddings": [_vec(t, 256) for t in texts]})


def _openai(self, body):
    if self.headers.get("Authorization") != f"Bearer {OPENAI_KEY}":
        return self._send(401, {"error": {"message": "Incorrect API key provided", "code": "invalid_api_key"}})
    model = body.get("model", "")
    STATE["requests"].append(("openai", model))
    bad = {"provider", "reasoning", "include_reasoning", "reasoning_effort"} & set(body)
    if bad:
        STATE["errors"].append(f"openai: unsupported fields {sorted(bad)}")
        return self._send(400, {"error": {"message": f"Unrecognized request argument: {sorted(bad)[0]}"}})
    if _flt("openai", model):
        return self._send(429, {"error": {"message": "Rate limit", "code": "rate_limit_exceeded"}},
                          headers={"retry-after": "1"})
    if model not in OPENAI_MODELS:
        return self._send(404, {"error": {"message": "model not found", "code": "model_not_found"}})
    tools_on = bool(body.get("tools")) and body.get("tool_choice") != "none"
    content, calls = scripted(body["messages"], tools_on)
    msg = {"role": "assistant", "content": content or None}
    if calls:
        msg["tool_calls"] = [{"id": f"call_o{i}", "type": "function",
                              "function": {"name": n, "arguments": json.dumps(a)}} for i, (n, a) in enumerate(calls)]
    self._send(200, {"model": model, "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100}})


def _anthropic(self, body):
    if self.headers.get("x-api-key") != ANTHROPIC_KEY:
        return self._send(401, {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}})
    if not self.headers.get("anthropic-version"):
        STATE["errors"].append("anthropic: missing anthropic-version header")
    model = body.get("model", "")
    STATE["requests"].append(("anthropic", model))
    if "max_tokens" not in body:
        STATE["errors"].append("anthropic: max_tokens required")
    roles = [m["role"] for m in body["messages"]]
    if any(r not in ("user", "assistant") for r in roles) or roles[:1] != ["user"]:
        STATE["errors"].append(f"anthropic: bad roles {roles}")
    if any(a == b for a, b in zip(roles, roles[1:])):
        STATE["errors"].append("anthropic: roles must alternate")
    pending = set()
    conv = [{"role": "system", "content": body.get("system", "")}]
    for m in body["messages"]:
        blocks = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
        text, has_img, n_res = [], False, 0
        for b in blocks:
            if b["type"] == "text":
                text.append(b["text"])
            elif b["type"] == "image":
                has_img = b["source"]["type"] == "base64"
            elif b["type"] == "tool_use":
                pending.add(b["id"])
            elif b["type"] == "tool_result":
                if b["tool_use_id"] not in pending:
                    STATE["errors"].append("anthropic: tool_result without tool_use")
                pending.discard(b["tool_use_id"])
                n_res += 1
        content = [{"type": "text", "text": " ".join(text)}, {"type": "image_url"}] if has_img else " ".join(text)
        conv.append({"role": m["role"], "content": content})
        conv += [{"role": "tool", "content": ""}] * n_res
    if _flt("anthropic", model):
        return self._send(529, {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})
    if model not in ANTHROPIC_MODELS:
        return self._send(404, {"type": "error", "error": {"type": "not_found_error", "message": "model"}})
    tools_on = bool(body.get("tools")) and (body.get("tool_choice") or {}).get("type") != "none"
    content, calls = scripted(conv, tools_on)
    out = ([{"type": "text", "text": content}] if content else []) + [
        {"type": "tool_use", "id": f"toolu_{i}", "name": n, "input": a} for i, (n, a) in enumerate(calls)]
    self._send(200, {"id": "msg_x", "type": "message", "role": "assistant", "model": model, "content": out,
                     "stop_reason": "tool_use" if calls else "end_turn",
                     "usage": {"input_tokens": 900, "output_tokens": 90}})


H._ollama_chat, H._ollama_embed, H._openai, H._anthropic = _ollama_chat, _ollama_embed, _openai, _anthropic

if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
