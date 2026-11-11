"""
Additional LLM providers behind the same interface as llm/providers.py:

    OllamaProvider     LOCAL   native /api/chat, /api/tags, /api/embed     (no key; OLLAMA_BASE_URL)
    OpenAIProvider     cloud   /v1/chat/completions (OpenAI-compatible)     (OPENAI_API_KEY; any compatible URL)
    AnthropicProvider  cloud   /v1/messages                                  (ANTHROPIC_API_KEY)
    BedrockProvider    cloud   bedrock-runtime Converse API via boto3       (BEDROCK_REGION + AWS credential chain)

All accept the app's OpenAI-style messages (system/user/assistant/tool, tool_calls)
and translate them to each wire format; all return ChatResult with prompt/completion
tokens and a cost estimate from llm/pricing.py. Errors become LLMError with the
same `kind` values, so the router's fallback rules apply unchanged.
"""
from __future__ import annotations

import base64
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx

from config.settings import Settings, get_logger
from llm.pricing import estimate_cost
from llm.providers import ChatResult, EmbedResult, KeyInfo, LLMError, OpenRouterProvider, ToolCall, _client

log = get_logger("adapters")


def _retry_after(resp: httpx.Response) -> float:
    try:
        return float(resp.headers.get("retry-after") or 0)
    except ValueError:
        return 0.0


def _mime(data: bytes) -> str:
    return "image/jpeg" if data[:3] == b"\xff\xd8\xff" else "image/png"


def _kind_for(status: int) -> str:
    return {401: "auth", 403: "auth", 404: "model", 408: "timeout", 413: "bad_request", 429: "rate_limit",
            402: "quota"}.get(status, "unavailable" if status >= 500 else "bad_request")


class _HttpMixin:
    name = label = ""
    local = False

    def _post(self, base: str, path: str, body: Dict, headers: Dict[str, str], timeout: float,
              model: str) -> Dict[str, Any]:
        try:
            resp = _client(base, timeout).post(path, json=body, headers=headers)
        except httpx.TimeoutException as exc:
            raise LLMError(f"{self.label} took too long to respond.", "timeout", self.name) from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Cannot reach {self.label}.", "connection", self.name) from exc
        if resp.status_code >= 400:
            err = LLMError(f"{self.label} returned an error for '{model}' ({resp.status_code}).",
                           _kind_for(resp.status_code), self.name, resp.status_code)
            err.retry_after = _retry_after(resp)  # type: ignore[attr-defined]
            if resp.status_code == 400:
                log.warning("%s rejected a request for %s: %s", self.label, model, resp.text[:200])
            raise err
        try:
            return resp.json()
        except ValueError as exc:
            raise LLMError(f"{self.label} returned an unreadable response.", "api", self.name) from exc


# =============================================================================
# Ollama (local)
# =============================================================================
class OllamaProvider(_HttpMixin):
    name, label, local = "ollama", "Ollama (local)", True

    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = settings.ollama_base_url

    def configured(self) -> bool:
        return bool(self.base_url)

    @staticmethod
    def _messages(messages: List[Dict[str, Any]], images: Optional[Sequence[bytes]]) -> List[Dict[str, Any]]:
        out = []
        for m in messages:
            mm: Dict[str, Any] = {"role": m["role"], "content": m.get("content") or ""}
            if m.get("tool_calls"):
                mm["tool_calls"] = [{"function": {"name": t["function"]["name"],
                                                  "arguments": json.loads(t["function"].get("arguments") or "{}")}}
                                    for t in m["tool_calls"]]
            out.append(mm)
        if images:
            out[-1]["images"] = [base64.b64encode(b).decode() for b in images]
        return out

    def chat(self, model: str, messages: List[Dict[str, Any]], tools: Optional[List[Dict]] = None,
             tool_choice: str = "auto", temperature: float = 0.1, max_tokens: int = 1500,
             images: Optional[Sequence[bytes]] = None, reasoning: Optional[str] = None,
             json_mode: bool = False) -> ChatResult:
        body: Dict[str, Any] = {"model": model, "messages": self._messages(messages, images), "stream": False,
                                "options": {"temperature": temperature, "num_predict": max_tokens}}
        if tools and tool_choice != "none":
            body["tools"] = tools
        if json_mode:
            body["format"] = "json"
        data = self._post(self.base_url, "/api/chat", body, {}, max(self.settings.llm_timeout, 120.0), model)
        msg = data.get("message") or {}
        calls = [ToolCall(f"call_{i}", (tc.get("function") or {}).get("name", ""),
                          json.dumps((tc.get("function") or {}).get("arguments") or {}))
                 for i, tc in enumerate(msg.get("tool_calls") or [])]
        content = (msg.get("content") or "").strip()
        if not content and not calls:
            raise LLMError(f"'{model}' returned an empty answer.", "empty", self.name)
        pt, ct = int(data.get("prompt_eval_count") or 0), int(data.get("eval_count") or 0)
        return ChatResult(content, calls, pt + ct, self.name, model, 0.0, "local",
                          "tool_calls" if calls else "stop", prompt_tokens=pt, completion_tokens=ct)

    def embed(self, model: str, texts: Sequence[str], dimensions: Optional[int] = None) -> EmbedResult:
        data = self._post(self.base_url, "/api/embed", {"model": model, "input": list(texts)}, {}, 120.0, model)
        vecs = data.get("embeddings") or []
        if len(vecs) != len(texts):
            raise LLMError("Ollama returned an incomplete embedding result.", "api", self.name)
        return EmbedResult([[float(x) for x in v] for v in vecs])

    def list_models(self, families: Sequence[str] = ()) -> List[str]:
        try:
            resp = _client(self.base_url, 10.0).get("/api/tags")
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError("Ollama is not reachable.", "connection", self.name) from exc
        return sorted(m.get("name", "") for m in resp.json().get("models") or [] if m.get("name"))

    def key_info(self) -> KeyInfo:
        self.list_models()
        return KeyInfo(label="local", is_free_tier=True)


# =============================================================================
# OpenAI (and any OpenAI-compatible server: vLLM, LM Studio, Azure-compatible gateways)
# =============================================================================
class OpenAIProvider(OpenRouterProvider):
    name, label = "openai", "OpenAI"
    key_env, key_url = "OPENAI_API_KEY", "platform.openai.com/api-keys"
    local = False

    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = settings.openai_api_key
        self.base_url = settings.openai_base_url

    def _headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise LLMError("No OpenAI API key is set (OPENAI_API_KEY).", "auth", self.name)
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _body(self, model, messages, tools, tool_choice, temperature, max_tokens, reasoning, json_mode):
        body: Dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature,
                                "max_tokens": max_tokens}
        if tools:
            body.update(tools=tools, tool_choice=tool_choice)
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    @staticmethod
    def _translate(status: int, payload: Dict, model: str) -> LLMError:
        return LLMError(f"OpenAI returned an error for '{model}' ({status}).", _kind_for(status), "openai", status)

    def chat(self, model, messages, tools=None, tool_choice="auto", temperature=0.1, max_tokens=1500, images=None,
             reasoning=None, json_mode=False) -> ChatResult:
        res = super().chat(model, messages, tools, tool_choice, temperature, max_tokens, images, reasoning, json_mode)
        res.provider = self.name
        res.cost = estimate_cost(self.name, model, res.prompt_tokens, res.completion_tokens)
        return res

    def list_models(self, families: Sequence[str] = ()) -> List[str]:
        data = self._request("GET", "/models", None, timeout=20.0)
        return sorted(str(m.get("id")) for m in data.get("data") or [] if m.get("id"))

    def key_info(self) -> KeyInfo:
        self.list_models()
        return KeyInfo(label="openai")


# =============================================================================
# Anthropic Messages API
# =============================================================================
def to_anthropic(messages: List[Dict[str, Any]], images: Optional[Sequence[bytes]] = None
                 ) -> Tuple[str, List[Dict[str, Any]]]:
    """OpenAI-style messages -> (system, alternating user/assistant messages with content blocks)."""
    system = "\n\n".join(m.get("content") or "" for m in messages if m["role"] == "system")
    out: List[Dict[str, Any]] = []

    def push(role: str, blocks: List[Dict[str, Any]]) -> None:
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": blocks})

    for m in messages:
        role = m["role"]
        if role == "system":
            continue
        if role == "tool":
            push("user", [{"type": "tool_result", "tool_use_id": m.get("tool_call_id", ""),
                           "content": m.get("content") or ""}])
        elif role == "assistant":
            blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                blocks.append({"type": "tool_use", "id": tc.get("id", ""), "name": fn.get("name", ""),
                               "input": json.loads(fn.get("arguments") or "{}")})
            if blocks:
                push("assistant", blocks)
        else:
            push("user", [{"type": "text", "text": m.get("content") or ""}])
    if images and out:
        last_user = next(m for m in reversed(out) if m["role"] == "user")
        last_user["content"] += [{"type": "image", "source": {"type": "base64", "media_type": _mime(b),
                                                              "data": base64.b64encode(b).decode()}} for b in images]
    if not out or out[0]["role"] != "user":
        out.insert(0, {"role": "user", "content": [{"type": "text", "text": "(start)"}]})
    return system, out


def tools_to_anthropic(tools: Optional[List[Dict]]) -> List[Dict[str, Any]]:
    return [{"name": t["function"]["name"], "description": t["function"].get("description", ""),
             "input_schema": t["function"].get("parameters") or {"type": "object", "properties": {}}}
            for t in tools or []]


class AnthropicProvider(_HttpMixin):
    name, label = "anthropic", "Anthropic"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = settings.anthropic_api_key
        self.base_url = settings.anthropic_base_url

    def configured(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> Dict[str, str]:
        return {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}

    def chat(self, model: str, messages: List[Dict[str, Any]], tools: Optional[List[Dict]] = None,
             tool_choice: str = "auto", temperature: float = 0.1, max_tokens: int = 1500,
             images: Optional[Sequence[bytes]] = None, reasoning: Optional[str] = None,
             json_mode: bool = False) -> ChatResult:
        if not self.api_key:
            raise LLMError("No Anthropic API key is set (ANTHROPIC_API_KEY).", "auth", self.name)
        system, msgs = to_anthropic(messages, images)
        body: Dict[str, Any] = {"model": model, "max_tokens": max_tokens, "temperature": temperature,
                                "messages": msgs}
        if system:
            body["system"] = system
        if tools:
            body["tools"] = tools_to_anthropic(tools)
            body["tool_choice"] = {"type": "none" if tool_choice == "none" else "auto"}
        data = self._post(self.base_url, "/v1/messages", body, self._headers(), self.settings.llm_timeout, model)
        text = "".join(b.get("text", "") for b in data.get("content") or [] if b.get("type") == "text").strip()
        calls = [ToolCall(b.get("id", f"toolu_{i}"), b.get("name", ""), json.dumps(b.get("input") or {}))
                 for i, b in enumerate(data.get("content") or []) if b.get("type") == "tool_use"]
        if not text and not calls:
            raise LLMError(f"'{model}' returned an empty answer.", "empty", self.name)
        u = data.get("usage") or {}
        pt, ct = int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0)
        return ChatResult(text, calls, pt + ct, self.name, str(data.get("model") or model),
                          estimate_cost(self.name, model, pt, ct), "anthropic", str(data.get("stop_reason") or ""),
                          prompt_tokens=pt, completion_tokens=ct)

    def list_models(self, families: Sequence[str] = ()) -> List[str]:
        return [self.settings.anthropic_model]

    def key_info(self) -> KeyInfo:
        return KeyInfo(label="anthropic")


# =============================================================================
# AWS Bedrock (Converse API)
# =============================================================================
class BedrockProvider:
    name, label, local = "bedrock", "AWS Bedrock", False

    def __init__(self, settings: Settings, client=None):
        self.settings = settings
        self._client = client

    def configured(self) -> bool:
        if not self.settings.bedrock_region:
            return False
        try:
            import boto3  # noqa: F401
            return True
        except ImportError:
            return False

    def client(self):
        if self._client is None:
            import boto3
            from botocore.config import Config

            self._client = boto3.client("bedrock-runtime", region_name=self.settings.bedrock_region,
                                        config=Config(read_timeout=self.settings.llm_timeout,
                                                      retries={"max_attempts": 1}))
        return self._client

    @staticmethod
    def converse_request(model: str, messages, tools, tool_choice, temperature, max_tokens, images) -> Dict[str, Any]:
        system, msgs = to_anthropic(messages, images)
        conv = []
        for m in msgs:
            blocks = []
            for b in m["content"]:
                if b["type"] == "text":
                    blocks.append({"text": b["text"]})
                elif b["type"] == "image":
                    blocks.append({"image": {"format": "jpeg" if "jpeg" in b["source"]["media_type"] else "png",
                                             "source": {"bytes": base64.b64decode(b["source"]["data"])}}})
                elif b["type"] == "tool_use":
                    blocks.append({"toolUse": {"toolUseId": b["id"], "name": b["name"], "input": b["input"]}})
                elif b["type"] == "tool_result":
                    blocks.append({"toolResult": {"toolUseId": b["tool_use_id"],
                                                  "content": [{"text": b["content"]}]}})
            conv.append({"role": m["role"], "content": blocks})
        req: Dict[str, Any] = {"modelId": model, "messages": conv,
                               "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature}}
        if system:
            req["system"] = [{"text": system}]
        if tools and tool_choice != "none":
            req["toolConfig"] = {"tools": [{"toolSpec": {"name": t["name"], "description": t["description"],
                                                         "inputSchema": {"json": t["input_schema"]}}}
                                           for t in tools_to_anthropic(tools)]}
        return req

    def chat(self, model: str, messages: List[Dict[str, Any]], tools: Optional[List[Dict]] = None,
             tool_choice: str = "auto", temperature: float = 0.1, max_tokens: int = 1500,
             images: Optional[Sequence[bytes]] = None, reasoning: Optional[str] = None,
             json_mode: bool = False) -> ChatResult:
        req = self.converse_request(model, messages, tools, tool_choice, temperature, max_tokens, images)
        try:
            resp = self.client().converse(**req)
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code", "") if hasattr(exc, "response") else ""
            kind = {"ThrottlingException": "rate_limit", "ServiceUnavailableException": "unavailable",
                    "ModelTimeoutException": "timeout", "AccessDeniedException": "auth",
                    "ResourceNotFoundException": "model", "ValidationException": "bad_request",
                    "ModelNotReadyException": "unavailable"}.get(code, "connection")
            raise LLMError(f"AWS Bedrock error for '{model}' ({code or type(exc).__name__}).", kind, self.name) from exc
        blocks = ((resp.get("output") or {}).get("message") or {}).get("content") or []
        text = "".join(b.get("text", "") for b in blocks).strip()
        calls = [ToolCall(b["toolUse"]["toolUseId"], b["toolUse"]["name"], json.dumps(b["toolUse"].get("input") or {}))
                 for b in blocks if "toolUse" in b]
        if not text and not calls:
            raise LLMError(f"'{model}' returned an empty answer.", "empty", self.name)
        u = resp.get("usage") or {}
        pt, ct = int(u.get("inputTokens") or 0), int(u.get("outputTokens") or 0)
        return ChatResult(text, calls, pt + ct, self.name, model, estimate_cost(self.name, model, pt, ct), "bedrock",
                          str(resp.get("stopReason") or ""), prompt_tokens=pt, completion_tokens=ct)

    def list_models(self, families: Sequence[str] = ()) -> List[str]:
        return [self.settings.bedrock_model]

    def key_info(self) -> KeyInfo:
        return KeyInfo(label=f"bedrock:{self.settings.bedrock_region}")
