"""
Two OpenAI-compatible providers behind one interface (plain httpx, no vendor SDK):

    GroqProvider        TEXT: answers, tool calls, verification   (free tier: gpt-oss, Llama)
    OpenRouterProvider  IMAGES: figure questions, OCR, figure descriptions (Gemma 4, Qwen 3.8)
                        + optional Qwen3 embeddings / reranker, key credit info

    chat()        POST /chat/completions   (tool calling, images, hidden reasoning control)
    embed()       POST /embeddings         (OpenRouter only)
    rerank()      POST /rerank             (OpenRouter only)
    key_info()    GET  /key                (OpenRouter only: credit limit / usage)
    list_models() GET  /models

Each provider sends only the parameters its API accepts (Groq rejects
OpenRouter's `provider` / `reasoning` objects, so they are never sent there).

Every call returns normalised dataclasses and raises LLMError with a
user-safe message (never containing the key). The model's hidden reasoning
is never stored or returned: the UI shows actions, not chain-of-thought.
"""
from __future__ import annotations

import base64
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import httpx

from config.settings import Settings, get_logger

log = get_logger("openrouter")

# Errors after which the router may try the next model in the chain.
FALLBACK_KINDS = {"rate_limit", "connection", "timeout", "model", "api", "quota", "unavailable", "empty"}


class LLMError(Exception):
    """`user_message` is safe to show in the UI (never contains keys)."""

    def __init__(self, user_message: str, kind: str = "error", provider: str = "", status: int = 0):
        super().__init__(user_message)
        self.user_message = user_message
        self.kind = kind
        self.provider = provider
        self.status = status


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON text


@dataclass
class ChatResult:
    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    tokens: int = 0
    provider: str = "openrouter"
    model: str = ""            # model that actually answered (as reported by OpenRouter)
    cost: float = 0.0          # USD, from usage.cost
    upstream: str = ""         # upstream provider that served it (e.g. "Alibaba", "Google AI Studio")
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class EmbedResult:
    vectors: List[List[float]]
    tokens: int = 0
    cost: float = 0.0


@dataclass
class KeyInfo:
    label: str = ""
    usage: float = 0.0             # USD spent with this key
    limit: Optional[float] = None  # USD limit on the key (None = unlimited)
    remaining: Optional[float] = None
    is_free_tier: bool = False


# ---------------------------------------------------------------- backup key pool (2.2.1)
# Process-wide health per key, stored under a short SHA-256 fingerprint (never the key itself).
_KEY_STATE: Dict[str, Dict[str, Any]] = {}
_KEY_LOCK = threading.Lock()
_KEY_COOLDOWN = {"rate_limit": 60.0, "quota": 600.0, "auth": 3600.0}


def _fp(key: str) -> str:
    import hashlib

    return hashlib.sha256(key.encode()).hexdigest()[:12]


def key_pool_status(provider: str, keys: Sequence[str]) -> List[Dict[str, Any]]:
    """[{"key": "key #1", "state": "ok" | "cooling" | "rejected", "until": ts, "reason"}] — no secrets."""
    now, out = time.time(), []
    with _KEY_LOCK:
        for i, k in enumerate(keys, 1):
            st = _KEY_STATE.get(f"{provider}:{_fp(k)}") or {}
            until = float(st.get("until") or 0)
            state = "ok" if until <= now else ("rejected" if st.get("reason") == "auth" else "cooling")
            out.append({"key": f"key #{i}", "state": state, "until": until if state != "ok" else 0,
                        "reason": st.get("reason", "") if state != "ok" else "", "uses": st.get("uses", 0)})
    return out


def reset_key_state() -> None:
    with _KEY_LOCK:
        _KEY_STATE.clear()


_CLIENTS: Dict[tuple, httpx.Client] = {}
_CLIENTS_LOCK = threading.Lock()


def _client(base_url: str, timeout: float) -> httpx.Client:
    key = (base_url, timeout)
    with _CLIENTS_LOCK:
        c = _CLIENTS.get(key)
        if c is None:
            c = httpx.Client(base_url=base_url, timeout=httpx.Timeout(timeout, connect=15.0),
                             limits=httpx.Limits(max_connections=32, max_keepalive_connections=16))
            _CLIENTS[key] = c
        return c


def _key_list(primary: str, backups: Sequence[str]) -> List[str]:
    return [k for k in dict.fromkeys([primary, *backups]) if k] if primary else []


def _png_data_url(data: bytes) -> str:
    mime = "image/jpeg" if data[:3] == b"\xff\xd8\xff" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(data).decode()


class OpenRouterProvider:
    name, label = "openrouter", "OpenRouter"
    local = False

    key_env, key_url = "OPENROUTER_API_KEY", "openrouter.ai/settings/keys"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = settings.openrouter_api_key
        self.api_keys = _key_list(self.api_key, getattr(settings, "openrouter_backup_keys", ()))
        self.base_url = settings.openrouter_base_url

    def configured(self) -> bool:
        return bool(self.api_key)

    # ---------------------------------------------------------- key pool
    def _keys_in_order(self, model: str = "") -> List[str]:
        """Healthy keys first (in configured order); if every key is cooling down, the soonest-available.
        Rate limits are tracked per key AND model (provider limits are per model); bad/empty keys per key."""
        keys = list(getattr(self, "api_keys", None) or ([self.api_key] if self.api_key else []))
        if len(keys) <= 1:
            return keys
        now = time.time()
        with _KEY_LOCK:
            until = {k: max(float((_KEY_STATE.get(f"{self.name}:{_fp(k)}") or {}).get("until") or 0),
                            float((_KEY_STATE.get(f"{self.name}:{_fp(k)}:{model}") or {}).get("until") or 0))
                     for k in keys}
        ready = [k for k in keys if until[k] <= now]
        return ready or sorted(keys, key=lambda k: until[k])[:1]

    def _key_failed(self, key: str, kind: str, retry_after: float = 0.0, model: str = "") -> None:
        cool = max(retry_after, _KEY_COOLDOWN.get(kind, 60.0))
        slot = f"{self.name}:{_fp(key)}" + (f":{model}" if kind == "rate_limit" and model else "")
        with _KEY_LOCK:
            st = _KEY_STATE.setdefault(slot, {"uses": 0})
            st.update(until=time.time() + cool, reason=kind)
        idx = (self.api_keys.index(key) + 1) if key in getattr(self, "api_keys", []) else 0
        log.warning("%s key #%d %s; switching to the next backup key for %.0f s", self.label, idx,
                    {"auth": "was rejected", "quota": "is out of credit", "rate_limit": "hit its rate limit"}
                    .get(kind, kind), cool)

    def _key_ok(self, key: str) -> None:
        with _KEY_LOCK:
            st = _KEY_STATE.setdefault(f"{self.name}:{_fp(key)}", {"uses": 0})
            st["uses"] = st.get("uses", 0) + 1

    # ------------------------------------------------------------ transport
    def _headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise LLMError(f"No {self.label} API key is set. Add {self.key_env} to the app's secrets "
                           f"(create one at {self.key_url}).", "auth", self.name)
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
                "HTTP-Referer": self.settings.app_url, "X-Title": self.settings.app_title}

    def _request(self, method: str, path: str, body: Optional[Dict] = None, model: str = "",
                 timeout: Optional[float] = None) -> Dict[str, Any]:
        """One HTTP call with a single retry for transient upstream failures (502/503/529)."""
        keys = self._keys_in_order(model)
        if len(keys) <= 1:
            return self._request_with_key(method, path, body, model, timeout, keys[0] if keys else "")
        last: Optional[LLMError] = None
        for key in keys:  # backup keys: same request, next key, on 401 / 402 / 429
            try:
                data = self._request_with_key(method, path, body, model, timeout, key)
                self._key_ok(key)
                return data
            except LLMError as exc:
                if exc.kind not in ("auth", "quota", "rate_limit", "paid"):
                    raise
                self._key_failed(key, "quota" if exc.kind == "paid" else exc.kind,
                                 float(getattr(exc, "retry_after", 0.0) or 0.0), model)
                last = exc
        assert last is not None
        if len(keys) > 1:
            last.user_message += f" All {len(keys)} {self.label} keys were tried."
        raise last

    def _request_with_key(self, method: str, path: str, body: Optional[Dict], model: str,
                          timeout: Optional[float], key: str) -> Dict[str, Any]:
        if key:
            self.api_key = key  # _headers() of every subclass reads self.api_key
        client = _client(self.base_url, timeout or self.settings.llm_timeout)
        last: Optional[LLMError] = None
        for attempt in range(2):
            try:
                resp = client.request(method, path, json=body, headers=self._headers())
            except httpx.TimeoutException as exc:
                raise LLMError(f"{self.label} took too long to respond.", "timeout", self.name) from exc
            except httpx.HTTPError as exc:
                raise LLMError(f"Cannot reach {self.label} right now. Check the internet connection.",
                               "connection", self.name) from exc
            if resp.status_code < 400:
                try:
                    data = resp.json()
                except ValueError as exc:
                    raise LLMError(f"{self.label} returned an unreadable response.", "api", self.name) from exc
                if isinstance(data, dict) and data.get("error") and not data.get("choices") \
                        and not data.get("data") and not data.get("results"):
                    raise self._translate(int(data["error"].get("code") or 502), data, model)
                return data
            try:
                payload = resp.json()
            except ValueError:
                payload = {"error": {"message": resp.text[:300]}}
            last = self._translate(resp.status_code, payload, model)
            last.provider = self.name
            try:
                last.retry_after = float(resp.headers.get("retry-after") or 0)  # type: ignore[attr-defined]
            except ValueError:
                pass
            if resp.status_code in (502, 503, 529) and attempt == 0:
                time.sleep(1.2)
                continue
            raise last
        assert last is not None
        raise last

    @staticmethod
    def _translate(status: int, payload: Dict, model: str) -> LLMError:
        err = (payload or {}).get("error") or {}
        msg = str(err.get("message") or "")[:300]
        meta = err.get("metadata") or {}
        name = model or "the model"
        if status == 401:
            return LLMError("OpenRouter rejected the API key. Create a new one at openrouter.ai/settings/keys.",
                            "auth", status=status)
        if status == 402:
            return LLMError(f"Not enough OpenRouter credits for '{name}'. Add credits or switch to the Free "
                            "profile.", "quota", status=status)
        if status == 403:
            reason = ", ".join(meta.get("reasons", [])) if isinstance(meta.get("reasons"), list) else ""
            return LLMError(f"'{name}' refused this request (moderation{': ' + reason if reason else ''}).",
                            "moderation", status=status)
        if status == 404:
            return LLMError(f"'{name}' is not available on OpenRouter (or no provider supports the requested "
                            "features).", "model", status=status)
        if status == 408:
            return LLMError("OpenRouter timed out.", "timeout", status=status)
        if status == 413:
            return LLMError("The request was too large for the model.", "bad_request", status=status)
        if status == 429:
            free = ":free" in name
            return LLMError(f"Rate limit reached for '{name}'" + (" (free endpoints allow ~20 requests/min)."
                            if free else "."), "rate_limit", status=status)
        if status == 400:
            low = msg.lower()
            if "tool" in low and ("call" in low or "parse" in low or "invalid" in low):
                return LLMError("The model produced an invalid tool call.", "tool_use_failed", status=status)
            if "context" in low and ("length" in low or "too long" in low):
                return LLMError("The question plus evidence is too long for the model.", "bad_request",
                                status=status)
            log.warning("OpenRouter rejected a request for %s: %s", name, msg)
            return LLMError("OpenRouter rejected the request.", "bad_request", status=status)
        if status in (502, 503, 529):
            return LLMError(f"'{name}' is temporarily unavailable upstream.", "unavailable", status=status)
        log.warning("OpenRouter error %s for %s: %s", status, name, msg)
        return LLMError("OpenRouter returned an error.", "api", status=status)

    # ----------------------------------------------------------------- chat
    def _body(self, model: str, messages: List[Dict[str, Any]], tools, tool_choice: str, temperature: float,
              max_tokens: int, reasoning: Optional[str], json_mode: bool) -> Dict[str, Any]:
        body: Dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature,
                                "max_tokens": max_tokens}
        effort = reasoning or self.settings.reasoning_effort
        # Hidden reasoning: bounded effort, and never returned to us (exclude).
        body["reasoning"] = {"enabled": False} if effort == "none" else {"effort": effort, "exclude": True}
        provider: Dict[str, Any] = {"allow_fallbacks": True}
        if self.settings.data_collection == "deny":
            provider["data_collection"] = "deny"
        if tools:
            body.update(tools=tools, tool_choice=tool_choice)
            provider["require_parameters"] = True  # only route to upstreams that really support tools
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        body["provider"] = provider
        return body

    def chat(self, model: str, messages: List[Dict[str, Any]], tools: Optional[List[Dict]] = None,
             tool_choice: str = "auto", temperature: float = 0.1, max_tokens: int = 1500,
             images: Optional[Sequence[bytes]] = None, reasoning: Optional[str] = None,
             json_mode: bool = False) -> ChatResult:
        if images:  # attach images to the last user message (OpenAI vision format)
            messages = [dict(m) for m in messages]
            last = messages[-1]
            parts: List[Dict[str, Any]] = [{"type": "text", "text": last.get("content", "")}]
            parts += [{"type": "image_url", "image_url": {"url": _png_data_url(b)}} for b in images]
            last["content"] = parts
        body = self._body(model, messages, tools, tool_choice, temperature, max_tokens, reasoning, json_mode)
        data = self._request("POST", "/chat/completions", body, model)
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"'{model}' returned no answer.", "empty")
        choice = choices[0]
        if choice.get("error"):
            raise self._translate(int(choice["error"].get("code") or 502), {"error": choice["error"]}, model)
        msg = choice.get("message") or {}
        calls = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            args = fn.get("arguments") or "{}"
            if not isinstance(args, str):
                args = json.dumps(args)
            calls.append(ToolCall(tc.get("id") or f"call_{i}", fn.get("name", ""), args))
        content = msg.get("content") or ""
        if isinstance(content, list):  # some upstreams return content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        usage = data.get("usage") or {}
        if not content.strip() and not calls:
            raise LLMError(f"'{model}' returned an empty answer.", "empty")
        return ChatResult(content=content.strip(), tool_calls=calls, tokens=int(usage.get("total_tokens") or 0),
                          provider=self.name, model=str(data.get("model") or model),
                          cost=float(usage.get("cost") or 0.0),
                          upstream=str(data.get("provider") or ""), finish_reason=str(choice.get("finish_reason") or ""),
                          prompt_tokens=int(usage.get("prompt_tokens") or 0),
                          completion_tokens=int(usage.get("completion_tokens") or 0))

    # ------------------------------------------------------- embeddings
    def embed(self, model: str, texts: Sequence[str], dimensions: Optional[int] = None) -> EmbedResult:
        body: Dict[str, Any] = {"model": model, "input": list(texts), "encoding_format": "float"}
        if dimensions:
            body["dimensions"] = int(dimensions)
        if self.settings.data_collection == "deny":
            body["provider"] = {"data_collection": "deny"}
        data = self._request("POST", "/embeddings", body, model, timeout=max(60.0, self.settings.llm_timeout))
        rows = sorted(data.get("data") or [], key=lambda r: r.get("index", 0))
        vectors = []
        for r in rows:
            emb = r.get("embedding")
            if isinstance(emb, str):  # base64 float32
                import numpy as np

                emb = np.frombuffer(base64.b64decode(emb), dtype="<f4").tolist()
            vectors.append([float(x) for x in emb or []])
        if len(vectors) != len(texts) or any(not v for v in vectors):
            raise LLMError("The embedding model returned an incomplete result.", "api")
        usage = data.get("usage") or {}
        return EmbedResult(vectors, int(usage.get("total_tokens") or 0), float(usage.get("cost") or 0.0))

    # ----------------------------------------------------------- rerank
    def rerank(self, model: str, query: str, documents: Sequence[str]) -> List[float]:
        """Relevance score (0..1) for every document, in input order."""
        body: Dict[str, Any] = {"model": model, "query": query, "documents": list(documents),
                                "top_n": len(documents)}
        data = self._request("POST", "/rerank", body, model, timeout=30.0)
        scores = [0.0] * len(documents)
        for r in data.get("results") or []:
            idx = int(r.get("index", -1))
            if 0 <= idx < len(scores):
                scores[idx] = float(r.get("relevance_score") or 0.0)
        return scores

    # ------------------------------------------------------- account info
    def key_info(self) -> KeyInfo:
        d = (self._request("GET", "/key", None, timeout=15.0) or {}).get("data") or {}
        limit = d.get("limit")
        remaining = d.get("limit_remaining")
        return KeyInfo(label=str(d.get("label") or ""), usage=float(d.get("usage") or 0.0),
                       limit=None if limit is None else float(limit),
                       remaining=None if remaining is None else float(remaining),
                       is_free_tier=bool(d.get("is_free_tier")))

    def models_raw(self, timeout: float = 20.0) -> List[Dict[str, Any]]:
        """GET /models: id, pricing, architecture.input_modalities, context_length (OpenRouter model objects)."""
        data = self._request("GET", "/models", None, timeout=timeout)
        return [m for m in data.get("data") or [] if isinstance(m, dict) and m.get("id")]

    def list_models(self, families: Sequence[str] = ("qwen/", "google/gemma")) -> List[str]:
        data = self._request("GET", "/models", None, timeout=20.0)
        ids = [str(m.get("id", "")) for m in data.get("data") or []]
        return sorted(i for i in ids if i and (not families or any(i.startswith(f) for f in families)))


# =============================================================================
# Groq (text): free tier, very fast, OpenAI-compatible
# =============================================================================
_GROQ_NON_CHAT = ("whisper", "orpheus", "guard", "tts", "playai", "distil", "safeguard")


class GroqProvider(OpenRouterProvider):
    """Same transport and response parsing; Groq-specific request body and errors.

    Groq's free tier reports no cost (cost = 0). Hidden reasoning of gpt-oss models
    is kept short (`reasoning_effort`) and never returned (`include_reasoning: false`)."""

    name, label = "groq", "Groq"
    key_env, key_url = "GROQ_API_KEY", "console.groq.com/keys"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = settings.groq_api_key
        self.api_keys = _key_list(self.api_key, getattr(settings, "groq_backup_keys", ()))
        self.base_url = settings.groq_base_url

    def _headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise LLMError("No Groq API key is set. Add GROQ_API_KEY to the app's secrets "
                           "(create a free key at console.groq.com/keys).", "auth", self.name)
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _body(self, model, messages, tools, tool_choice, temperature, max_tokens, reasoning, json_mode):
        body: Dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature,
                                "max_completion_tokens": max_tokens}
        if model.startswith("openai/gpt-oss"):
            effort = reasoning or self.settings.reasoning_effort
            body["reasoning_effort"] = "low" if effort == "none" else effort
            body["include_reasoning"] = False
        if tools:
            body.update(tools=tools, tool_choice=tool_choice)
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    @staticmethod
    def _translate(status: int, payload: Dict, model: str) -> LLMError:
        err = (payload or {}).get("error") or {}
        code = str(err.get("code") or "")
        msg = str(err.get("message") or "")[:300]
        name = model or "the model"
        if status == 401:
            return LLMError("Groq rejected the API key. Create a new one at console.groq.com/keys.", "auth",
                            "groq", status)
        if status == 413 or "rate_limit" in code or status == 429:
            return LLMError(f"Groq's free limit was reached for '{name}' (requests or tokens per minute/day).",
                            "rate_limit", "groq", status)
        if status == 404 or code == "model_not_found" or "decommissioned" in msg.lower():
            return LLMError(f"'{name}' is not available on Groq.", "model", "groq", status)
        if status == 400 and (code == "tool_use_failed" or "tool call" in msg.lower()):
            return LLMError("The model produced an invalid tool call.", "tool_use_failed", "groq", status)
        if status == 400 and "context_length" in (code + msg.lower()):
            return LLMError("The question plus evidence is too long for the model.", "bad_request", "groq", status)
        if status == 400:
            log.warning("Groq rejected a request for %s: %s", name, msg)
            return LLMError("Groq rejected the request.", "bad_request", "groq", status)
        if status in (498, 500, 502, 503):
            return LLMError(f"Groq is temporarily unavailable for '{name}'.", "unavailable", "groq", status)
        return LLMError("Groq returned an error.", "api", "groq", status)

    def models_raw(self, timeout: float = 20.0) -> List[Dict[str, Any]]:
        """GET /models: id, owned_by, active, context_window, max_completion_tokens (Groq model objects)."""
        data = self._request("GET", "/models", None, timeout=timeout)
        return [m for m in data.get("data") or [] if isinstance(m, dict) and m.get("id")]

    def list_models(self, families: Sequence[str] = ()) -> List[str]:
        data = self._request("GET", "/models", None, timeout=20.0)
        return sorted(str(m.get("id")) for m in data.get("data") or []
                      if m.get("id") and not any(h in str(m["id"]).lower() for h in _GROQ_NON_CHAT))

    def key_info(self) -> KeyInfo:  # Groq has no key-info endpoint; listing models proves the key works
        self.list_models()
        return KeyInfo(label="groq", is_free_tier=True)

    def embed(self, *a, **kw):  # pragma: no cover - not offered by this app on Groq
        raise LLMError("Embeddings are not used on Groq.", "bad_request", "groq")

    def rerank(self, *a, **kw):  # pragma: no cover
        raise LLMError("Reranking is not used on Groq.", "bad_request", "groq")


# =============================================================================
# AnyRouter (free text + vision fallbacks): OpenAI-compatible gateway
# =============================================================================
class AnyRouterProvider(OpenRouterProvider):
    """https://anyrouter.dev/api/v1 — plain OpenAI-compatible body (no OpenRouter-only objects).

    Which AnyRouter models may be called is decided by llm/discovery.py (verified free routes only);
    this class only speaks the wire format. Every response's reported cost is checked by the router's
    cost guard: a "free" model that reports a charge is excluded at once."""

    name, label = "anyrouter", "AnyRouter"
    key_env, key_url = "ANYROUTER_API_KEY", "anyrouter.dev"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = settings.anyrouter_api_key
        self.api_keys = _key_list(self.api_key, ())
        self.base_url = settings.anyrouter_base_url

    def _headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise LLMError("No AnyRouter API key is set. Add ANYROUTER_API_KEY to the app's secrets.", "auth",
                           self.name)
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
        err = (payload or {}).get("error") or {}
        code = str(err.get("code") or err.get("type") or "")
        msg = str(err.get("message") or "")[:300]
        name = model or "the model"
        low = (code + " " + msg).lower()
        if status == 401:
            return LLMError("AnyRouter rejected the API key (ANYROUTER_API_KEY).", "auth", "anyrouter", status)
        if status == 402 or "insufficient_balance" in low:
            # a free route must never need credits: treat as "not free for this account"
            return LLMError(f"AnyRouter asked for credits for '{name}', so it is not free for this account.",
                            "paid", "anyrouter", status)
        if status == 403 and ("free_tier" in low or "subscription" in low or "plan" in low):
            return LLMError("AnyRouter free models need an eligible plan on this account (free-tier entitlement "
                            "missing). An API key alone does not enable free usage.", "entitlement", "anyrouter",
                            status)
        if status == 403:
            if "moderation" in low or "content" in low or "policy" in low:
                return LLMError(f"'{name}' refused this request (content policy).", "moderation", "anyrouter",
                                status)
            return LLMError(f"AnyRouter denied access to '{name}'.", "permission", "anyrouter", status)
        if status == 404:
            return LLMError(f"'{name}' is not available on AnyRouter.", "model", "anyrouter", status)
        if status == 408:
            return LLMError("AnyRouter timed out.", "timeout", "anyrouter", status)
        if status == 413:
            return LLMError("The request (or image) is too large for the model.", "bad_request", "anyrouter", status)
        if status == 415 or (status == 400 and ("image" in low or "modal" in low)):
            return LLMError(f"'{name}' does not accept this input format.", "unsupported", "anyrouter", status)
        if status == 429:
            return LLMError(f"AnyRouter rate limit reached for '{name}'.", "rate_limit", "anyrouter", status)
        if status == 400:
            log.warning("AnyRouter rejected a request for %s (%s)", name, code or "bad request")
            return LLMError("AnyRouter rejected the request.", "bad_request", "anyrouter", status)
        if status in (500, 502, 503, 504, 529):
            return LLMError(f"'{name}' is temporarily unavailable on AnyRouter.", "unavailable", "anyrouter", status)
        return LLMError("AnyRouter returned an error.", "api", "anyrouter", status)

    def models_raw(self, timeout: float = 20.0) -> List[Dict[str, Any]]:
        """GET /models with this key: the caller-specific catalogue (access + can_route per model)."""
        data = self._request("GET", "/models", None, timeout=timeout)
        rows = data.get("data") if isinstance(data, dict) else data
        return [r for r in rows or [] if isinstance(r, dict) and r.get("id")]

    def list_models(self, families: Sequence[str] = ()) -> List[str]:
        return sorted(str(m["id"]) for m in self.models_raw())

    def key_info(self) -> KeyInfo:  # listing models proves the key works (no generation, no cost)
        self.models_raw(timeout=15.0)
        return KeyInfo(label="anyrouter")

    def embed(self, *a, **kw):  # pragma: no cover
        raise LLMError("Embeddings are not used on AnyRouter.", "bad_request", "anyrouter")

    def rerank(self, *a, **kw):  # pragma: no cover
        raise LLMError("Reranking is not used on AnyRouter.", "bad_request", "anyrouter")


def make_provider(name: str, settings: Settings):
    from llm.adapters import AnthropicProvider, BedrockProvider, OllamaProvider, OpenAIProvider

    cls = {"groq": GroqProvider, "openrouter": OpenRouterProvider, "ollama": OllamaProvider, "openai": OpenAIProvider,
           "anthropic": AnthropicProvider, "bedrock": BedrockProvider,
           "anyrouter": AnyRouterProvider}.get(name, OpenRouterProvider)
    return cls(settings)
