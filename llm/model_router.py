"""
Policy-aware model routing across local and cloud providers.

    router = ModelRouter(settings)
    router.set_sensitivity(True)                      # confidential documents in this request
    routed = router.chat("master", messages, tools=..., complexity="low")
    routed.routing  -> {"strategy", "candidates", "excluded": {entry: reason}, "attempts"}

Selection, per role (master | fast | verifier | vision), in this order:
  1. candidate chain (llm/catalog.chain_for_settings): MANUAL_MODEL first, then the configured priority
     order (env override > profile default), plus OpenAI / Anthropic / Bedrock / Ollama when configured;
  2. exclusions: provider not configured · LLM_MODE (local → only Ollama, cloud → no Ollama) ·
     DATA SENSITIVITY (confidential/PII requests never reach a cloud model unless ALLOW_SENSITIVE_CLOUD=true,
     and then only providers in CLOUD_PROVIDERS_ALLOWED) · context window too small · circuit breaker open;
  3. ordering (ROUTING_STRATEGY): priority (default) · cost · latency (observed EWMA) ·
     auto (= cost+latency for low-complexity requests, priority otherwise).
Execution: each candidate is retried LLM_MAX_RETRIES times with exponential backoff for transient errors
(timeout, connection, 5xx); a 429 is retried once only if Retry-After ≤ 2 s, otherwise the next candidate
is tried. Auth errors, moderation refusals and malformed requests stop at once. Consecutive failures open
a per-model circuit for CIRCUIT_COOLDOWN_S. Tokens, cost, latency and failures are tracked per model.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from config.settings import Settings, get_logger
from llm.catalog import LOCAL_PROVIDERS, PROVIDERS, ROLES, VISION_CAPABLE, cost_guard, split
from llm.pricing import blended, context_window
from llm.providers import FALLBACK_KINDS, ChatResult, KeyInfo, LLMError, make_provider

log = get_logger("router")

__all__ = ["ModelRouter", "RoutedResult", "ProviderStatus", "ROLES", "USAGE", "reset_state", "EVENTS",
           "validate_images", "TEXT_ROLES"]

_ROLE_REASONING = {"fast": "none", "verifier": "none", "vision": "none"}
_RETRY_KINDS = {"timeout", "connection", "unavailable"}
_USED_FOR = {"groq": "text answers (free tier)", "openrouter": "images (figures, scanned pages)",
             "anyrouter": "free text + vision fallbacks (verified free routes only)",
             "ollama": "local / offline answers", "openai": "cloud answers", "anthropic": "cloud answers",
             "bedrock": "cloud answers (AWS)"}


@dataclass
class ProviderStatus:
    name: str
    label: str
    configured: bool
    connected: bool = False
    models: List[str] = field(default_factory=list)
    message: str = ""
    key: Optional[KeyInfo] = None
    used_for: str = ""
    local: bool = False


@dataclass
class RoutedResult:
    result: ChatResult
    attempts: List[Tuple[str, str, str]]  # (provider, model, outcome)
    latency_ms: float
    routing: Dict[str, Any] = field(default_factory=dict)

    @property
    def cost(self) -> float:
        return self.result.cost


# ------------------------------------------------------------- process-wide state
_LOCK = threading.Lock()
_CIRCUIT: Dict[str, Dict[str, float]] = {}     # entry -> {"fails", "open_until"}
_LATENCY: Dict[str, float] = {}                # entry -> EWMA ms
USAGE: Dict[str, Dict[str, float]] = {}        # entry -> calls, failures, tokens, cost, latency_ms
EVENTS: "deque[Dict[str, Any]]" = deque(maxlen=300)  # recent attempts (dashboard): no prompts, no keys
TEXT_ROLES = ("master", "fast", "verifier")
# discovery routing: kinds that end the attempt for this model but let the chain continue
_SKIP_PROVIDER_KINDS = {"auth", "permission", "entitlement"}
_SKIP_MODEL_KINDS = {"paid", "unsupported"}
_sleep = time.sleep  # patched in tests

_IMAGE_MAGIC = ((b"\x89PNG\r\n\x1a\n", "image/png"), (b"\xff\xd8\xff", "image/jpeg"), (b"GIF87a", "image/gif"),
                (b"GIF89a", "image/gif"))


def _image_mime(data: bytes) -> str:
    for magic, mime in _IMAGE_MAGIC:
        if data.startswith(magic):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return ""


def validate_images(images: Optional[Sequence[bytes]], max_mb: float) -> List[bytes]:
    """Check every image before any model is called: bytes, known MIME type (PNG/JPEG/GIF/WebP by magic
    bytes), size limit, base64-encodable. Raises LLMError(kind="invalid_image") — never retried."""
    out = []
    for i, img in enumerate(images or []):
        if not isinstance(img, (bytes, bytearray)) or not img:
            raise LLMError(f"Image {i + 1} is empty or not binary image data.", "invalid_image")
        if not _image_mime(bytes(img)):
            raise LLMError(f"Image {i + 1} is not a PNG, JPEG, GIF or WebP image.", "invalid_image")
        if len(img) > max_mb * 1024 * 1024:
            raise LLMError(f"Image {i + 1} is larger than {max_mb:g} MB (MODEL_MAX_IMAGE_MB).", "invalid_image")
        try:
            import base64

            base64.b64encode(bytes(img))
        except Exception as exc:  # pragma: no cover - bytes always encode
            raise LLMError(f"Image {i + 1} could not be encoded.", "invalid_image") from exc
        out.append(bytes(img))
    return out


def reset_state() -> None:
    with _LOCK:
        _CIRCUIT.clear()
        _LATENCY.clear()
        USAGE.clear()
        EVENTS.clear()
    try:
        from llm import discovery
        from llm.providers import reset_key_state

        discovery.reset()
        reset_key_state()
    except Exception:  # pragma: no cover
        pass


def _record(entry: str, ok: bool, ms: float = 0.0, tokens: int = 0, cost: float = 0.0,
            threshold: int = 3, cooldown: float = 60.0, outcome: str = "") -> None:
    try:
        from monitoring.metrics import observe_llm
        observe_llm(entry, "ok" if ok else (outcome or "error"))
    except Exception:
        pass
    with _LOCK:
        u = USAGE.setdefault(entry, {"calls": 0, "failures": 0, "tokens": 0, "cost": 0.0, "latency_ms": 0.0})
        u["calls"] += 1
        c = _CIRCUIT.setdefault(entry, {"fails": 0, "open_until": 0.0})
        if ok:
            u["tokens"] += tokens
            u["cost"] += cost
            u["latency_ms"] += ms
            c["fails"], c["open_until"] = 0, 0.0
            _LATENCY[entry] = ms if entry not in _LATENCY else 0.7 * _LATENCY[entry] + 0.3 * ms
        else:
            u["failures"] += 1
            c["fails"] += 1
            if c["fails"] >= threshold:
                c["open_until"] = time.time() + cooldown
                log.warning("circuit opened for %s (%d consecutive failures)", entry, int(c["fails"]))


def circuit_open(entry: str) -> bool:
    with _LOCK:
        return _CIRCUIT.get(entry, {}).get("open_until", 0.0) > time.time()


def _estimate_tokens(messages: Sequence[Dict[str, Any]], max_tokens: int) -> int:
    chars = 0
    for m in messages:
        c = m.get("content")
        chars += len(c) if isinstance(c, str) else sum(len(str(p.get("text", ""))) for p in c or [])
    return chars // 4 + max_tokens


class ModelRouter:
    def __init__(self, settings: Settings, sensitive: bool = False):
        self.discovery = bool(getattr(settings, "discovery_enabled", False))
        if self.discovery:  # discovery routing uses its own timeout / retry budget (MODEL_REQUEST_TIMEOUT ...)
            import dataclasses

            settings = dataclasses.replace(settings, llm_timeout=settings.model_request_timeout,
                                           llm_max_retries=settings.model_max_retries)
        self.settings = settings
        self.sensitive = sensitive
        self.complexity_hint = "normal"          # set per question by the pipeline (low | normal | high)
        self.last_routing: Dict[str, Any] = {}   # explanation of the most recent choice (shown in the UI)
        self.by_name = {name: make_provider(name, settings) for name in PROVIDERS}
        self.providers = [p for p in self.by_name.values() if p.configured()]

    def set_sensitivity(self, sensitive: bool) -> None:
        self.sensitive = bool(sensitive)

    # ------------------------------------------------------------------ planning
    def _exclusion(self, provider_name: str, model: str, role: str, need_tokens: int) -> str:
        s = self.settings
        p = self.by_name.get(provider_name)
        if p is None:
            return "unknown provider"
        if not p.configured():
            return "not configured"
        local = provider_name in LOCAL_PROVIDERS
        if s.llm_mode == "local" and not local:
            return "LLM_MODE=local"
        if s.llm_mode == "cloud" and local:
            return "LLM_MODE=cloud"
        if not local:
            if self.sensitive and not s.allow_sensitive_cloud:
                return "sensitive data: cloud models not allowed"
            if s.cloud_providers_allowed and provider_name not in s.cloud_providers_allowed:
                return "provider not allowed by policy"
        entry = f"{provider_name}:{model}"
        if self.discovery:
            guard = cost_guard(entry, s)
            if guard:
                log.info("model excluded: %s — %s", entry, guard)
                return guard
            from llm import discovery

            why = discovery.excluded(entry)
            if why:
                return f"excluded: {why}"
            if role == "vision" and provider_name == "anyrouter" and not discovery.verified_vision(entry, s):
                return "no verified image input"
            if role in TEXT_ROLES and provider_name == "anyrouter":
                info = discovery.info_for(entry, s)
                if info is None or not info.text:
                    return "not a verified text model"
        if role == "vision" and provider_name not in VISION_CAPABLE and not (local and s.ollama_vision_model) \
                and not (self.discovery and provider_name == "anyrouter"):
            return "no vision support"
        ctx = context_window(provider_name, model)
        if self.discovery and provider_name in ("anyrouter", "openrouter"):
            from llm import discovery

            info = discovery.info_for(entry, s)
            ctx = info.context if info is not None and info.context else ctx
        if need_tokens and ctx < need_tokens:
            return f"context window too small (needs ~{need_tokens} tokens)"
        if circuit_open(f"{provider_name}:{model}"):
            return "circuit open (recent failures)"
        return ""

    def plan(self, role: str, need_tokens: int = 0, complexity: str = "normal") -> Tuple[list, Dict[str, str]]:
        cands, excluded, circuit_only = [], {}, []
        for entry in self.settings.models_for(role):
            name, model = split(entry)
            why = self._exclusion(name, model, role, need_tokens)
            if why.startswith("circuit open"):
                circuit_only.append((self.by_name[name], model, entry))
            if why:
                excluded[entry] = why
            else:
                cands.append((self.by_name[name], model, entry))
        if not cands and circuit_only:  # everything is cooling down: still try rather than fail
            cands = circuit_only
        strategy = self.settings.routing_strategy
        if strategy == "auto":
            strategy = "cost" if complexity == "low" else "priority"
        if strategy == "cost":
            cands.sort(key=lambda c: (blended(c[0].name, c[1]), _LATENCY.get(c[2], 1e9)))
        elif strategy == "latency":
            cands.sort(key=lambda c: _LATENCY.get(c[2], 1e9))
        return cands, excluded

    def chain(self, role: str):
        """[(provider, model)] usable for the role right now (after policy filters)."""
        return [(p, m) for p, m, _ in self.plan(role)[0]]

    def has_role(self, role: str) -> bool:
        return bool(self.chain(role))

    @property
    def enabled(self) -> bool:
        return bool(self.providers)

    # -------------------------------------------------------------------- chat
    def chat(self, role: str, messages: List[Dict[str, Any]], tools: Optional[List[Dict]] = None,
             tool_choice: str = "auto", images: Optional[Sequence[bytes]] = None,
             max_tokens: Optional[int] = None, reasoning: Optional[str] = None,
             json_mode: bool = False, temperature: Optional[float] = None,
             complexity: str = "normal") -> RoutedResult:
        s = self.settings
        t0 = time.perf_counter()
        max_tokens = max_tokens or s.max_output_tokens
        if complexity == "normal":
            complexity = self.complexity_hint
        if images and self.discovery:
            if role in TEXT_ROLES:  # never send image references to the text-only chain
                raise LLMError("Images can only be sent to the vision model chain.", "unsupported")
            images = validate_images(images, s.model_max_image_mb)
        cands, excluded = self.plan(role, _estimate_tokens(messages, max_tokens), complexity)
        routing: Dict[str, Any] = {"role": role, "strategy": s.routing_strategy, "mode": s.llm_mode,
                                   "complexity": complexity, "sensitive": self.sensitive,
                                   "candidates": [c[2] for c in cands], "excluded": excluded}
        self.last_routing = routing
        attempts: List[Tuple[str, str, str]] = []
        if not cands:
            if self.sensitive and any("sensitive" in v for v in excluded.values()):
                msg = ("This request involves confidential documents, which may only be processed by a local "
                       "model, and no local model (Ollama) is configured. Set OLLAMA_BASE_URL or ask an admin.")
            elif s.llm_mode == "local":
                msg = "Local-only mode is on but no local model is configured. Set OLLAMA_BASE_URL."
            elif role == "vision" and self.discovery:
                msg = ("No verified-free vision model is available right now (AnyRouter catalogue: no routable free "
                       "model with image input, or all failed). Text answers still work.")
            elif role == "vision":
                msg = "No vision model is configured (OPENROUTER_API_KEY, OpenAI, Anthropic or OLLAMA_VISION_MODEL)."
            else:
                msg = ("No AI model is configured for this task. Add GROQ_API_KEY (free at console.groq.com/keys), "
                       "another provider key, or OLLAMA_BASE_URL for a local model.")
            err = LLMError(msg, "no_provider")
            err.routing = routing  # type: ignore[attr-defined]
            raise err
        effort = reasoning or _ROLE_REASONING.get(role) or s.reasoning_effort
        temp = s.llm_temperature if temperature is None else temperature
        last: Optional[LLMError] = None
        log_rows: List[Dict[str, Any]] = []
        routing["attempt_log"] = log_rows
        dead_providers: set = set()
        for provider, model, entry in cands:
            if provider.name in dead_providers:
                continue
            for attempt in range(s.llm_max_retries + 1):
                t_call = time.perf_counter()
                try:
                    result = provider.chat(model, messages, tools=tools, tool_choice=tool_choice, temperature=temp,
                                           max_tokens=max_tokens, images=images, reasoning=effort,
                                           json_mode=json_mode)
                    ms = (time.perf_counter() - t_call) * 1000
                    _record(entry, True, ms, result.tokens, result.cost, s.circuit_failures, s.circuit_cooldown_s)
                    attempts.append((provider.name, model, "ok"))
                    self._log(log_rows, role, provider.name, model, attempt + 1, "ok", ms, "success")
                    if self.discovery:
                        self._after_success(entry, ms, result, routing)
                    routing["attempts"] = attempts
                    routing["chosen"] = entry
                    return RoutedResult(result, attempts, (time.perf_counter() - t0) * 1000, routing)
                except LLMError as exc:
                    ms = (time.perf_counter() - t_call) * 1000
                    _record(entry, False, threshold=s.circuit_failures, cooldown=s.circuit_cooldown_s,
                            outcome=exc.kind)
                    attempts.append((provider.name, model, exc.kind))
                    last = exc
                    if self.discovery:
                        from llm import discovery

                        discovery.note_failure(entry, "unsupported_vision" if (exc.kind == "unsupported" and images)
                                               else exc.kind, s)
                        if exc.kind in _SKIP_PROVIDER_KINDS:  # bad key / no entitlement: stop this provider only
                            self._log(log_rows, role, provider.name, model, attempt + 1, exc.kind, ms,
                                      "provider skipped")
                            dead_providers.add(provider.name)
                            break
                        if exc.kind in _SKIP_MODEL_KINDS:
                            self._log(log_rows, role, provider.name, model, attempt + 1, exc.kind, ms, "model skipped")
                            break
                    if exc.kind not in FALLBACK_KINDS:
                        self._log(log_rows, role, provider.name, model, attempt + 1, exc.kind, ms, "stopped")
                        exc.routing = routing  # type: ignore[attr-defined]
                        raise
                    wait = float(getattr(exc, "retry_after", 0.0) or 0.0)
                    if self.discovery:  # 429 / 5xx / timeout / connection: bounded exponential backoff
                        retry = attempt < s.llm_max_retries and exc.kind in (_RETRY_KINDS | {"rate_limit"})
                        delay = min(max(wait, 0.5 * 2 ** attempt), 8.0)
                    else:
                        retry = attempt < s.llm_max_retries and (
                            exc.kind in _RETRY_KINDS or (exc.kind == "rate_limit" and 0 < wait <= 2.0))
                        delay = wait if exc.kind == "rate_limit" else min(0.4 * 2 ** attempt, 4.0)
                    self._log(log_rows, role, provider.name, model, attempt + 1, exc.kind, ms,
                              "retry" if retry else "next model")
                    if not retry:
                        break
                    (_sleep if self.discovery else time.sleep)(delay)
            log.warning("%s failed (%s); trying the next model for '%s'", entry, last.kind if last else "?", role)
        if last is None:  # every candidate's provider was skipped before a call
            last = LLMError("No model in the chain could be used.", "no_provider")
        last.user_message += " No other model in the chain could answer."
        last.routing = routing  # type: ignore[attr-defined]
        raise last

    # --------------------------------------------------------- 2.2 helpers
    @staticmethod
    def _log(rows: List[Dict[str, Any]], role: str, provider: str, model: str, attempt: int, category: str,
             ms: float, result: str) -> None:
        row = {"ts": round(time.time(), 3), "role": role, "pipeline": "vision" if role == "vision" else "text",
               "provider": provider, "model": model, "attempt": attempt, "category": category,
               "latency_ms": round(ms, 1), "result": result}
        rows.append(row)
        with _LOCK:
            EVENTS.append(row)
        log.info("model attempt pipeline=%s provider=%s model=%s attempt=%d category=%s latency_ms=%.0f result=%s",
                 row["pipeline"], provider, model, attempt, category, ms, result)

    def _after_success(self, entry: str, ms: float, result: ChatResult, routing: Dict[str, Any]) -> None:
        from llm import discovery

        discovery.note_success(entry, ms)
        if entry.startswith("anyrouter:") and (result.cost or 0) > 0:
            # cost guard, after the fact: a "free" route reported a charge → never use it again automatically
            discovery.exclude(entry, f"reported a cost of ${result.cost:.6f} (not free)")
            routing["cost_alert"] = f"{entry} reported a charge; it has been excluded"
            log.error("cost guard: %s reported cost %.6f although classified free; excluded", entry, result.cost)

    # ------------------------------------------------- 2.2 public pipeline API
    def generate_text(self, messages: List[Dict[str, Any]], role: str = "master", **kw) -> RoutedResult:
        """TEXT pipeline only (Groq ×4 → AnyRouter free text ×2 in discovery mode). Never sends images."""
        if role not in TEXT_ROLES:
            raise ValueError("generate_text() is for text roles: master, fast, verifier")
        if kw.get("images"):
            raise LLMError("generate_text() does not accept images; use analyze_image().", "unsupported")
        return self.chat(role, messages, **kw)

    def analyze_image(self, prompt: str, images: Sequence[bytes], system: str = "", **kw) -> RoutedResult:
        """VISION pipeline only (AnyRouter free vision ×3 in discovery mode). Images are validated first."""
        if not images:
            raise LLMError("analyze_image() needs at least one image.", "invalid_image")
        msgs: List[Dict[str, Any]] = ([{"role": "system", "content": system}] if system else [])
        msgs.append({"role": "user", "content": prompt})
        return self.chat("vision", msgs, images=images, **kw)

    def generate_multimodal_response(self, question: str, evidence: str, images: Sequence[bytes] = (),
                                     image_note: str = "", system: str = "") -> Dict[str, Any]:
        """Mixed question: vision chain with the images; if every vision model fails, the text chain answers
        from the textual evidence ONLY and the result says so (`visual_interpretation` False). Original images
        are never modified or discarded."""
        user = f"Question: {question}\n\nEvidence:\n{evidence}"
        base = [{"role": "system", "content": system}] if system else []
        warnings: List[str] = []
        if images:
            try:
                routed = self.chat("vision", base + [{"role": "user", "content": user + (
                    f"\n\nAttached images (in order): {image_note}" if image_note else "")}], images=images)
                return {"result": routed, "visual_interpretation": True, "pipeline": "vision", "warnings": warnings}
            except LLMError as exc:
                if exc.kind == "invalid_image":
                    warnings.append(f"The figure images could not be used ({exc.user_message}).")
                else:
                    warnings.append("No vision model could read the figure images this time, so the answer uses "
                                    "the text evidence (captions, nearby text) only.")
        routed = self.chat("master", base + [{"role": "user", "content": user}])
        return {"result": routed, "visual_interpretation": False, "pipeline": "text", "warnings": warnings}

    # ------------------------------------------------------------------ status
    def status(self) -> List[ProviderStatus]:
        out = []
        for name, p in self.by_name.items():
            st = ProviderStatus(name, p.label, p.configured(), used_for=_USED_FOR.get(name, ""),
                                local=name in LOCAL_PROVIDERS)
            if st.configured:
                try:
                    st.key = p.key_info()
                    st.connected = True
                except LLMError as exc:
                    st.message = exc.user_message
                except Exception as exc:  # e.g. boto3 credential errors
                    st.message = f"{type(exc).__name__}"
                if st.connected:
                    try:
                        st.models = p.list_models()
                    except Exception:
                        st.models = []
            out.append(st)
        return out

    def health(self) -> Dict[str, bool]:
        """Cheap liveness per configured provider (used by /readyz and the Models page)."""
        return {s.name: s.connected for s in self.status() if s.configured}
