"""
Dynamic model discovery + free-only cost guard (2.2).

Two independent pipelines, built from the providers' own model lists (no hard-coded model ids):

    TEXT    groq primary · groq fallback 1-3 · anyrouter free text 1-2      (roles master / fast / verifier)
    VISION  anyrouter free vision 1-3                                       (role vision)

Groq     GET {GROQ_BASE_URL}/models        → active chat models (speech, guard, embedding and agentic
                                              "compound" systems excluded), ranked by the app's preferred
                                              order, then context window, then size.
AnyRouter GET {ANYROUTER_BASE_URL}/models   → per-key catalogue with `access`, `can_route`, pricing and
                                              capability metadata. A model is accepted as FREE only when
                                              ALL of these hold (fail closed — anything unknown is rejected):
                                                * can_route is explicitly true for this key;
                                                * access marks a free route (free / pool), not BYOK-only and
                                                  not a credit price;
                                                * every pricing value present is explicitly zero (or "free"),
                                                  and at least prompt/input AND completion/output are given;
                                                * the model is not deprecated/disabled and is a concrete model
                                                  (virtual aliases such as anyrouter/free or anyrouter/auto
                                                  hide which model answers and are not used).
                                              Vision candidates additionally need an explicit image input
                                              modality or a vision capability flag.

Catalogue: cached in memory and in DATA_DIR/model_catalog.json, refreshed at most every
MODEL_CATALOG_REFRESH_MINUTES (never per query). A cached AnyRouter selection older than the refresh
interval is re-validated before it is used again; if re-validation is impossible, AnyRouter models are
NOT used (fail closed) while Groq keeps working.

Health: models that fail MODEL_FAILURE_EXCLUDE times in a row, return 404, ask for credits, or report a
non-zero cost are excluded and the next ranked candidate takes their slot until the next refresh.
Every exclusion is logged with its reason (never with a key).
"""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from config.settings import Settings, get_logger

log = get_logger("discovery")

GROQ_SLOTS = 4             # primary + 3 fallbacks
ANY_TEXT_SLOTS = 2
ANY_VISION_SLOTS = 3
OR_TEXT_SLOTS = 2          # OpenRouter free text (only when placed in a chain with @openrouter_text)
OR_VISION_SLOTS = 3        # OpenRouter free vision (only when placed in a chain with @openrouter_vision)
CATALOG_VERSION = 2
# Chain placeholders usable in MASTER_MODELS / FAST_MODELS / VERIFIER_MODELS / VISION_MODELS (discovery mode):
TOKENS = ("@groq", "@anyrouter_text", "@anyrouter_vision", "@openrouter_text", "@openrouter_vision")

_GROQ_EXCLUDE = ("whisper", "orpheus", "guard", "tts", "playai", "distil", "safeguard", "embed", "compound",
                 "allam")  # speech, safety classifiers, embeddings, agentic web systems, narrow-language
_GROQ_PREFERRED = ("openai/gpt-oss-120b", "llama-3.3-70b-versatile", "openai/gpt-oss-20b",
                   "meta-llama/llama-4-maverick", "meta-llama/llama-4-scout", "qwen/qwen3-32b",
                   "moonshotai/kimi-k2", "llama-3.1-8b-instant")
_ALIAS_PREFIXES = ("anyrouter/",)
_PRICE_KEYS_IN = ("prompt", "input", "input_tokens", "prompt_tokens")
_PRICE_KEYS_OUT = ("completion", "output", "output_tokens", "completion_tokens")


# =============================================================================
# Records
# =============================================================================
@dataclass
class ModelInfo:
    provider: str
    id: str
    context: int = 0
    text: bool = True
    vision: bool = False
    tools: bool = False
    free: bool = False
    free_reason: str = ""          # why it is (or is not) free — shown in the admin UI
    vision_reason: str = ""
    available: bool = True
    access: str = ""
    can_route: Optional[bool] = None
    pricing: Dict[str, Any] = field(default_factory=dict)

    @property
    def entry(self) -> str:
        return f"{self.provider}:{self.id}"


@dataclass
class Catalog:
    version: int = CATALOG_VERSION
    refreshed_at: float = 0.0
    groq: List[ModelInfo] = field(default_factory=list)          # ranked text candidates
    any_text: List[ModelInfo] = field(default_factory=list)      # ranked verified-free text candidates
    any_vision: List[ModelInfo] = field(default_factory=list)    # ranked verified-free vision candidates
    or_text: List[ModelInfo] = field(default_factory=list)       # OpenRouter ':free' text candidates
    or_vision: List[ModelInfo] = field(default_factory=list)     # OpenRouter ':free' vision candidates
    rejected: Dict[str, str] = field(default_factory=dict)       # entry -> reason
    errors: Dict[str, str] = field(default_factory=dict)         # provider -> safe error message
    anyrouter_validated_at: float = 0.0

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "Catalog":
        def models(key):
            return [ModelInfo(**{k: v for k, v in m.items() if k in ModelInfo.__dataclass_fields__})
                    for m in d.get(key) or []]
        return cls(int(d.get("version") or 0), float(d.get("refreshed_at") or 0), models("groq"),
                   models("any_text"), models("any_vision"), models("or_text"), models("or_vision"),
                   dict(d.get("rejected") or {}), dict(d.get("errors") or {}),
                   float(d.get("anyrouter_validated_at") or 0))


# =============================================================================
# Classification (pure functions — unit tested with mocked catalogue entries)
# =============================================================================
def _num(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        t = v.strip().lower().replace("$", "")
        if t in ("free", "0", "0.0", "0.00"):
            return 0.0
        try:
            return float(t)
        except ValueError:
            return None
    return None


def _flatten_pricing(p: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(p, dict):
        for k, v in p.items():
            key = f"{prefix}{k}".lower()
            if isinstance(v, dict):
                out.update(_flatten_pricing(v, key + "."))
            else:
                out[key] = v
    return out


def pricing_verdict(m: Dict[str, Any]) -> Tuple[Optional[bool], str]:
    """(True = verified zero, False = paid, None = unknown) + reason."""
    raw = m.get("pricing")
    if isinstance(raw, str):
        return (True, "pricing: free") if raw.strip().lower() == "free" else (None, f"pricing not understood ({raw[:20]})")
    flat = _flatten_pricing(raw)
    if not flat:
        return None, "unknown pricing (no pricing metadata)"
    values = {k: _num(v) for k, v in flat.items() if v is not None and v != ""}
    has_in = any(k.split(".")[-1] in _PRICE_KEYS_IN for k in values)
    has_out = any(k.split(".")[-1] in _PRICE_KEYS_OUT for k in values)
    if not (has_in and has_out):
        return None, "unknown pricing (input/output price missing)"
    if any(v is None for v in values.values()):
        bad = next(k for k, v in values.items() if v is None)
        return None, f"unknown pricing (unreadable value for '{bad}')"
    nonzero = {k: v for k, v in values.items() if v and v > 0}
    if nonzero:
        k, v = next(iter(nonzero.items()))
        return False, f"paid ({k}={v:g})"
    return True, "pricing verified zero"


def _access_text(a: Any) -> str:
    if isinstance(a, str):
        return a.lower()
    if isinstance(a, dict):
        parts = []
        for k, v in a.items():
            if v is True:
                parts.append(str(k))
            elif isinstance(v, (str, int, float)) and v not in ("", 0, False):
                parts.append(f"{k}:{v}")
        return " ".join(parts).lower()
    if isinstance(a, (list, tuple)):
        return " ".join(_access_text(x) for x in a)
    return ""


def access_verdict(m: Dict[str, Any]) -> Tuple[Optional[bool], str]:
    """(True = free route, False = BYOK-only / paid, None = unknown) + reason."""
    if "access" not in m or m.get("access") in (None, "", {}, []):
        return None, "unknown access (no access metadata)"
    a = _access_text(m["access"])
    free = bool(re.search(r"\b(free|pool)\b", a))
    byok = "byok" in a or "own key" in a or "own_key" in a
    priced = bool(re.search(r"\$\s*\d|credit|paid|price", a))
    if free and not priced:
        return True, f"free route ({a[:40]})"
    if byok:
        return False, "BYOK-only (needs your own provider key; $0 gateway fee is NOT free)"
    if priced:
        return False, f"paid route ({a[:40]})"
    return None, f"access does not establish a free route ({a[:40]})"


def _modalities(m: Dict[str, Any], kind: str) -> List[str]:
    arch = m.get("architecture") or {}
    vals: List[str] = []
    for src in (m.get(f"{kind}_modalities"), arch.get(f"{kind}_modalities"),
                (m.get("modalities") or {}).get(kind) if isinstance(m.get("modalities"), dict) else None):
        if isinstance(src, (list, tuple)):
            vals += [str(x).lower() for x in src]
        elif isinstance(src, str):
            vals += [x.strip().lower() for x in re.split(r"[,+ ]", src) if x.strip()]
    modality = str(arch.get("modality") or m.get("modality") or "").lower()
    if "->" in modality:
        left, right = modality.split("->", 1)
        vals += re.split(r"[+,]", left if kind == "input" else right)
    return [v.strip() for v in vals if v.strip()]


def _caps(m: Dict[str, Any]) -> List[str]:
    c = m.get("capabilities")
    if isinstance(c, dict):
        return [str(k).lower() for k, v in c.items() if v]
    if isinstance(c, (list, tuple)):
        return [str(x).lower() for x in c]
    return []


def vision_verdict(m: Dict[str, Any]) -> Tuple[bool, str]:
    ins, caps = _modalities(m, "input"), _caps(m)
    if "image" in ins or "images" in ins:
        return True, "image input modality"
    if "vision" in caps or "image" in caps or "images" in caps or m.get("supports_vision") is True:
        return True, "vision capability flag"
    return False, "no verified image input"


def _text_out(m: Dict[str, Any]) -> bool:
    outs = _modalities(m, "output")
    return not outs or "text" in outs


def _context(m: Dict[str, Any]) -> int:
    for k in ("context_length", "context_window", "max_context_length"):
        v = m.get(k) or (m.get("top_provider") or {}).get(k)
        if isinstance(v, (int, float)) and v > 0:
            return int(v)
    return 0


def _available(m: Dict[str, Any]) -> Tuple[bool, str]:
    status = str(m.get("status") or "").lower()
    if m.get("active") is False or m.get("available") is False or m.get("deprecated") is True \
            or status in ("deprecated", "disabled", "unavailable", "retired", "offline"):
        return False, "unavailable / deprecated"
    return True, ""


def classify_anyrouter(m: Dict[str, Any]) -> ModelInfo:
    mid = str(m.get("id"))
    info = ModelInfo("anyrouter", mid, context=_context(m), access=_access_text(m.get("access"))[:60],
                     can_route=m.get("can_route") if isinstance(m.get("can_route"), bool) else None,
                     pricing=_flatten_pricing(m.get("pricing")) if isinstance(m.get("pricing"), dict) else {})
    info.text = _text_out(m)
    params = [str(p).lower() for p in m.get("supported_parameters") or []]
    info.tools = "tools" in params or "tools" in _caps(m)
    info.vision, info.vision_reason = vision_verdict(m)
    ok, why = _available(m)
    if not ok:
        info.available, info.free_reason = False, why
        return info
    if mid.startswith(_ALIAS_PREFIXES) or m.get("members"):
        info.free_reason = "virtual router alias (hides which model answers) — not used"
        return info
    if info.can_route is not True:
        info.free_reason = "cannot be routed with this key now" if info.can_route is False \
            else "can_route unknown"
        return info
    acc, acc_why = access_verdict(m)
    if acc is not True:
        info.free_reason = acc_why
        return info
    price, price_why = pricing_verdict(m)
    if price is not True:
        info.free_reason = price_why
        return info
    info.free, info.free_reason = True, f"{acc_why}; {price_why}"
    return info


def classify_openrouter(m: Dict[str, Any]) -> ModelInfo:
    """OpenRouter: free = id ends with ':free' AND pricing explicitly zero (both conditions, fail closed)."""
    mid = str(m.get("id"))
    info = ModelInfo("openrouter", mid, context=_context(m), access="openrouter",
                     pricing=_flatten_pricing(m.get("pricing")) if isinstance(m.get("pricing"), dict) else {})
    info.text = _text_out(m)
    params = [str(p).lower() for p in m.get("supported_parameters") or []]
    info.tools = "tools" in params
    info.vision, info.vision_reason = vision_verdict(m)
    ok, why = _available(m)
    if not ok:
        info.available, info.free_reason = False, why
        return info
    if not mid.endswith(":free"):
        info.free_reason = "not a ':free' model id"
        return info
    price, price_why = pricing_verdict(m)
    if price is not True:
        info.free_reason = price_why
        return info
    info.free, info.free_reason = True, f"':free' id; {price_why}"
    return info


def rank_openrouter(rows: Iterable[Dict[str, Any]]) -> Tuple[List[ModelInfo], List[ModelInfo], Dict[str, str]]:
    text, vision, rejected = [], [], {}
    for r in rows:
        info = classify_openrouter(r)
        if not info.free:
            if str(r.get("id", "")).endswith(":free"):  # only report would-be free models (the list is huge)
                rejected[info.entry] = info.free_reason
            continue
        if info.vision:
            vision.append(info)
        if info.text:
            text.append(info)
    text.sort(key=lambda m: (not m.tools, m.vision, -m.context, m.id))   # text-only first, like AnyRouter
    vision.sort(key=lambda m: (-m.context, m.id))
    return text, vision, rejected


def _size_b(model_id: str) -> float:
    m = re.findall(r"(\d+(?:\.\d+)?)b\b", model_id.lower())
    return max((float(x) for x in m), default=0.0)


def rank_groq(rows: Iterable[Dict[str, Any]], preferred: Iterable[str] = ()) -> Tuple[List[ModelInfo], Dict[str, str]]:
    pref = [p.split(":", 1)[-1] for p in preferred] + list(_GROQ_PREFERRED)
    out, rejected = [], {}
    for r in rows:
        mid = str(r.get("id"))
        low = mid.lower()
        if any(h in low for h in _GROQ_EXCLUDE):
            rejected[f"groq:{mid}"] = "not a text-generation chat model"
            continue
        ok, why = _available(r)
        if not ok:
            rejected[f"groq:{mid}"] = why
            continue
        out.append(ModelInfo("groq", mid, context=_context(r), text=True, tools=True, free=True,
                             free_reason="Groq account (free tier limits apply)", access="groq"))

    def key(m: ModelInfo):
        idx = pref.index(m.id) if m.id in pref else len(pref)
        return (idx, -m.context, -_size_b(m.id), m.id)

    out.sort(key=key)
    return out, rejected


def rank_anyrouter(rows: Iterable[Dict[str, Any]]) -> Tuple[List[ModelInfo], List[ModelInfo], Dict[str, str]]:
    text, vision, rejected = [], [], {}
    for r in rows:
        info = classify_anyrouter(r)
        if not info.free:
            rejected[info.entry] = info.free_reason
            log.info("model excluded: %s — %s", info.entry, info.free_reason)
            continue
        if info.vision:
            vision.append(info)
        else:
            rejected.setdefault(f"{info.entry} (vision)", info.vision_reason)
        if info.text:
            text.append(info)
    text.sort(key=lambda m: (not m.tools, m.vision, -m.context, m.id))   # prefer tool-capable, text-only first
    vision.sort(key=lambda m: (-m.context, m.id))
    return text, vision, rejected


# =============================================================================
# Catalogue manager (process-wide)
# =============================================================================
_LOCK = threading.RLock()
_CATALOG: Optional[Catalog] = None
_HEALTH: Dict[str, Dict[str, Any]] = {}      # entry -> fails, excluded_reason, last_ok, last_error, last_check
_REFRESHING = threading.Event()


def _path(settings: Settings) -> Path:
    return Path(settings.data_dir) / "model_catalog.json"


def reset() -> None:
    global _CATALOG
    with _LOCK:
        _CATALOG = None
        _HEALTH.clear()


def _interval(settings: Settings) -> float:
    return settings.model_catalog_refresh_minutes * 60.0


def refresh(settings: Settings) -> Catalog:
    """Query both providers now (admin button / scheduler / first use). Keys never appear in errors."""
    from llm.providers import AnyRouterProvider, GroqProvider, LLMError

    cat = Catalog(refreshed_at=time.time())
    if settings.groq_api_key:
        try:
            pref = [e for e in getattr(settings, "master_models", ()) if e.startswith("groq:")]
            cat.groq, rej = rank_groq(GroqProvider(settings).models_raw(timeout=20.0), pref)
            cat.rejected.update(rej)
        except LLMError as exc:
            cat.errors["groq"] = exc.user_message
        except Exception as exc:  # pragma: no cover - unexpected payload
            cat.errors["groq"] = f"model list unreadable ({type(exc).__name__})"
    if settings.anyrouter_api_key:
        try:
            cat.any_text, cat.any_vision, rej = rank_anyrouter(AnyRouterProvider(settings).models_raw(timeout=20.0))
            cat.rejected.update(rej)
            cat.anyrouter_validated_at = time.time()
        except LLMError as exc:
            cat.errors["anyrouter"] = exc.user_message
        except Exception as exc:  # pragma: no cover
            cat.errors["anyrouter"] = f"model list unreadable ({type(exc).__name__})"
    if settings.openrouter_api_key:
        from llm.providers import OpenRouterProvider

        try:
            cat.or_text, cat.or_vision, rej = rank_openrouter(OpenRouterProvider(settings).models_raw(timeout=20.0))
            cat.rejected.update(rej)
        except LLMError as exc:
            cat.errors["openrouter"] = exc.user_message
        except Exception as exc:  # pragma: no cover
            cat.errors["openrouter"] = f"model list unreadable ({type(exc).__name__})"
    global _CATALOG
    with _LOCK:
        _CATALOG = cat
        for e in list(_HEALTH):  # a refresh gives replaced models a new chance
            _HEALTH[e]["excluded_reason"] = "" if _HEALTH[e].get("excluded_reason", "").startswith("failed") \
                else _HEALTH[e].get("excluded_reason", "")
            _HEALTH[e]["fails"] = 0
    try:
        from storage.cache import write_json_atomic

        write_json_atomic(_path(settings), cat.to_json())
    except OSError:
        log.warning("could not save the model catalogue")
    short = {"groq": len(cat.groq), "anyrouter_text": len(cat.any_text), "anyrouter_vision": len(cat.any_vision),
             "openrouter_text": len(cat.or_text), "openrouter_vision": len(cat.or_vision)}
    log.info("model catalogue refreshed: %s; errors=%s", short, sorted(cat.errors))
    return cat


def _load_disk(settings: Settings) -> Optional[Catalog]:
    from storage.cache import read_json

    d = read_json(_path(settings))
    if isinstance(d, dict) and d.get("version") == CATALOG_VERSION:
        try:
            return Catalog.from_json(d)
        except TypeError:
            return None
    return None


def catalog(settings: Settings, allow_network: bool = True) -> Catalog:
    """Current catalogue. Fresh → returned as is. Missing or older than the refresh interval → re-validated
    (one refresh, shared by concurrent callers). If re-validation fails, stale AnyRouter entries are dropped."""
    global _CATALOG
    with _LOCK:
        cat = _CATALOG or _load_disk(settings)
        _CATALOG = cat
        fresh = cat is not None and time.time() - cat.refreshed_at < _interval(settings)
        if fresh or not allow_network:
            return _fail_closed(cat, settings) if cat else Catalog()
        try:
            return refresh(settings)
        except Exception:  # pragma: no cover - refresh() already contains provider errors
            log.exception("catalogue refresh failed")
            return _fail_closed(cat, settings) if cat else Catalog()


def _fail_closed(cat: Catalog, settings: Settings) -> Catalog:
    """Never use an AnyRouter selection whose free status was not re-validated within the interval."""
    if cat.any_text or cat.any_vision:
        if time.time() - cat.anyrouter_validated_at >= _interval(settings):
            log.warning("AnyRouter selection not re-validated in time: AnyRouter models disabled until refresh")
            return Catalog(cat.version, cat.refreshed_at, cat.groq, [], [], cat.or_text, cat.or_vision,
                           cat.rejected,
                           {**cat.errors, "anyrouter": "free status not re-validated; AnyRouter models paused"},
                           cat.anyrouter_validated_at)
    return cat


# ------------------------------------------------------------------- health
def _h(entry: str) -> Dict[str, Any]:
    return _HEALTH.setdefault(entry, {"fails": 0, "excluded_reason": "", "last_ok": 0.0, "last_error": "",
                                      "last_check": 0.0, "ok": 0, "failed": 0, "latency_ms": 0.0})


def note_success(entry: str, ms: float) -> None:
    with _LOCK:
        h = _h(entry)
        h.update(fails=0, last_ok=time.time(), last_check=time.time())
        h["ok"] += 1
        h["latency_ms"] = ms if not h["latency_ms"] else 0.7 * h["latency_ms"] + 0.3 * ms


def note_failure(entry: str, kind: str, settings: Settings) -> None:
    with _LOCK:
        h = _h(entry)
        h["fails"] += 1
        h["failed"] += 1
        h.update(last_error=kind, last_check=time.time())
        if kind in ("model", "paid", "unsupported_vision"):
            exclude(entry, {"model": "model not found (404)", "paid": "asked for credits (not free)",
                            "unsupported_vision": "rejected image input"}[kind])
        elif h["fails"] >= settings.model_failure_exclude:
            exclude(entry, f"failed {h['fails']} times in a row")


def exclude(entry: str, reason: str) -> None:
    with _LOCK:
        _h(entry)["excluded_reason"] = reason
    log.warning("model excluded until next catalogue refresh: %s — %s", entry, reason)


def excluded(entry: str) -> str:
    with _LOCK:
        return (_HEALTH.get(entry) or {}).get("excluded_reason", "")


def health() -> Dict[str, Dict[str, Any]]:
    with _LOCK:
        return {k: dict(v) for k, v in _HEALTH.items()}


# ------------------------------------------------------------------- chains
@dataclass
class Selection:
    text: List[Tuple[str, str]]        # (slot label, entry)
    vision: List[Tuple[str, str]]
    shortages: List[str]
    groups: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)   # placeholder -> slots


def select(settings: Settings, cat: Optional[Catalog] = None) -> Selection:
    """Slots filled from the ranked candidates, skipping excluded models (automatic replacement)."""
    cat = cat or catalog(settings)
    groq = [m for m in cat.groq if not excluded(m.entry)][:GROQ_SLOTS]
    vision = [m for m in cat.any_vision if not excluded(m.entry)][:ANY_VISION_SLOTS]
    vision_ids = {m.id for m in vision}
    text_pool = [m for m in cat.any_text if not excluded(m.entry)]
    any_text = [m for m in text_pool if m.id not in vision_ids][:ANY_TEXT_SLOTS]   # distinct from vision if possible
    if len(any_text) < ANY_TEXT_SLOTS:
        any_text += [m for m in text_pool if m not in any_text][:ANY_TEXT_SLOTS - len(any_text)]
    or_vision = [m for m in cat.or_vision if not excluded(m.entry)][:OR_VISION_SLOTS]
    or_vision_ids = {m.id for m in or_vision}
    or_pool = [m for m in cat.or_text if not excluded(m.entry)]
    or_text = [m for m in or_pool if m.id not in or_vision_ids][:OR_TEXT_SLOTS]   # distinct if possible
    if len(or_text) < OR_TEXT_SLOTS:
        or_text += [m for m in or_pool if m not in or_text][:OR_TEXT_SLOTS - len(or_text)]
    labels = ["Groq primary", "Groq fallback 1", "Groq fallback 2", "Groq fallback 3"]
    text = [(labels[i], m.entry) for i, m in enumerate(groq)]
    text += [(f"AnyRouter text fallback {i + 1}", m.entry) for i, m in enumerate(any_text)]
    vis = [(f"AnyRouter vision fallback {i + 1}", m.entry) for i, m in enumerate(vision)]
    extra = {"@groq": [(labels[i], m.entry) for i, m in enumerate(groq)],
             "@anyrouter_text": [(f"AnyRouter text fallback {i + 1}", m.entry) for i, m in enumerate(any_text)],
             "@anyrouter_vision": vis,
             "@openrouter_text": [(f"OpenRouter free text {i + 1}", m.entry) for i, m in enumerate(or_text)],
             "@openrouter_vision": [(f"OpenRouter free vision {i + 1}", m.entry) for i, m in enumerate(or_vision)]}
    short = []
    if settings.groq_api_key and len(groq) < GROQ_SLOTS:
        short.append(f"only {len(groq)} of {GROQ_SLOTS} Groq text models available")
    if settings.anyrouter_api_key and len(any_text) < ANY_TEXT_SLOTS:
        short.append(f"only {len(any_text)} of {ANY_TEXT_SLOTS} verified-free AnyRouter text models available")
    if settings.anyrouter_api_key and len(vision) < ANY_VISION_SLOTS:
        short.append(f"only {len(vision)} of {ANY_VISION_SLOTS} verified-free AnyRouter vision models available")
    for s in short:
        log.warning("model shortage: %s", s)
    return Selection(text, vis, short, extra)


def info_for(entry: str, settings: Settings) -> Optional[ModelInfo]:
    cat = _CATALOG or _load_disk(settings)
    if not cat:
        return None
    for m in cat.groq + cat.any_text + cat.any_vision + cat.or_text + cat.or_vision:
        if m.entry == entry:
            return m
    return None


def verified_free(entry: str, settings: Settings) -> bool:
    """Cost guard used by the router: AnyRouter entries must be in the (re-validated) free catalogue."""
    if not entry.startswith("anyrouter:"):
        return True
    cat = catalog(settings, allow_network=False)
    return any(m.entry == entry and m.free for m in cat.any_text + cat.any_vision) and not excluded(entry)


def verified_vision(entry: str, settings: Settings) -> bool:
    cat = catalog(settings, allow_network=False)
    return any(m.entry == entry and m.vision for m in cat.any_vision) and not excluded(entry)


def chains(settings: Settings) -> Dict[str, Tuple[str, ...]]:
    sel = select(settings)
    text = tuple(e for _, e in sel.text)
    return {"master": text, "fast": text, "verifier": text, "vision": tuple(e for _, e in sel.vision)}


def expand(entries: Iterable[str], settings: Settings) -> Tuple[str, ...]:
    """Replace chain placeholders (@groq, @anyrouter_text, @anyrouter_vision, @openrouter_text,
    @openrouter_vision) with the discovered slots, in the order written; explicit 'provider:model' entries are
    kept. Unknown placeholders are dropped with a warning."""
    entries = list(entries)
    if not any(e.startswith("@") for e in entries):
        return tuple(entries)
    groups = select(settings).groups
    out: List[str] = []
    for e in entries:
        if e.startswith("@"):
            if e not in groups:
                log.warning("unknown model chain placeholder %s (use one of %s)", e, ", ".join(TOKENS))
                continue
            out += [entry for _, entry in groups[e]]
        else:
            out.append(e)
    return tuple(dict.fromkeys(out))


def configured_slots(settings: Settings) -> Dict[str, List[Tuple[str, str]]]:
    """Text and vision slots as configured (placeholders expanded) — for the admin dashboard."""
    sel = select(settings)
    labels = {e: lbl for grp in sel.groups.values() for lbl, e in grp}

    def slots(role: str, default: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        raw = tuple(getattr(settings, f"{role}_models", ()) or ())
        if not raw:
            return default
        return [(labels.get(e, "configured"), e) for e in expand(raw, settings)]

    return {"text": slots("master", sel.text), "vision": slots("vision", sel.vision)}


def summary(settings: Settings) -> Dict[str, Any]:
    """Everything the admin page needs (no secrets)."""
    slots = configured_slots(settings)          # may perform the first discovery
    cat = catalog(settings, allow_network=False)
    sel = select(settings, cat)
    hl = health()
    infos = {m.entry: m for m in cat.groq + cat.any_text + cat.any_vision + cat.or_text + cat.or_vision}

    def row(slot: str, entry: str) -> Dict[str, Any]:
        m, h = infos.get(entry), hl.get(entry, {})
        return {"slot": slot, "model": entry, "available": bool(m and m.available) and not h.get("excluded_reason"),
                "free verified": bool(m and m.free), "vision": bool(m and m.vision), "context": m.context if m else 0,
                "last health check": h.get("last_check", 0.0), "ok": h.get("ok", 0), "failed": h.get("failed", 0),
                "latency ms": round(h.get("latency_ms", 0.0), 1), "last error": h.get("last_error", ""),
                "why free": m.free_reason if m else ""}

    return {"refreshed_at": cat.refreshed_at, "anyrouter_validated_at": cat.anyrouter_validated_at,
            "text": [row(s, e) for s, e in slots["text"]], "vision": [row(s, e) for s, e in slots["vision"]],
            "shortages": sel.shortages, "errors": dict(cat.errors),
            "rejected": dict(sorted(cat.rejected.items())),
            "excluded": {e: h["excluded_reason"] for e, h in hl.items() if h.get("excluded_reason")}}


def save_json_safe(obj: Any) -> str:  # helper for the UI download (no secrets inside)
    return json.dumps(obj, indent=2, default=str)
