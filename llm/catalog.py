"""
Model catalog: which model plays which role.

The rule of this app: **text goes to Groq (free, fast), images go to Qwen / Gemma (OpenRouter).**

    master     writes the cited answer for TEXT questions          -> Groq gpt-oss-120b
    fast       tiny jobs (abbreviation resolution)                 -> Groq gpt-oss-20b
    verifier   optional LLM cross-check (off by default: tokens)    -> Groq Llama 3.3 70B (other family)
    vision     anything that needs to SEE an image:                -> OpenRouter Gemma 4 / Qwen 3.8
               * questions about figures, graphs, diagrams (the figure images are attached)
               * OCR of scanned pages while indexing
               * optional figure descriptions while indexing (VISION_INDEX, off by default)

Every chain entry is "provider:model". Entries whose provider has no key are
skipped, and the router walks the chain in order on recoverable errors
(rate limit, model down, no credits). The last text fallback is a free Gemma
on OpenRouter, so answers keep coming when Groq's daily free quota runs out.

Profiles (MODEL_PROFILE):
    free     Groq free tier for text + Gemma 4 ":free" for images -> $0
    quality  Groq for text + Qwen 3.8 (paid, cents) first for images -> better image reading

Override any chain with MASTER_MODELS / FAST_MODELS / VERIFIER_MODELS / VISION_MODELS,
e.g. VISION_MODELS=openrouter:qwen/qwen3.8-flash,openrouter:google/gemma-4-31b-it:free
"""
from __future__ import annotations

from typing import Dict, Tuple

ROLES = ("master", "fast", "verifier", "vision")
PROVIDERS = ("groq", "openrouter", "ollama", "openai", "anthropic", "bedrock", "anyrouter")
LOCAL_PROVIDERS = frozenset({"ollama"})
VISION_CAPABLE = ("openrouter", "openai", "anthropic", "bedrock")  # + Ollama when OLLAMA_VISION_MODEL is set
# AnyRouter is vision-capable only per model, when discovery verified an image input (llm/discovery.py)

# Groq production models (console.groq.com/docs/models, November 2026)
GROQ_120B = "groq:openai/gpt-oss-120b"
GROQ_20B = "groq:openai/gpt-oss-20b"
GROQ_LLAMA70 = "groq:llama-3.3-70b-versatile"
GROQ_LLAMA8 = "groq:llama-3.1-8b-instant"
# OpenRouter multimodal models (openrouter.ai, November 2026)
GEMMA_31B_FREE = "openrouter:google/gemma-4-31b-it:free"
GEMMA_26B_FREE = "openrouter:google/gemma-4-26b-a4b-it:free"
GEMMA_31B = "openrouter:google/gemma-4-31b-it"
QWEN_FLASH = "openrouter:qwen/qwen3.8-flash"
QWEN_27B = "openrouter:qwen/qwen3.8-27b"

_TEXT = {
    "master": (GROQ_120B, GROQ_LLAMA70, GEMMA_31B_FREE),
    "fast": (GROQ_20B, GROQ_LLAMA8),
    "verifier": (GROQ_LLAMA70, GROQ_20B),
}
PROFILES: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "free": {**_TEXT, "vision": (GEMMA_31B_FREE, GEMMA_26B_FREE, QWEN_27B)},
    "quality": {**_TEXT, "vision": (QWEN_FLASH, GEMMA_31B, GEMMA_31B_FREE)},
}
PROFILE_LABELS = {
    "free": "🆓 Free - Groq for text, Gemma 4 free for images ($0)",
    "quality": "🖼️ Quality images - Groq for text, Qwen 3.8 for images (cents)",
}
DEFAULT_PROFILE = "free"

# Optional OpenRouter retrieval models (EMBED_BACKEND / RERANK_BACKEND = openrouter). Default is local = 0 tokens.
EMBED_MODEL = "qwen/qwen3-embedding-8b"
RERANK_MODEL = "qwen/qwen3-reranker-8b"
EMBED_QUERY_INSTRUCTION = ("Instruct: Given a question about a technical document, retrieve the passages, "
                           "table rows, equations or figure descriptions that answer it\nQuery: ")


def chain_for(profile: str, role: str) -> Tuple[str, ...]:
    return PROFILES.get(profile, PROFILES[DEFAULT_PROFILE]).get(role, ())


def chain_for_settings(settings, role: str) -> Tuple[str, ...]:
    """Full candidate chain for a role: explicit override (or profile default) + configured extra providers.
    Order = priority (user configuration). The router then filters by mode, sensitivity, health and context."""
    override = tuple(getattr(settings, f"{role}_models", ()) or ())
    discovery = bool(getattr(settings, "discovery_enabled", False))
    if override and any(e.startswith("@") for e in override):
        if discovery:  # @groq, @anyrouter_text, @anyrouter_vision, @openrouter_text, @openrouter_vision
            from llm.discovery import expand

            override = expand(override, settings)
        else:
            override = tuple(e for e in override if not e.startswith("@"))
    if discovery and not override:
        # 2.2: independent TEXT (Groq ×4 → AnyRouter free ×2) and VISION (AnyRouter free ×3) chains from
        # the providers' live catalogues (llm/discovery.py). Paid providers only with explicit approval.
        from llm.discovery import chains

        base = list(chains(settings).get(role, ()))
    else:
        base = list(override or chain_for(settings.model_profile, role))
    paid_ok = not discovery or (getattr(settings, "model_allow_paid_fallback", False)
                                and not getattr(settings, "model_free_only", True))
    if not override:
        cloud = {"openai": f"openai:{settings.openai_model}", "anthropic": f"anthropic:{settings.anthropic_model}",
                 "bedrock": f"bedrock:{settings.bedrock_model}"}
        for name, entry in cloud.items():
            if settings.provider_configured(name) and paid_ok:
                base.append(entry)
        if settings.provider_configured("ollama"):
            local = {"vision": settings.ollama_vision_model,
                     "fast": settings.ollama_fast_model or settings.ollama_model}.get(role, settings.ollama_model)
            if local:
                base.append(f"ollama:{local}")
    manual = (getattr(settings, "manual_model", "") or "").strip()
    if manual:
        base = [manual] + [e for e in base if e != manual]
    return tuple(dict.fromkeys(base))


def is_local(entry: str) -> bool:
    return split(entry)[0] in LOCAL_PROVIDERS


def split(entry: str) -> Tuple[str, str]:
    """'groq:openai/gpt-oss-120b' -> ('groq', 'openai/gpt-oss-120b'). Unprefixed ids default to OpenRouter."""
    head, sep, rest = entry.partition(":")
    if sep and head in PROVIDERS:
        return head, rest
    return "openrouter", entry


def is_free(entry: str) -> bool:
    provider, model = split(entry)
    return provider in ("groq", "ollama") or model.endswith(":free")


PAID_PROVIDERS = frozenset({"openai", "anthropic", "bedrock"})


def cost_guard(entry: str, settings) -> str:
    """Central free-only guard (discovery routing). "" = allowed, otherwise the reason it is blocked.
    Groq and local Ollama are free to this app; AnyRouter only when the re-validated catalogue proves a free
    route; OpenRouter only ':free' ids; OpenAI / Anthropic / Bedrock only with MODEL_ALLOW_PAID_FALLBACK=true
    AND MODEL_FREE_ONLY=false."""
    provider, model = split(entry)
    if not getattr(settings, "discovery_enabled", False):
        return ""
    paid_approved = getattr(settings, "model_allow_paid_fallback", False) and \
        not getattr(settings, "model_free_only", True)
    if provider == "anyrouter":
        from llm.discovery import verified_free

        return "" if verified_free(entry, settings) else "cost guard: not a verified free AnyRouter route"
    if provider in ("groq", "ollama") or paid_approved:
        return ""
    if provider == "openrouter" and model.endswith(":free"):
        return ""
    return "cost guard: paid model (MODEL_FREE_ONLY=true / MODEL_ALLOW_PAID_FALLBACK=false)"


def short_name(entry: str) -> str:
    """'groq:openai/gpt-oss-120b' -> 'gpt-oss-120b' (compact UI labels)."""
    return split(entry)[1].split("/", 1)[-1]
