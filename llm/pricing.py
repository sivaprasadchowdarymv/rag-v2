"""
Cost estimates (USD per 1M tokens: input, output) and context windows for cost-aware routing.

Prices are list prices at the time of writing and are ESTIMATES used for routing
and dashboards; OpenRouter reports the real billed cost (usage.cost), which is
used instead whenever present. Override or extend with LLM_PRICES_JSON, e.g.
    LLM_PRICES_JSON={"openai:gpt-4o-mini": [0.15, 0.6, 128000]}
"""
from __future__ import annotations

import json
import os
from typing import Dict, Tuple

# entry -> (input $/1M, output $/1M, context window tokens)
_TABLE: Dict[str, Tuple[float, float, int]] = {
    "openai:gpt-4o-mini": (0.15, 0.60, 128_000),
    "openai:gpt-4o": (2.50, 10.0, 128_000),
    "openai:gpt-4.1": (2.00, 8.00, 1_000_000),
    "openai:gpt-4.1-mini": (0.40, 1.60, 1_000_000),
    "anthropic:claude-haiku-4-5": (1.00, 5.00, 200_000),
    "anthropic:claude-sonnet-4-5": (3.00, 15.0, 200_000),
    "bedrock:anthropic.claude-3-5-haiku-20241022-v1:0": (0.80, 4.00, 200_000),
    "groq:openai/gpt-oss-120b": (0.0, 0.0, 131_072),   # free tier
    "groq:openai/gpt-oss-20b": (0.0, 0.0, 131_072),
    "groq:llama-3.3-70b-versatile": (0.0, 0.0, 131_072),
    "groq:llama-3.1-8b-instant": (0.0, 0.0, 131_072),
}
_DEFAULT_CTX = {"ollama": 8_192, "groq": 131_072, "openrouter": 128_000, "openai": 128_000,
                "anthropic": 200_000, "bedrock": 200_000}


def _table() -> Dict[str, Tuple[float, float, int]]:
    raw = os.getenv("LLM_PRICES_JSON", "")
    if not raw:
        return _TABLE
    try:
        extra = {k: (float(v[0]), float(v[1]), int(v[2])) for k, v in json.loads(raw).items()}
        return {**_TABLE, **extra}
    except (ValueError, TypeError, IndexError, KeyError):
        return _TABLE


def price(provider: str, model: str) -> Tuple[float, float]:
    if provider == "ollama" or model.endswith(":free"):
        return 0.0, 0.0
    row = _table().get(f"{provider}:{model}")
    return (row[0], row[1]) if row else (1.0, 3.0)  # unknown paid model: conservative estimate


def estimate_cost(provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pin, pout = price(provider, model)
    return round((prompt_tokens * pin + completion_tokens * pout) / 1_000_000, 8)


def context_window(provider: str, model: str) -> int:
    row = _table().get(f"{provider}:{model}")
    if row:
        return row[2]
    if provider == "ollama":
        return int(os.getenv("OLLAMA_NUM_CTX", _DEFAULT_CTX["ollama"]))
    return _DEFAULT_CTX.get(provider, 32_000)


def blended(provider: str, model: str) -> float:
    pin, pout = price(provider, model)
    return pin * 0.8 + pout * 0.2  # typical RAG call: mostly input tokens
