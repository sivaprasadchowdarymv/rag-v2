"""
Reranking: Qwen3-Reranker through OpenRouter, with a local cross-encoder fallback.

    rerank(query, texts)        candidate ranking for retrieval (one remote call per question)
    support_scores(sent, texts) claim-support check for the verifier (many small calls,
                                so it prefers the LOCAL model: ~ms each, no cost)

Backends (RERANK_BACKEND):
    auto        OpenRouter when a key is set, otherwise local; any remote failure
                falls back to local for that call, and the app keeps working.
    openrouter  remote only (falls back to local on error)
    local       fastembed TextCrossEncoder (ms-marco-MiniLM-L-6-v2, ONNX on CPU, ~80 MB)

Remote scores are relevance probabilities (0..1); local scores are logits.
They are never mixed within one ranking. For the support check both are mapped
onto the same decision threshold (see support_scores).
"""
from __future__ import annotations

import threading
import time
from typing import List, Optional, Sequence, Tuple

from config.settings import Settings, get_logger

log = get_logger("reranker")
_lock = threading.Lock()
MAX_DOC_CHARS = 2000


def _build(model: str, cache_dir: str, local_only: bool):
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    log.info("Loading local reranker %s%s", model, " (cache only)" if local_only else "")
    return TextCrossEncoder(model_name=model, cache_dir=cache_dir, local_files_only=local_only)


def _load(model: str, cache_dir: str, blocking: bool = False):
    """Never stalls a question: cache-only load, background download when missing (rag/model_failures.py)."""
    from rag.model_failures import get_model

    return get_model(f"rerank:{model}", lambda local_only: _build(model, cache_dir, local_only), blocking=blocking)


def warm_local(settings: Settings) -> None:
    with _lock:
        _load(settings.rerank_model, str(settings.data_dir / "models"), blocking=True)


def _local(query: str, texts: Sequence[str], settings: Settings) -> List[float]:
    with _lock:
        model = _load(settings.rerank_model, str(settings.data_dir / "models"))
    return [float(s) for s in model.rerank(query, [t[:1500] for t in texts])]


def _remote(query: str, texts: Sequence[str], settings: Settings) -> List[float]:
    from llm.providers import OpenRouterProvider

    return OpenRouterProvider(settings).rerank(settings.rerank_model_remote, query,
                                               [t[:MAX_DOC_CHARS] or " " for t in texts])


def rerank(query: str, texts: Sequence[str], settings: Settings) -> Tuple[Optional[List[float]], float, str]:
    """Return (scores or None, latency_ms, warning)."""
    if not settings.rerank or not texts:
        return None, 0.0, ""
    t0 = time.perf_counter()
    if settings.remote_rerank:
        try:
            return _remote(query, texts, settings), (time.perf_counter() - t0) * 1000, ""
        except Exception as exc:  # LLMError or anything unexpected: fall back to local
            log.warning("Remote reranking failed (%s); using the local cross-encoder",
                        getattr(exc, "kind", type(exc).__name__))
    try:
        return _local(query, texts, settings), (time.perf_counter() - t0) * 1000, ""
    except Exception as exc:
        log.warning("Reranking skipped: %s", exc)
        return None, (time.perf_counter() - t0) * 1000, "Reranking was skipped (model unavailable); results use hybrid ranking."


def support_scores(sentence: str, texts: Sequence[str], settings: Settings) -> List[float]:
    """Scores on the LOCAL threshold scale (settings.support_threshold).

    Local logits are returned as they are. If the local model is unavailable,
    remote relevance r is mapped so that r == support_threshold_remote lands
    exactly on support_threshold."""
    try:
        return _local(sentence, texts, settings)
    except Exception:
        if not settings.has_openrouter:
            raise
    remote = _remote(sentence, texts, settings)
    shift = settings.support_threshold - settings.support_threshold_remote
    return [r + shift for r in remote]
