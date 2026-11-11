"""
Embeddings: Qwen3-Embedding-8B through OpenRouter, or local fastembed.

    remote  "openrouter:qwen/qwen3-embedding-8b@1024"   (default when OPENROUTER_API_KEY is set)
            state-of-the-art multilingual retrieval embeddings; Matryoshka vectors
            truncated to EMBED_DIMS; instruction prefix on queries only.
            Batches of 64 texts, 4 batches in parallel. Costs about $0.01 per
            million tokens (a 100-page datasheet is well under one cent).
    local   nomic-embed-text-v1.5 (quantised ONNX on CPU, ~130 MB, 768 dims),
            downloaded once. Used with EMBED_BACKEND=local or without a key.

The model id is stored on every index (DocumentIndex.embed_model) and is part
of the cache key, so query and document vectors always come from the same
model. Vectors are stored L2-normalised in one matrix, so cosine similarity for
all nodes is a single matrix-vector product (see retrieval.py).
"""
from __future__ import annotations

import gc
import os
import threading
from collections import OrderedDict
from typing import Callable, List, Optional

import numpy as np

from config.settings import Settings, get_logger
from rag.models import DocumentIndex
from storage.vector_store import normalize_rows

log = get_logger("embeddings")

EMBED_CHAR_LIMIT = 2000  # unchanged from the original RAG
# Small batches + capped threads keep peak RAM low on free hosts (Streamlit Cloud ~2.7 GB);
# a 10-page paper can produce 800+ chunks. Override with EMBED_BATCH_SIZE / EMBED_THREADS.
EMBED_BATCH_SIZE = max(1, int(os.getenv("EMBED_BATCH_SIZE", "8") or 8))
EMBED_THREADS = max(1, int(os.getenv("EMBED_THREADS", "2") or 2))

ProgressFn = Optional[Callable[[float, str], None]]


REMOTE_PREFIX = "openrouter:"
OLLAMA_PREFIX = "ollama:"
REMOTE_BATCH = 64
REMOTE_WORKERS = 4


class EmbeddingError(Exception):
    def __init__(self, model: str, detail: str = ""):
        if model.startswith(OLLAMA_PREFIX):
            self.user_message = (f"The local embedding model '{model[len(OLLAMA_PREFIX):]}' on Ollama could not be "
                                 f"reached{': ' + detail if detail else ''}. Keyword search still works.")
        elif model.startswith(REMOTE_PREFIX):
            self.user_message = (
                f"The embedding model '{model[len(REMOTE_PREFIX):].split('@')[0]}' on OpenRouter could not be "
                f"reached{': ' + detail if detail else ''}. Semantic search is off for now; keyword search "
                "still works, and missing vectors are filled in automatically next time."
            )
        else:
            self.user_message = (
                f"The embedding model '{model.split('/')[-1]}' could not be loaded "
                "(it downloads once, about 130 MB). Semantic search is off; keyword "
                "search still works. Please try again in a minute."
            )
        super().__init__(self.user_message)


def active_model(settings: Settings) -> str:
    """Embedding model id for NEW indexes and their queries."""
    if settings.embed_backend == "ollama" and settings.ollama_base_url:
        return f"{OLLAMA_PREFIX}{settings.ollama_embed_model}"
    if settings.remote_embeddings:
        return f"{REMOTE_PREFIX}{settings.embed_model_remote}@{settings.embed_dims}"
    return settings.embed_model


def is_remote(model: str) -> bool:
    return model.startswith(REMOTE_PREFIX)


def _parse_remote(model: str):
    name, _, dims = model[len(REMOTE_PREFIX):].partition("@")
    return name, (int(dims) if dims.isdigit() else None)


# Nomic models are trained with task prefixes; using them improves retrieval.
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

_load_lock = threading.Lock()
_nolock = __import__("contextlib").nullcontext()


def _build(model_name: str, cache_dir: str, local_only: bool):
    from fastembed import TextEmbedding  # heavy import: only when needed

    log.info("Loading embedding model %s%s", model_name, " (cache only)" if local_only else "")
    # kSameAsRequested stops onnxruntime's memory arena from doubling on every growth
    providers = [("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})]
    try:
        return TextEmbedding(model_name=model_name, cache_dir=cache_dir, threads=EMBED_THREADS, providers=providers,
                             local_files_only=local_only)
    except TypeError:  # option not supported by this fastembed/onnxruntime: fall back to defaults
        return TextEmbedding(model_name=model_name, cache_dir=cache_dir, local_files_only=local_only)


def _model(model_name: str, cache_dir: str, blocking: bool = False):
    """Query path: cache-only (never stalls); ingestion path (blocking=True): may download once."""
    from rag.model_failures import get_model

    with _load_lock if blocking else _nolock:
        return get_model(f"embed:{model_name}", lambda local_only: _build(model_name, cache_dir, local_only),
                         blocking=blocking)


def _uses_prefixes(model: str) -> bool:
    return "nomic" in model.lower()


def embedding_signature(settings: Settings, model: Optional[str] = None) -> str:
    """Identifies how vectors were made; part of the embedding cache key."""
    model = model or active_model(settings)
    if is_remote(model):
        return f"{model}|instr-v1"
    if model.startswith(OLLAMA_PREFIX):
        return f"{model}" + ("|prefix-v1" if _uses_prefixes(model) else "")
    return f"fastembed:{model}" + ("|prefix-v1" if _uses_prefixes(model) else "")


def warm_up(settings: Settings) -> None:
    """Start loading the LOCAL models in the background when the app opens.

    The local cross-encoder is always useful (claim-support checks, reranker
    fallback); the local embedding model only when it is the active one."""

    def _load() -> None:
        try:
            if not settings.remote_embeddings and settings.embed_backend != "ollama":
                _model(settings.embed_model, str(settings.data_dir / "models"), blocking=True)
            if settings.rerank:
                from retrieval.reranker import warm_local

                warm_local(settings)
        except Exception as exc:  # reported properly when actually used
            log.warning("Background model load failed: %s", exc)

    threading.Thread(target=_load, name="embed-warmup", daemon=True).start()


def _embed_remote(texts: List[str], settings: Settings, model: str, kind: str) -> List[List[float]]:
    from llm.catalog import EMBED_QUERY_INSTRUCTION
    from llm.providers import LLMError, OpenRouterProvider

    name, dims = _parse_remote(model)
    if kind == "query":
        texts = [EMBED_QUERY_INSTRUCTION + t for t in texts]
    provider = OpenRouterProvider(settings)
    if not provider.configured():
        raise EmbeddingError(model, "no OPENROUTER_API_KEY")
    batches = [texts[i: i + REMOTE_BATCH] for i in range(0, len(texts), REMOTE_BATCH)]

    def run(batch: List[str]) -> List[List[float]]:
        vecs = provider.embed(name, batch, dims).vectors
        return [v[:dims] if dims else v for v in vecs]  # Matryoshka: the first `dims` values are a valid embedding

    try:
        if len(batches) == 1:
            return run(batches[0])
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=REMOTE_WORKERS) as pool:
            return [v for part in pool.map(run, batches) for v in part]
    except LLMError as exc:
        log.warning("Remote embedding failed: %s", exc.kind)
        raise EmbeddingError(model, exc.user_message.rstrip(".")) from exc


def _embed_ollama(texts: List[str], settings: Settings, model: str, kind: str) -> List[List[float]]:
    from llm.adapters import OllamaProvider
    from llm.providers import LLMError

    name = model[len(OLLAMA_PREFIX):]
    if _uses_prefixes(name):
        texts = [(QUERY_PREFIX if kind == "query" else DOC_PREFIX) + t for t in texts]
    provider = OllamaProvider(settings)
    if not provider.configured():
        raise EmbeddingError(model, "OLLAMA_BASE_URL is not set")
    try:
        return [v for i in range(0, len(texts), REMOTE_BATCH)
                for v in provider.embed(name, texts[i:i + REMOTE_BATCH]).vectors]
    except LLMError as exc:
        raise EmbeddingError(model, exc.user_message.rstrip(".")) from exc


def embed_texts(texts: List[str], settings: Settings, model: str, kind: str = "document") -> List[List[float]]:
    if model.startswith(OLLAMA_PREFIX):
        return _embed_ollama(texts, settings, model, kind)
    if is_remote(model):
        return _embed_remote(texts, settings, model, kind)
    if _uses_prefixes(model):
        prefix = QUERY_PREFIX if kind == "query" else DOC_PREFIX
        texts = [prefix + t for t in texts]
    try:
        engine = _model(model, str(settings.data_dir / "models"), blocking=kind == "document")
        return [np.asarray(v, dtype=np.float32).tolist() for v in engine.embed(texts, batch_size=EMBED_BATCH_SIZE)]
    except Exception as exc:
        log.exception("Embedding failed")
        raise EmbeddingError(model) from exc


_QVEC: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_QVEC_MAX = 512
QVEC_STATS = {"hits": 0, "misses": 0}


def embed_query(query: str, settings: Settings, model: Optional[str] = None) -> np.ndarray:
    model = model or active_model(settings)
    key = (model, query[:EMBED_CHAR_LIMIT])
    cached = _QVEC.get(key)
    if cached is not None:
        _QVEC.move_to_end(key)
        QVEC_STATS["hits"] += 1
        return cached
    QVEC_STATS["misses"] += 1
    vec = embed_texts([query[:EMBED_CHAR_LIMIT]], settings, model, kind="query")[0]
    out = normalize_rows(np.asarray([vec], dtype=np.float32))[0]
    out.setflags(write=False)
    _QVEC[key] = out
    while len(_QVEC) > _QVEC_MAX:
        _QVEC.popitem(last=False)
    return out


def fill_missing_embeddings(index: DocumentIndex, settings: Settings, progress: ProgressFn = None) -> int:
    """Embed nodes that have no embedding yet (in place). Returns how many."""
    n = len(index.nodes)
    if len(index.has_embedding) != n:
        index.has_embedding = np.zeros(n, dtype=bool)
    todo = [i for i in range(n) if not index.has_embedding[i] and index.nodes[i].content]
    if not todo:
        return 0
    done = 0
    step = REMOTE_BATCH * REMOTE_WORKERS if is_remote(index.embed_model) else EMBED_BATCH_SIZE
    for start in range(0, len(todo), step):
        batch = todo[start : start + step]
        vectors = embed_texts([index.nodes[i].content[:EMBED_CHAR_LIMIT] for i in batch], settings, index.embed_model)
        matrix = normalize_rows(np.asarray(vectors, dtype=np.float32))
        if index.embeddings.shape != (n, matrix.shape[1]):
            index.embeddings = np.zeros((n, matrix.shape[1]), dtype=np.float32)
            index.has_embedding[:] = False
        index.embeddings[batch] = matrix
        index.has_embedding[batch] = True
        done += len(batch)
        del vectors, matrix
        if done % (EMBED_BATCH_SIZE * 16) == 0 or is_remote(index.embed_model):
            gc.collect()
        if progress:
            progress(done / len(todo), f"Embedding {done}/{len(todo)} chunks")
    log.info("Embedding completed: %d nodes", done)
    return done
