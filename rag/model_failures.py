"""
Local model loading that never stalls a question.

fastembed retries a blocked or offline download for ~40 s. To keep questions fast:
  * query path (embed a question, rerank, claim-support, relevance): load from the local cache only
    (`local_files_only=True`, ~0.5 s even when missing). If the model is not cached, ONE background download is
    started and the request continues with keyword search / RRF order;
  * ingestion path (embedding a new document): a blocking download is allowed, unless a download failed within
    MODEL_RETRY_S (default 600 s);
  * MODEL_DOWNLOAD=false never downloads (air-gapped installs with pre-seeded data/models).
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable, Dict, Set

from config.settings import get_logger

log = get_logger("models")
_FAILED: Dict[str, float] = {}
_READY: Dict[str, Any] = {}
_DOWNLOADING: Set[str] = set()
_LOCK = threading.Lock()


def retry_after_s() -> float:
    try:
        return float(os.getenv("MODEL_RETRY_S", "600"))
    except ValueError:
        return 600.0


def downloads_allowed() -> bool:
    return os.getenv("MODEL_DOWNLOAD", "true").strip().lower() not in ("0", "false", "no", "off")


def recently_failed(key: str) -> bool:
    with _LOCK:
        t = _FAILED.get(key)
        return t is not None and time.time() - t < retry_after_s()


def mark_failed(key: str) -> None:
    with _LOCK:
        _FAILED[key] = time.time()


def clear() -> None:
    with _LOCK:
        _FAILED.clear()
        _READY.clear()
        _DOWNLOADING.clear()


def _download(key: str, factory: Callable[[bool], Any]) -> None:
    try:
        model = factory(False)
        with _LOCK:
            _READY[key] = model
        log.info("model %s downloaded and ready", key)
    except Exception as exc:
        mark_failed(key)
        log.warning("model %s could not be downloaded (%s); keyword search stays active", key, type(exc).__name__)
    finally:
        with _LOCK:
            _DOWNLOADING.discard(key)


def get_model(key: str, factory: Callable[[bool], Any], blocking: bool = False) -> Any:
    """`factory(local_only)` builds the model. Raises RuntimeError at once if it is not available yet."""
    with _LOCK:
        if key in _READY:
            return _READY[key]
    try:
        model = factory(True)  # cached files only: fast success or fast failure
        with _LOCK:
            _READY[key] = model
        return model
    except Exception:
        pass
    if not downloads_allowed() or recently_failed(key):
        raise RuntimeError(f"model {key} is not available locally")
    if blocking:
        with _LOCK:
            busy = key in _DOWNLOADING
        if not busy:
            _download(key, factory)
            with _LOCK:
                if key in _READY:
                    return _READY[key]
        raise RuntimeError(f"model {key} could not be loaded")
    with _LOCK:
        start = key not in _DOWNLOADING
        if start:
            _DOWNLOADING.add(key)
    if start:
        threading.Thread(target=_download, args=(key, factory), name=f"download-{key}", daemon=True).start()
    raise RuntimeError(f"model {key} is downloading in the background")
