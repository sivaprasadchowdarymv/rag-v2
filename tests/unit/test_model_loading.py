"""Model loading never stalls a question: cache-only on the query path, one background download."""
from __future__ import annotations

import threading
import time

import pytest

from rag import model_failures as mf


def test_query_path_fails_fast_and_downloads_in_background():
    calls = []
    done = threading.Event()

    def factory(local_only):
        calls.append(local_only)
        if local_only:
            raise ValueError("not cached")
        time.sleep(0.3)  # simulated download
        done.set()
        return "MODEL"

    t0 = time.perf_counter()
    with pytest.raises(RuntimeError, match="background"):
        mf.get_model("embed:x", factory)
    assert time.perf_counter() - t0 < 0.2  # did not wait for the download
    with pytest.raises(RuntimeError):
        mf.get_model("embed:x", factory)  # second request: no second download started
    assert done.wait(2) and calls.count(False) == 1
    time.sleep(0.05)
    assert mf.get_model("embed:x", factory) == "MODEL"  # ready afterwards, without another load


def test_failed_download_is_not_retried_within_window(monkeypatch):
    n = {"dl": 0}

    def factory(local_only):
        if local_only:
            raise ValueError("missing")
        n["dl"] += 1
        raise ValueError("403")

    with pytest.raises(RuntimeError):
        mf.get_model("rerank:y", factory, blocking=True)
    with pytest.raises(RuntimeError, match="not available"):
        mf.get_model("rerank:y", factory, blocking=True)
    assert n["dl"] == 1
    monkeypatch.setenv("MODEL_DOWNLOAD", "false")
    with pytest.raises(RuntimeError):
        mf.get_model("embed:z", factory, blocking=True)
    assert n["dl"] == 1  # air-gapped mode never downloads


def test_cached_model_loads_without_download():
    assert mf.get_model("embed:c", lambda local_only: "LOCAL" if local_only else 1 / 0) == "LOCAL"
