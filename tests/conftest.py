"""
Shared pytest fixtures. The whole suite runs OFFLINE:
  * tests/fakes/fastembed  replaces the real embedding/reranker models;
  * tests/fakes/fake_llm_server.py  fakes the OpenRouter API (chat, embeddings, rerank, key).
"""
from __future__ import annotations

import dataclasses
import os
import socket
import struct
import subprocess
import sys
import time
import urllib.request
import zlib
from pathlib import Path

import pytest

# Ignore the developer's own .env / .streamlit/secrets.toml: tests must behave the same on every machine.
os.environ["RAG_IGNORE_DOTENV"] = "1"

ROOT = Path(__file__).resolve().parent.parent
FAKES = Path(__file__).resolve().parent / "fakes"
sys.path.insert(0, str(FAKES))  # the stub `fastembed` must win over the real one
sys.path.insert(1, str(ROOT))

for var in ("GROQ_API_KEY", "GROQ_BASE_URL", "IMAGE_ANSWERS", "OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "MODEL_PROFILE", "MASTER_MODELS", "FAST_MODELS",
            "VERIFIER_MODELS", "VISION_MODELS", "EMBED_BACKEND", "RERANK_BACKEND", "VERIFY_WITH_LLM",
            "VISION_INDEX", "OCR_SCANNED", "DAILY_BUDGET_USD", "REASONING_EFFORT", "DATA_COLLECTION",
            "OLLAMA_BASE_URL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "BEDROCK_REGION", "LLM_MODE", "MANUAL_MODEL",
            "AUTH_MODE", "DATABASE_URL", "ROUTING_STRATEGY", "ALLOW_SENSITIVE_CLOUD"):
    os.environ.pop(var, None)


def _png(w: int, h: int) -> bytes:
    rows = b"".join(b"\x00" + bytes([255, 255, 255]) * w for _ in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def make_datasheet() -> bytes:
    import pymupdf

    doc = pymupdf.open()

    def lines(page, y, items, size=10):
        for item in items:
            page.insert_text((50, y), item, fontsize=size)
            y += size + 5
        return y

    def table(page, x, y, rows, colw=120, rowh=18):
        for r, row in enumerate(rows):
            for c, cell in enumerate(row):
                rect = pymupdf.Rect(x + c * colw, y + r * rowh, x + (c + 1) * colw, y + (r + 1) * rowh)
                page.draw_rect(rect, color=(0, 0, 0), width=0.8)
                page.insert_text((rect.x0 + 3, rect.y1 - 5), cell, fontsize=9)
        return y + len(rows) * rowh + 10

    p = doc.new_page()
    y = lines(p, 60, ["LM7805X VOLTAGE REGULATOR", "", "FEATURES", "Output current up to 1.5 A",
                      "Thermal overload protection", "Output voltage VOUT = 5.0V", "",
                      "Absolute Maximum Ratings", "Input voltage VIN = 35V",
                      "Operating temperature TA = 125°C"])
    p = doc.new_page()
    y = lines(p, 60, ["ELECTRICAL CHARACTERISTICS", "Conditions: VIN = 10V, IO = 500mA, TJ = 25°C"])
    y = table(p, 50, y + 10, [["Parameter", "Min", "Typ", "Max"], ["Output voltage", "4.8", "5.0", "5.2"],
                               ["Quiescent current", "", "5", "8 mA"], ["Peak output current", "", "2.2", "A"]])
    lines(p, y + 20, ["Dropout voltage VDO = 2.0V"])
    p = doc.new_page()
    y = lines(p, 60, ["POWER DISSIPATION", "The power dissipated in the regulator is",
                      "PD = (VIN - VOUT) x IOUT",
                      "where PD is the power dissipation in watts, VIN the input voltage",
                      "and IOUT the output current.", "Junction temperature TJ = TA + PD x RthJA"])
    p.insert_image(pymupdf.Rect(50, y + 10, 300, y + 160), stream=_png(200, 120))
    lines(p, y + 175, ["Figure 1. Output voltage vs input voltage"])
    p = doc.new_page()
    y = lines(p, 60, ["PIN CONFIGURATION"])
    y = table(p, 50, y + 10, [["Pin", "Name", "Function"], ["1", "INPUT", "Unregulated input"],
                               ["2", "GND", "Ground"], ["3", "OUTPUT", "Regulated output"]])
    p.insert_image(pymupdf.Rect(50, y + 10, 300, y + 160), stream=_png(180, 110))
    lines(p, y + 175, ["Figure 2. Functional block diagram"])
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture(autouse=True)
def _isolated_index_cache():
    """Each test starts with an empty in-memory index cache (it is per process in the app)."""
    from rag import pipeline

    from ops import answer_cache, telemetry
    from ops.guard import LIMITER

    pipeline._registry.clear()
    answer_cache._ANSWERS = None
    LIMITER._hits.clear()
    telemetry.reset()
    from concepts.graph import clear_cache
    clear_cache()
    from llm.model_router import reset_state
    reset_state()
    from rag.model_failures import clear as clear_model_failures
    clear_model_failures()
    yield
    pipeline._registry.clear()


@pytest.fixture(scope="session")
def sample_pdf() -> bytes:
    return make_datasheet()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def fake_llm():
    port = _free_port()
    proc = subprocess.Popen([sys.executable, str(FAKES / "fake_llm_server.py"), str(port)])
    url = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            urllib.request.urlopen(url + "/_log", timeout=0.5)
            break
        except OSError:
            time.sleep(0.1)
    yield url
    proc.terminate()


@pytest.fixture
def server(fake_llm):
    urllib.request.urlopen(fake_llm + "/_reset")
    return fake_llm


def control(url: str, fail: str) -> None:
    urllib.request.urlopen(f"{url}/_ctl?fail={fail}")


def server_log(url: str) -> dict:
    import json

    return json.load(urllib.request.urlopen(url + "/_log"))


@pytest.fixture
def make_settings(tmp_path, fake_llm):
    from config.settings import load_settings

    def _make(**overrides):
        base = load_settings()
        # Offline defaults: local (fake) embeddings + reranker, no indexing-time vision, no LLM cross-check.
        # Tests that exercise the OpenRouter retrieval / vision / verifier paths switch them on explicitly.
        defaults = dict(data_dir=tmp_path, groq_api_key="gsk_test", groq_base_url=fake_llm + "/openai/v1",
                        openrouter_api_key="sk-or-test", openrouter_base_url=fake_llm + "/api/v1",
                        model_profile="free", rerank=True, embed_backend="local", rerank_backend="local",
                        vision_index=False, ocr_scanned=False, verify_with_llm=False, llm_timeout=10.0,
                        llm_max_retries=0)
        defaults.update(overrides)
        return dataclasses.replace(base, **defaults)

    return _make


@pytest.fixture
def indexed(sample_pdf, make_settings):
    from rag.pipeline import get_index

    settings = make_settings()
    index, _ = get_index(sample_pdf, "LM7805X.pdf", settings)
    return index, settings
