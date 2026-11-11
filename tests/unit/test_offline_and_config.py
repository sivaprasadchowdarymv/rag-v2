"""2.1: new features work offline; every new setting is documented; FAQ never exposes secrets."""
from __future__ import annotations

import re
import socket
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def no_network(monkeypatch):
    def guard(*a, **k):
        raise OSError("network disabled for this test")

    monkeypatch.setattr(socket, "create_connection", guard)
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **k: guard())


def test_faq_metadata_and_cache_work_offline(no_network, indexed):
    from ops.answer_cache import AnswerCache
    from rag.metadata_intel import build_profile
    from ui import faq

    assert faq.search("how do I upload a document")[0]["id"] == "up-how"
    assert faq.fallback("workspace link")
    index, _ = indexed
    assert build_profile(index)["parameters"]
    c = AnswerCache(4, 60)
    c.store("ctx", "What is VOUT?", 1, ["d"])
    assert c.lookup("ctx", "what's vout")[0] == 1


def test_new_settings_documented_in_env_example():
    env = (ROOT / ".env.example").read_text(encoding="utf-8")
    for name in ("ANSWER_CACHE_ENABLED", "ANSWER_CACHE_SIMILARITY", "ANSWER_CACHE_SCOPE", "ANSWER_CACHE_SIZE",
                 "ANSWER_CACHE_TTL"):
        assert re.search(rf"^#?\s*{name}=", env, re.M), name


def test_new_settings_defaults_and_bounds(monkeypatch):
    from config.settings import load_settings

    s = load_settings()
    assert s.answer_cache_enabled and s.answer_cache_scope == "user" and 0 < s.answer_cache_similarity <= 1
    monkeypatch.setenv("ANSWER_CACHE_SCOPE", "everyone")      # invalid -> safe default
    monkeypatch.setenv("ANSWER_CACHE_SIMILARITY", "7")         # clamped
    s = load_settings()
    assert s.answer_cache_scope == "user" and s.answer_cache_similarity <= 1.0


def test_faq_contains_no_secrets_or_internal_prompts():
    from rag import prompts
    from ui import faq

    text = " ".join(str(e["q"]) + " " + str(e["a"]) for e in faq.FAQ)
    assert not re.search(r"(gsk_|sk-or-|sk-ant-|sk-[A-Za-z0-9]{10,}|AKIA[0-9A-Z]{8})", text)
    for name in dir(prompts):
        val = getattr(prompts, name)
        if isinstance(val, str) and len(val) > 80:
            assert val[:80] not in text  # no system prompt text in help answers
    assert len({e["cat"] for e in faq.FAQ}) == 12 and len(faq.FAQ) >= 36


def test_custom_faq_entries(tmp_path):
    from ui import faq

    e = faq.add_custom(tmp_path, "How do I reach IT support?", "Call extension 1234.", "Troubleshooting", "guide")
    assert faq.search("reach it support", data_dir=tmp_path)[0]["id"] == e["id"]
    with pytest.raises(ValueError):
        faq.add_custom(tmp_path, "x", "y")
    assert faq.delete_custom(tmp_path, str(e["id"])) and not faq.custom_entries(tmp_path)
