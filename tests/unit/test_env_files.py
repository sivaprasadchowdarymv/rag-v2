"""Env templates parse to the intended values (regression: empty value + inline comment)."""
from pathlib import Path

import pytest
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name", [".env.example", "docs/env.complete.example"])
def test_no_comment_is_parsed_as_a_value(name):
    vals = dotenv_values(ROOT / name)
    bad = {k: v for k, v in vals.items() if v and v.lstrip().startswith("#")}
    assert not bad, bad
    assert vals["APP_PASSWORD"] == "" and vals["DATA_DIR"] == "" and vals["OLLAMA_BASE_URL"] == ""


def test_loader_ignores_comment_values(monkeypatch):
    from config.settings import load_settings

    monkeypatch.setenv("APP_PASSWORD", "# if set, the whole app asks for this password")
    monkeypatch.setenv("OLLAMA_BASE_URL", "# e.g. http://localhost:11434")
    s = load_settings()
    assert s.app_password == "" and s.ollama_base_url == ""


def test_complete_env_builds_all_three_providers(monkeypatch, tmp_path):
    vals = dotenv_values(ROOT / "docs/env.complete.example")
    assert vals["MASTER_MODELS"] == "@groq,@anyrouter_text,@openrouter_text"
    assert vals["VISION_MODELS"] == "@openrouter_vision,@anyrouter_vision"
    assert vals["MODEL_FREE_ONLY"] == "true" and vals["MODEL_ALLOW_PAID_FALLBACK"] == "false"
    for k, v in vals.items():
        monkeypatch.setenv(k, v or "")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from config.settings import load_settings

    s = load_settings()
    assert s.discovery_enabled and s.master_models[0] == "@groq" and s.model_profile == "free"
