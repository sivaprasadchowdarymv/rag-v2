"""launch.py: .env parsing, generated .env, Streamlit Cloud secrets, pre-flight check (no network)."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("launch", ROOT / "launch.py")
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)  # type: ignore[union-attr]


def test_parse_env_handles_comments_quotes_and_empty_values():
    env = launch.parse_env('A=1\n# c\nB=   # note only\nC="x # y"\nD=val  # trailing\nbad line\nE=\n')
    assert env == {"A": "1", "B": "", "C": "x # y", "D": "val", "E": ""}


def test_generated_env_loads_into_settings(tmp_path, monkeypatch):
    text = launch.build_env({"GROQ_API_KEY": "gsk_a", "GROQ_API_KEY_2": "gsk_b", "OPENROUTER_API_KEY": "sk-or-x",
                             "AUTH_MODE": "users", "ADMIN_USERNAME": "admin", "ADMIN_PASSWORD": "Passw0rd1234",
                             "POSTGRES_PASSWORD": "pg"})
    env = launch.parse_env(text)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from config.settings import load_settings

    s = load_settings()
    assert s.groq_backup_keys == ("gsk_b",) and s.auth_mode == "users" and not s.discovery_enabled
    assert s.models_for("master")[0] == "groq:openai/gpt-oss-120b"
    assert len(s.models_for("vision")) == 6 and all(e.startswith("openrouter:") for e in s.models_for("vision"))
    assert s.db_url.startswith("sqlite")


def test_secrets_toml_is_valid_and_warns():
    import tomllib

    env = {"GROQ_API_KEY": 'gsk_"q\\x', "AUTH_MODE": "none", "DATABASE_URL": "", "OLLAMA_BASE_URL": "http://x",
           "EMPTY": ""}
    toml, warnings = launch.env_to_toml(env)
    data = tomllib.loads(toml)
    assert data == {"GROQ_API_KEY": 'gsk_"q\\x', "AUTH_MODE": "none"}       # local-only + empty dropped
    joined = " ".join(warnings)
    assert "AUTH_MODE" in joined and "DATABASE_URL" in joined and "HF_TOKEN" in joined


def test_secrets_for_a_proper_cloud_setup_have_no_warnings():
    env = {"GROQ_API_KEY": "g", "AUTH_MODE": "users", "DATABASE_URL": "postgresql://u:p@ep-x.neon.tech/db?sslmode=require",
           "HF_TOKEN": "hf", "HF_DATASET_REPO": "me/data", "LLM_MODE": "hybrid"}
    assert launch.env_to_toml(env)[1] == []


def test_example_secrets_file_parses_and_matches_launcher_chains():
    import tomllib

    data = tomllib.loads((ROOT / ".streamlit/secrets.toml.example").read_text(encoding="utf-8"))
    for k, v in launch.CHAINS.items():
        assert data[k] == v, k


def test_strong_password_is_url_safe():
    p = launch.strong_password()
    assert len(p) == 20 and p.isalnum() and any(c.isdigit() for c in p)


def test_check_command_runs_offline(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_x")
    monkeypatch.setenv("DATABASE_URL", "")
    with pytest.raises(SystemExit) as e:
        launch.cmd_check(argparse.Namespace(online=False))
    out = capsys.readouterr().out
    assert e.value.code == 0 and "Database (SQLite): connected" in out and "gsk_x" not in out


def test_check_reports_bad_database_without_leaking_password(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_x")
    monkeypatch.setenv("DATABASE_URL", "postgresql://rag:TopSecret99@127.0.0.1:1/rag")
    with pytest.raises(SystemExit) as e:
        launch.cmd_check(argparse.Namespace(online=False))
    out = capsys.readouterr().out
    assert e.value.code == 1 and "cannot connect" in out and "TopSecret99" not in out
