"""Hugging Face Docker Space packaging: launch.py spaces (no network: the Hub API is mocked)."""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("launch_spaces", ROOT / "launch.py")
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)  # type: ignore[union-attr]

GOOD = {"GROQ_API_KEY": "gsk_secret1", "GROQ_API_KEY_2": "gsk_secret2", "OPENROUTER_API_KEY": "sk-or-s",
        "AUTH_MODE": "users", "ADMIN_USERNAME": "admin", "ADMIN_PASSWORD": "Passw0rd1234",
        "DATABASE_URL": "postgresql://u:p@ep-x.neon.tech/db?sslmode=require", "HF_TOKEN": "hf_tok",
        "HF_DATASET_REPO": "me/rag-data", "LLM_MODE": "hybrid", "DATA_DIR": "C:/x", "OLLAMA_MODEL": "q",
        "POSTGRES_PASSWORD": "pg", "MASTER_MODELS": "groq:a", "EMPTY": ""}


def test_space_env_splits_secrets_from_variables():
    secrets, variables, warnings = launch.space_env(GOOD)
    assert set(secrets) == {"GROQ_API_KEY", "GROQ_API_KEY_2", "OPENROUTER_API_KEY", "ADMIN_PASSWORD",
                            "DATABASE_URL", "HF_TOKEN"}
    assert variables["LLM_MODE"] == "cloud" and variables["MASTER_MODELS"] == "groq:a"
    assert "ADMIN_USERNAME" in variables and "HF_DATASET_REPO" in variables
    for skipped in ("DATA_DIR", "OLLAMA_MODEL", "POSTGRES_PASSWORD", "EMPTY"):
        assert skipped not in secrets and skipped not in variables
    assert not any(v.startswith(("gsk_", "sk-or", "hf_")) for v in variables.values())
    assert warnings == ["LLM_MODE=hybrid changed to cloud for the Space (there is no Ollama there)."]


def test_space_env_warns_for_unsafe_or_temporary_setups():
    secrets, variables, warnings = launch.space_env({"DATABASE_URL": "sqlite:///x.db"})
    text = " ".join(warnings)
    assert "DATABASE_URL" not in secrets and variables["LLM_MODE"] == "cloud"
    for part in ("AUTH_MODE", "SQLite", "HF_TOKEN", "GROQ_API_KEY"):
        assert part in text


def test_space_folder_has_docker_readme_and_no_secrets(tmp_path):
    src = tmp_path / "src"
    for rel, body in {"app.py": "x", "README.md": "# Project", "Dockerfile.spaces": "FROM python",
                      ".env": "GROQ_API_KEY=gsk_real", ".env.backup": "x", ".streamlit/config.toml": "[server]",
                      ".streamlit/secrets.toml": "K='v'", ".streamlit/secrets.toml.example": "x",
                      "data/app.db": "x", "tests/test_a.py": "x", "x-1.0-py3-none-any.whl": "x",
                      "docs/USER_GUIDE.md": "g", "docs/screenshots/a.png": "x", "assets/logo.png": "x",
                      ".git/config": "x", "pkg/__pycache__/m.pyc": "x", "pkg/m.py": "x"}.items():
        f = src / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(body)
    dest = tmp_path / "space"
    files = {p.as_posix() for p in launch.build_space(dest, src, title="My RAG")}
    assert files == {"app.py", "README.md", "PROJECT_README.md", "Dockerfile", "Dockerfile.spaces",
                     ".streamlit/config.toml", "docs/USER_GUIDE.md", "assets/logo.png", "pkg/m.py"}
    readme = (dest / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("---\ntitle: My RAG\n") and "sdk: docker" in readme and "app_port: 7860" in readme
    assert (dest / "PROJECT_README.md").read_text() == "# Project"
    assert (dest / "Dockerfile").read_text() == "FROM python"
    launch.build_space(dest, src)  # rebuilding replaces the folder
    assert (dest / "README.md").read_text(encoding="utf-8").startswith("---\ntitle: RAG Ultimate\n")


def test_real_project_space_build_is_clean(tmp_path):
    files = [p.as_posix() for p in launch.build_space(tmp_path / "s")]
    assert "Dockerfile" in files and "app.py" in files and "deploy/hf-space/start.sh" in files
    assert "docs/USER_GUIDE.md" in files  # the in-app Help centre reads it
    bad = [f for f in files if re.search(r"(^|/)\.env|secrets\.toml|\.whl$|^data/|^tests/|^\.git/", f)]
    assert bad == []


def test_space_dockerfile_meets_spaces_rules():
    docker = (ROOT / "Dockerfile.spaces").read_text()
    assert "useradd -m -u 1000 user" in docker and "USER user" in docker and "COPY --chown=user" in docker
    assert "EXPOSE 7860" in docker
    start = (ROOT / "deploy/hf-space/start.sh").read_text()
    assert "--server.enableXsrfProtection=false" in start and '--server.port="${PORT:-7860}"' in start
    assert "python -m database.migrations" in start and "/data" in start
    assert "app_port: 7860" in launch.SPACE_README


class _FakeApi:
    calls: list = []

    def __init__(self, token=None):
        self.token = token

    def whoami(self):
        if self.token != "hf_write":
            raise PermissionError("401")
        return {"name": "me"}

    def create_repo(self, repo, **kw):
        _FakeApi.calls.append(("create", repo, kw))

    def add_space_secret(self, repo, k, v):
        _FakeApi.calls.append(("secret", k, v))

    def add_space_variable(self, repo, k, v):
        _FakeApi.calls.append(("var", k, v))

    def upload_folder(self, **kw):
        _FakeApi.calls.append(("upload", kw["repo_id"], kw["repo_type"]))


def _args(**kw):
    base = dict(push=None, token=None, public=False, no_secrets=False, title="RAG Ultimate")
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.fixture
def fake_project(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("\n".join(f"{k}={v}" for k, v in GOOD.items()))
    monkeypatch.setattr(launch, "ENV_FILE", env)
    monkeypatch.setattr(launch, "SPACE_DIR", tmp_path / "build" / "hf-space")
    monkeypatch.setattr(launch, "build_space", lambda title="": [Path("app.py")])
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=_FakeApi))
    monkeypatch.delenv("HF_WRITE_TOKEN", raising=False)
    _FakeApi.calls = []
    return tmp_path


def test_push_creates_private_docker_space_and_sends_settings(fake_project, capsys):
    launch.cmd_spaces(_args(push="me/rag", token="hf_write"))
    calls = _FakeApi.calls
    assert calls[0] == ("create", "me/rag", {"repo_type": "space", "space_sdk": "docker", "private": True,
                                             "exist_ok": True})
    assert ("secret", "GROQ_API_KEY", "gsk_secret1") in calls and ("var", "LLM_MODE", "cloud") in calls
    assert calls[-1] == ("upload", "me/rag", "space")
    out = capsys.readouterr().out
    assert "gsk_secret" not in out and "Passw0rd" not in out and "hf_write" not in out  # names only
    assert "https://huggingface.co/spaces/me/rag" in out


def test_push_without_secrets_and_bad_inputs(fake_project, capsys):
    launch.cmd_spaces(_args(push="me/rag", token="hf_write", no_secrets=True, public=True))
    assert [c[0] for c in _FakeApi.calls] == ["create", "upload"] and _FakeApi.calls[0][2]["private"] is False
    with pytest.raises(SystemExit):
        launch.cmd_spaces(_args(push="not a repo", token="hf_write"))
    with pytest.raises(SystemExit):
        launch.cmd_spaces(_args(push="me/rag", token="hf_bad"))
    assert "rejected the token" in capsys.readouterr().out


def test_db_probe(monkeypatch):
    spec2 = importlib.util.spec_from_file_location("db_probe", ROOT / "deploy/hf-space/db_probe.py")
    probe = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(probe)  # type: ignore[union-attr]
    assert probe.reachable("") and probe.reachable("sqlite:///x.db")
    assert not probe.reachable("postgresql://u:p@127.0.0.1:1/db", timeout=1)
    assert not probe.reachable("postgresql:///nohost")
    import socket
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen()
    try:
        assert probe.reachable(f"postgresql://u:p@127.0.0.1:{srv.getsockname()[1]}/db?sslmode=require")
    finally:
        srv.close()
