"""Windows portability guards: explicit UTF-8 for text files, tests independent of the developer's .env."""
from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_TEXT_IO = re.compile(r"\.(write_text|read_text)\(|[^.\w]open\([^)]*[\"'][rwa]\+?[\"']")


def test_text_file_io_declares_utf8():
    """Windows defaults to cp1252, which cannot store characters such as Δ — µ °; every text read/write in
    application code must pass encoding="utf-8"."""
    offenders = []
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith((".venv/", "tests/", "data/")):
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if _TEXT_IO.search(line) and "encoding" not in " ".join(lines[i:i + 3]):
                offenders.append(f"{rel}:{i + 1}: {line.strip()}")
    assert not offenders, "text I/O without encoding:\n" + "\n".join(offenders)


def test_tests_ignore_local_env():
    from config.settings import load_settings
    assert os.environ.get("RAG_IGNORE_DOTENV") == "1"
    s = load_settings()
    assert s.auth_mode == "none" and s.llm_mode == "hybrid" and not s.ollama_base_url
