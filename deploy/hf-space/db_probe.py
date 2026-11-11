"""Exit 0 if the DATABASE_URL host:port accepts a TCP connection within 8 s, else 1. Never prints the URL."""
from __future__ import annotations

import os
import socket
import sys
from urllib.parse import urlsplit


def reachable(url: str, timeout: float = 8.0) -> bool:
    if not url or url.startswith("sqlite"):
        return True
    try:
        parts = urlsplit(url)
        host, port = parts.hostname, parts.port or 5432
    except ValueError:
        return False
    if not host:
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError as exc:
        print(f"database probe: {host}:{port} unreachable ({type(exc).__name__})", file=sys.stderr)
        return False


if __name__ == "__main__":
    sys.exit(0 if reachable(os.environ.get("DATABASE_URL", "")) else 1)
