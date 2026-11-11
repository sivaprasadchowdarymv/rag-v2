"""Create an admin account (AUTH_MODE=users).

    python -m scripts.create_admin --username admin --tenant default        (password asked interactively)
"""
from __future__ import annotations

import argparse
import getpass

from config.settings import load_settings
from security.auth import AuthError, AuthService


def main() -> None:  # pragma: no cover - interactive CLI
    ap = argparse.ArgumentParser()
    ap.add_argument("--username", required=True)
    ap.add_argument("--tenant", default="default")
    ap.add_argument("--role", default="admin", choices=["admin", "editor", "viewer"])
    args = ap.parse_args()
    pw = getpass.getpass("Password (≥10 chars, letters + digits): ")
    if pw != getpass.getpass("Repeat password: "):
        raise SystemExit("Passwords differ.")
    try:
        user = AuthService(load_settings()).create_user(args.username, pw, [args.role], args.tenant)
    except AuthError as exc:
        raise SystemExit(exc.user_message) from None
    print(f"created {user.username} ({args.role}) in tenant {args.tenant}")


if __name__ == "__main__":  # pragma: no cover
    main()
