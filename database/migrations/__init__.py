"""
Versioned schema migrations (plain Python, SQLite + PostgreSQL).

Each migration module defines `VERSION` (int), `NAME` and `upgrade(conn)`.
Applied versions are recorded in `schema_migrations`; `migrate(engine)` applies
the missing ones in order, each in its own transaction, and is idempotent.

    python -m database.migrations            # upgrade the configured DATABASE_URL
    python -m database.migrations --status
"""
from __future__ import annotations

import importlib
import pkgutil
import time
from typing import List, Tuple

from sqlalchemy import Column, Float, Integer, MetaData, String, Table, select
from sqlalchemy.engine import Engine

_meta = MetaData()
schema_migrations = Table("schema_migrations", _meta, Column("version", Integer, primary_key=True),
                          Column("name", String(200)), Column("applied_at", Float))


def discover() -> List[Tuple[int, str, object]]:
    out = []
    for info in pkgutil.iter_modules(__path__):  # type: ignore[name-defined]
        if info.name.startswith("m") and info.name[1:5].isdigit():
            mod = importlib.import_module(f"{__name__}.{info.name}")
            out.append((int(mod.VERSION), str(mod.NAME), mod))
    out.sort(key=lambda t: t[0])
    versions = [v for v, _, _ in out]
    if len(set(versions)) != len(versions):
        raise RuntimeError("duplicate migration versions")
    return out


def applied(engine: Engine) -> List[int]:
    _meta.create_all(engine, tables=[schema_migrations])
    with engine.connect() as conn:
        return [r[0] for r in conn.execute(select(schema_migrations.c.version))]


def migrate(engine: Engine, target: int = 0) -> List[int]:
    """Apply pending migrations (up to `target` if > 0). Returns the versions applied now."""
    done, now_applied = set(applied(engine)), []
    for version, name, mod in discover():
        if version in done or (target and version > target):
            continue
        with engine.begin() as conn:  # one transaction per migration
            mod.upgrade(conn)
            conn.execute(schema_migrations.insert().values(version=version, name=name, applied_at=time.time()))
        now_applied.append(version)
    return now_applied


def main() -> None:  # pragma: no cover - CLI
    import argparse

    from config.settings import load_settings
    from database.repository import make_engine

    ap = argparse.ArgumentParser(description="Apply database migrations")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    engine = make_engine(load_settings().db_url)
    if args.status:
        have = set(applied(engine))
        for v, n, _ in discover():
            print(f"{v:04d} {n:40s} {'applied' if v in have else 'PENDING'}")
        return
    print("applied:", migrate(engine) or "nothing (up to date)")


if __name__ == "__main__":  # pragma: no cover
    main()
