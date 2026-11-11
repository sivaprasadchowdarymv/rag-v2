"""
Backup and restore: database + data directory (indexes, figures, originals, workspaces, evaluations).

    python -m scripts.backup create  [--dest backups]          -> backups/rag-backup-YYYYmmdd-HHMMSS/
    python -m scripts.backup verify  backups/rag-backup-...
    python -m scripts.backup restore backups/rag-backup-... [--yes]
    python -m scripts.backup restore backups/latest              (newest backup in that folder)

Database
    SQLite      online copy with the sqlite3 backup API (consistent while the app runs)
    PostgreSQL  pg_dump --format=custom when available, else a portable JSON export of every table
                (restore: pg_restore --clean, or JSON import into a freshly migrated schema)
Files: data/{cache,documents,objects,users,evaluations,telemetry} as tar.gz (local models are excluded:
they are re-downloaded). manifest.json holds SHA-256 checksums; restore refuses a backup that fails them.
Restore is destructive for the target database and data folders and asks for confirmation (--yes to skip).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Dict, Optional

from sqlalchemy import MetaData, insert, select, text
from sqlalchemy.engine import make_url

from config.settings import Settings, get_logger, load_settings

log = get_logger("backup")
DATA_PARTS = ("cache", "documents", "objects", "users", "evaluations", "telemetry")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(v):
    if isinstance(v, (bytes, bytearray)):
        return {"__b64__": base64.b64encode(v).decode()}
    return v


def _from_json(v):
    if isinstance(v, dict) and set(v) == {"__b64__"}:
        return base64.b64decode(v["__b64__"])
    return v


def _pg_env(url) -> Dict[str, str]:
    env = dict(os.environ)
    if url.password:
        env["PGPASSWORD"] = url.password
    return env


def _pg_args(url) -> list:
    args = ["-h", url.host or "localhost", "-U", url.username or "postgres", "-d", url.database]
    return args + (["-p", str(url.port)] if url.port else [])


def export_json(engine, out: Path) -> None:
    meta = MetaData()
    meta.reflect(engine)
    data = {}
    with engine.connect() as c:
        for name, table in meta.tables.items():
            data[name] = [{k: _jsonable(v) for k, v in row._mapping.items()} for row in c.execute(select(table))]
    out.write_text(json.dumps(data, default=str), encoding="utf-8")


def import_json(engine, src: Path) -> None:
    from database.migrations import migrate
    from database.models import Base

    data = json.loads(src.read_text(encoding="utf-8"))
    Base.metadata.drop_all(engine)
    with engine.begin() as c:
        c.execute(text("DROP TABLE IF EXISTS schema_migrations"))
    migrate(engine)
    meta = MetaData()
    meta.reflect(engine)
    order = [t.name for t in meta.sorted_tables]
    with engine.begin() as c:
        for name in order:
            if name == "schema_migrations":
                continue
            rows = [{k: _from_json(v) for k, v in r.items()} for r in data.get(name, [])]
            if rows:
                c.execute(insert(meta.tables[name]), rows)
        if engine.dialect.name == "postgresql":  # keep autoincrement sequences ahead of restored ids
            for name in order:
                t = meta.tables[name]
                if "id" in t.c and str(t.c.id.type).upper().startswith("INTEGER"):
                    # table names come from the reflected schema, never from user input
                    c.execute(text(f"SELECT setval(pg_get_serial_sequence('{name}', 'id'), "  # nosec B608
                                   f"COALESCE((SELECT MAX(id) FROM {name}), 0) + 1, false)"))


def create(settings: Settings, dest: Path) -> Path:
    from database.repository import get_repository

    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = Path(dest) / f"rag-backup-{stamp}"
    target.mkdir(parents=True, exist_ok=False)
    url = make_url(settings.db_url)
    repo = get_repository(settings)
    if url.get_backend_name() == "sqlite":
        src = sqlite3.connect(url.database)
        dst = sqlite3.connect(target / "database.sqlite")
        with dst:
            src.backup(dst)
        src.close()
        dst.close()
        db_file, db_kind = "database.sqlite", "sqlite"
    elif shutil.which("pg_dump"):
        subprocess.run(["pg_dump", "--format=custom", "--no-owner", "-f", str(target / "database.pgdump")]
                       + _pg_args(url), check=True, env=_pg_env(url), capture_output=True)
        db_file, db_kind = "database.pgdump", "pg_dump"
    else:
        export_json(repo.engine, target / "database.json")
        db_file, db_kind = "database.json", "json"
    data_dir = Path(settings.data_dir)
    with tarfile.open(target / "data.tar.gz", "w:gz") as tar:
        for part in DATA_PARTS:
            if (data_dir / part).exists():
                tar.add(data_dir / part, arcname=part)
    manifest = {"created": stamp, "db_kind": db_kind, "db_file": db_file, "db_dialect": url.get_backend_name(),
                "files": {f: _sha(target / f) for f in (db_file, "data.tar.gz")}, "format": 1}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("backup written to %s", target)
    return target


def resolve(path: Path) -> Path:
    """Accept a backup folder, or "latest" / a parent folder (newest backup inside). Friendly errors otherwise."""
    path = Path(path)
    parent = path.parent if path.name != "latest" else path.parent
    if path.name == "latest" or (path.is_dir() and not (path / "manifest.json").exists()):
        base = parent if path.name == "latest" else path
        found = sorted(base.glob("rag-backup-*")) if base.is_dir() else []
        if not found:
            raise SystemExit(f"No backups found in '{base}'. Create one with: python -m scripts.backup create --dest backups")
        return found[-1]
    if not (path / "manifest.json").exists():
        found = sorted(path.parent.glob("rag-backup-*")) if path.parent.is_dir() else []
        hint = ("\nAvailable backups:\n  " + "\n  ".join(str(f) for f in found)) if found else \
            "\nNo backups exist yet. Create one with: python -m scripts.backup create --dest backups"
        raise SystemExit(f"Backup folder '{path}' was not found.{hint}\n(Tip: use 'backups\\latest' for the newest one.)")
    return path


def verify(path: Path) -> Dict:
    path = resolve(path)
    manifest = json.loads((Path(path) / "manifest.json").read_text(encoding="utf-8"))
    for f, digest in manifest["files"].items():
        if _sha(Path(path) / f) != digest:
            raise ValueError(f"checksum mismatch for {f}: the backup is damaged")
    return manifest


def restore(settings: Settings, path: Path) -> None:
    from database.repository import get_repository, reset_repositories

    path = resolve(path)
    manifest = verify(path)
    url = make_url(settings.db_url)
    reset_repositories()
    if manifest["db_kind"] == "sqlite":
        if url.get_backend_name() != "sqlite":
            raise ValueError("this backup holds a SQLite database; restore it with a sqlite DATABASE_URL")
        src = sqlite3.connect(path / manifest["db_file"])
        dst = sqlite3.connect(url.database)
        with dst:
            src.backup(dst)
        src.close()
        dst.close()
    elif manifest["db_kind"] == "pg_dump":
        subprocess.run(["pg_restore", "--clean", "--if-exists", "--no-owner", str(path / manifest["db_file"])]
                       + _pg_args(url), check=True, env=_pg_env(url), capture_output=True)
    else:
        from database.repository import make_engine
        import_json(make_engine(settings.db_url), path / manifest["db_file"])
    data_dir = Path(settings.data_dir)
    for part in DATA_PARTS:
        if (data_dir / part).exists():
            shutil.rmtree(data_dir / part)
    with tarfile.open(path / "data.tar.gz", "r:gz") as tar:
        tar.extractall(data_dir, filter="data")  # rejects absolute paths / links outside the target
    from rag import pipeline
    pipeline._registry.clear()
    get_repository(settings)  # re-open (and migrate, if the backup is older)
    log.info("restored %s", path)


def main(argv: Optional[list] = None) -> None:  # pragma: no cover - CLI
    ap = argparse.ArgumentParser(description="Backup / restore RAG Ultimate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--dest", default="backups")
    v = sub.add_parser("verify")
    v.add_argument("path")
    r = sub.add_parser("restore")
    r.add_argument("path")
    r.add_argument("--yes", action="store_true")
    args = ap.parse_args(argv)
    s = load_settings()
    if args.cmd == "create":
        print(create(s, Path(args.dest)))
    elif args.cmd == "verify":
        print(json.dumps(verify(Path(args.path)), indent=2))
    else:
        target = resolve(Path(args.path))
        if not args.yes and input(f"Restore {target} over {s.db_url} and {s.data_dir}? [y/N] ").lower() != "y":
            print("aborted")
            return
        restore(s, target)
        print(f"restored {target}")


if __name__ == "__main__":  # pragma: no cover
    main()
