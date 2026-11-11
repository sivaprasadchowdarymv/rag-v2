"""Phase 9: SQLite + PostgreSQL parity, migrations on existing data, DB conversations, object storage (S3 via moto),
backup → wipe → restore round trips. PostgreSQL tests run when TEST_DATABASE_URL is set."""
from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import create_engine, inspect, text

from core.service import AskRequest, Principal, RAGService
from database.migrations import applied, migrate
from database.repository import Repository, reset_repositories

PG = os.getenv("TEST_DATABASE_URL", "")


def _fresh_pg_schema(url: str) -> str:
    """Isolated PostgreSQL schema per test (search_path), dropped afterwards by the fixture."""
    schema = "t_" + uuid.uuid4().hex[:8]
    eng = create_engine(url.replace("postgresql://", "postgresql+psycopg://"))
    with eng.begin() as c:
        c.execute(text(f"CREATE SCHEMA {schema}"))
    eng.dispose()
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}options=-csearch_path%3D{schema}", schema


@pytest.fixture(params=["sqlite", "postgresql"])
def db_url(request, tmp_path):
    if request.param == "sqlite":
        yield f"sqlite:///{tmp_path / 'app.db'}"
        return
    if not PG:
        pytest.skip("TEST_DATABASE_URL not set")
    url, schema = _fresh_pg_schema(PG)
    yield url
    reset_repositories()
    eng = create_engine(PG.replace("postgresql://", "postgresql+psycopg://"))
    with eng.begin() as c:
        c.execute(text(f"DROP SCHEMA {schema} CASCADE"))
    eng.dispose()


def test_repository_parity(db_url):
    repo = Repository(db_url)
    assert repo.health() and applied(repo.engine) == [1, 2]
    d, v, st = repo.register_version(tenant_id="t", owner_id="u", name="a.pdf", content_hash="h1", filename="a.pdf",
                                     chunks=[(i, "text", 1, "S", 10) for i in range(50)], meta={"k": [1, 2]})
    assert st == "new" and repo.chunk_count(v.id) == 50 and repo.versions(d.id)[0].meta == {"k": [1, 2]}
    repo.audit("t", "u", "x", detail={"a": 1})
    repo.add_feedback("t", "u", "m1", 1, ["clear"])
    assert repo.feedback_summary("t") == {"up": 1} and repo.list_audit("t")[0].detail == {"a": 1}
    job = repo.create_job("t", "u", "f.pdf")
    repo.update_job(job.id, status="failed", error="boom")
    assert [j.id for j in repo.list_jobs("t", "failed")] == [job.id]
    repo.set_policy("t", {"restricted_topics": ["salary"]})
    assert repo.get_policy("t") == {"restricted_topics": ["salary"]}
    assert repo.delete_document(d.id) == ["h1"]


def test_migration_upgrades_existing_v1_database(tmp_path):
    url = f"sqlite:///{tmp_path / 'old.db'}"
    eng = create_engine(url)
    migrate(eng, target=1)  # a database created by release 2.0.0 (schema v1)
    with eng.begin() as c:
        c.execute(text("INSERT INTO conversations (id, tenant_id, user_id, title, doc_ids, created_at, updated_at) "
                       "VALUES ('c1', 't', 'u', 'Old chat', '[]', 1, 1)"))
    assert "archived" not in {c["name"] for c in inspect(eng).get_columns("conversations")}
    assert migrate(eng) == [2]  # upgrade in place
    with eng.connect() as c:
        row = c.execute(text("SELECT title, archived FROM conversations WHERE id='c1'")).one()
    assert row[0] == "Old chat" and not row[1]
    assert "ix_audit_tenant_ts" in {i["name"] for i in inspect(eng).get_indexes("audit_events")}


def test_db_conversations_are_user_scoped(db_url):
    from chat.db_store import DBConversationStore
    from rag.models import QueryResult
    repo = Repository(db_url)
    a, b = DBConversationStore(repo, "t", "alice"), DBConversationStore(repo, "t", "bob")
    conv = a.create(["d1"])
    reply = a.add_exchange(conv, "What is VOUT?", QueryResult("answer", "5 V", question="What is VOUT?"))
    a.replace_answer(conv, reply.id, QueryResult("answer", "5.0 V"))
    loaded = a.load(conv.id)
    assert loaded.title == "What is VOUT?" and loaded.messages[1].content == "5.0 V"
    assert loaded.messages[1].variants[0]["content"] == "5 V" and loaded.last_question() == "What is VOUT?"
    assert b.load(conv.id) is None and b.list() == [] and len(a.list()) == 1
    b.delete(conv.id)
    assert a.load(conv.id) is not None
    a.delete(conv.id)
    assert a.list() == []


def test_s3_object_store_with_moto(make_settings):
    moto = pytest.importorskip("moto")
    with moto.mock_aws():
        from storage.objects import S3ObjectStore, make_object_store
        s = make_settings(storage_backend="s3", s3_bucket="rag-docs", s3_access_key="x", s3_secret_key="y",
                          s3_endpoint="")
        os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
        store = make_object_store(s)
        assert isinstance(store, S3ObjectStore)
        store.put("t1/abc.pdf", b"%PDF-1")
        assert store.get("t1/abc.pdf") == b"%PDF-1" and store.list("t1/") == ["t1/abc.pdf"]
        store.delete("t1/abc.pdf")
        assert store.get("t1/abc.pdf") is None
        with pytest.raises(ValueError):
            store.put("../etc/passwd", b"x")


def test_backup_restore_round_trip(db_url, make_settings, server, tmp_path, sample_pdf):
    from ingestion.jobs import IngestionService
    from scripts import backup
    s = make_settings(database_url=db_url, data_dir=tmp_path / "data")
    (tmp_path / "data").mkdir(exist_ok=True)
    p = Principal("u1", "default", ("admin",))
    res = IngestionService(s).ingest(sample_pdf, "LM7805X.pdf", p)
    before = RAGService(s).ask(AskRequest("What is the peak output current?", [res.index]), p)
    path = backup.create(s, tmp_path / "backups")
    manifest = backup.verify(path)
    assert manifest["db_kind"] in ("sqlite", "pg_dump") and set(manifest["files"]) >= {"data.tar.gz"}
    # disaster: drop the documents table data and the indexes on disk
    IngestionService(s).delete(res.document_id, p)
    from rag.pipeline import open_index
    assert open_index(res.index.doc_id, "LM7805X.pdf", s) is None
    backup.restore(s, path)
    from database.repository import get_repository
    repo = get_repository(s)
    doc = repo.document_for_hash(res.index.doc_id)
    assert doc is not None and doc.name == "LM7805X.pdf"
    ix = open_index(res.index.doc_id, "LM7805X.pdf", s)
    after = RAGService(s).ask(AskRequest("What is the peak output current?", [ix]), p)
    assert after.kind == before.kind == "answer" and "2.2" in after.answer
    assert "document_ingested" in [e.action for e in repo.list_audit("default")]


def test_backup_detects_tampering(make_settings, tmp_path, server):
    from scripts import backup
    s = make_settings(database_url=f"sqlite:///{tmp_path / 'x.db'}")
    from database.repository import get_repository
    get_repository(s)
    path = backup.create(s, tmp_path / "b")
    with open(path / "data.tar.gz", "ab") as f:
        f.write(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        backup.verify(path)


def test_json_export_import(db_url, tmp_path):
    from scripts.backup import export_json, import_json
    repo = Repository(db_url)
    repo.register_version(tenant_id="t", owner_id="u", name="a", content_hash="h", filename="a.txt",
                          chunks=[(0, "text", 1, "S", 3)])
    export_json(repo.engine, tmp_path / "db.json")
    import_json(repo.engine, tmp_path / "db.json")
    assert repo.document_for_hash("h").name == "a"
    d, v, _ = repo.register_version(tenant_id="t", owner_id="u", name="b", content_hash="h2", filename="b.txt",
                                    chunks=[(0, "text", 1, "S", 3)])
    assert repo.chunk_count(v.id) == 1  # sequences continue after restored ids


def test_restore_helpers_for_wrong_names(make_settings, tmp_path, server):
    from scripts import backup
    s = make_settings(database_url=f"sqlite:///{tmp_path / 'y.db'}")
    from database.repository import get_repository
    get_repository(s)
    made = backup.create(s, tmp_path / "backups")
    assert backup.resolve(tmp_path / "backups" / "latest") == made
    assert backup.resolve(tmp_path / "backups") == made
    with pytest.raises(SystemExit) as e:
        backup.resolve(tmp_path / "backups" / "rag-backup-20261110-150500")
    assert "Available backups" in str(e.value) and made.name in str(e.value)
    with pytest.raises(SystemExit, match="No backups"):
        backup.resolve(tmp_path / "empty" / "latest")


def test_indexes_restored_after_disk_wipe(make_settings, server, tmp_path, sample_pdf):
    """Streamlit Community Cloud: local disk is wiped on restart; DB is external and indexes are synced."""
    from core.documents import readable_indexes
    from ingestion.jobs import IngestionService
    from storage.remote import make_sync
    db = f"sqlite:///{tmp_path / 'external.db'}"
    remote = f"file://{tmp_path / 'remote'}"
    p = Principal("u1", "default", ("admin",))
    s1 = make_settings(database_url=db, data_dir=tmp_path / "disk1", auth_mode="users")
    res = IngestionService(s1).ingest(sample_pdf, "LM7805X.pdf", p)
    sync1 = make_sync(s1.data_dir, "", remote, 60)
    for prefix in (f"cache/{res.index.doc_id}", f"documents/{res.index.doc_id}"):
        sync1.track(prefix, pull=False)
    assert sync1.flush("test")
    from rag import pipeline
    pipeline._registry.clear()
    s2 = make_settings(database_url=db, data_dir=tmp_path / "disk2", auth_mode="users")  # fresh, empty disk
    assert readable_indexes(s2, p) == []  # without restore the index is gone
    sync2 = make_sync(s2.data_dir, "", remote, 60)
    pairs = readable_indexes(s2, p, before_open=lambda h: [sync2.track(f"cache/{h}"),
                                                          sync2.track(f"documents/{h}")])
    assert [ix.filename for _, ix in pairs] == ["LM7805X.pdf"] and pairs[0][1].nodes
