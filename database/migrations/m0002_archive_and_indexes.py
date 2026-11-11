"""Conversation archiving + composite indexes for tenant-scoped audit and job queries."""
from sqlalchemy import inspect, text

VERSION = 2
NAME = "conversation archiving, audit/job indexes"


def upgrade(conn) -> None:
    insp = inspect(conn)
    cols = {c["name"] for c in insp.get_columns("conversations")}
    if "archived" not in cols:
        conn.execute(text("ALTER TABLE conversations ADD COLUMN archived BOOLEAN NOT NULL DEFAULT FALSE"))
    existing = {i["name"] for t in ("audit_events", "ingestion_jobs") for i in insp.get_indexes(t)}
    if "ix_audit_tenant_ts" not in existing:
        conn.execute(text("CREATE INDEX ix_audit_tenant_ts ON audit_events (tenant_id, ts)"))
    if "ix_jobs_tenant_status" not in existing:
        conn.execute(text("CREATE INDEX ix_jobs_tenant_status ON ingestion_jobs (tenant_id, status)"))
