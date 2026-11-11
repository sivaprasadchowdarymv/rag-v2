"""Initial schema: every table of database/models.py as of release 2.0."""
VERSION = 1
NAME = "initial schema"

TABLES = ("tenants", "users", "api_tokens", "documents", "document_versions", "chunks", "ingestion_jobs",
          "conversations", "messages", "feedback", "model_configs", "evaluation_runs", "audit_events",
          "policy_settings")


LATER_COLUMNS = {"conversations": {"archived"}}  # added by later migrations


def upgrade(conn) -> None:
    """Create the v1 tables from a copy of the model metadata without columns that later migrations add,
    so upgrading an old database and creating a new one end in the same schema."""
    from sqlalchemy import MetaData

    from database.models import Base

    meta = MetaData()
    for name in TABLES:
        t = Base.metadata.tables[name].to_metadata(meta)
        for col in LATER_COLUMNS.get(name, ()):
            t._columns.remove(t.c[col])
    meta.create_all(conn)
