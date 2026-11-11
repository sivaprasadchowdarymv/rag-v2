"""
Repository: every database read/write of the application goes through here.

    repo = get_repository(settings)          # one per DATABASE_URL, migrations applied on first use
    doc, ver, status = repo.register_version(tenant_id="t1", owner_id="u1", name="LM7805.pdf",
                                             content_hash="ab12...", filename="LM7805.pdf", ...)

Objects are returned detached (expire_on_commit=False), so callers can read
them after the session closes. All tenant-scoped queries take tenant_id
explicitly: there is no "current tenant" global.
"""
from __future__ import annotations

import hashlib
import secrets
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from sqlalchemy import create_engine, delete, event, func, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from config.settings import Settings, get_logger
from database.migrations import migrate
from database.models import (
    ApiToken, AuditEvent, Chunk, Document, DocumentVersion, EvaluationRun, Feedback, IngestionJob, ModelConfig,
    PolicySetting, Tenant, User,
)

log = get_logger("db")

DEFAULT_TENANT = "default"


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        engine = create_engine(url, connect_args={"check_same_thread": False, "timeout": 30})

        @event.listens_for(engine, "connect")
        def _pragmas(dbapi_conn, _):  # pragma: no cover - trivial
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.close()
        return engine
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class Repository:
    def __init__(self, url: str):
        self.url = url
        self.engine = make_engine(url)
        migrate(self.engine)
        self._session = sessionmaker(self.engine, expire_on_commit=False)

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self._session() as s:
            try:
                yield s
                s.commit()
            except Exception:
                s.rollback()
                raise

    def health(self) -> bool:
        try:
            with self.engine.connect() as c:
                c.exec_driver_sql("SELECT 1")
            return True
        except Exception:
            log.exception("database health check failed")
            return False

    # ------------------------------------------------------------------ tenants
    def ensure_tenant(self, tenant_id: str, name: str = "") -> Tenant:
        with self.session() as s:
            t = s.get(Tenant, tenant_id)
            if t is None:
                t = Tenant(id=tenant_id, name=name or tenant_id)
                s.add(t)
            return t

    def list_tenants(self) -> List[Tenant]:
        with self.session() as s:
            return list(s.scalars(select(Tenant).order_by(Tenant.id)))

    # -------------------------------------------------------------------- users
    def create_user(self, tenant_id: str, username: str, password_hash: str, roles: Sequence[str]) -> User:
        self.ensure_tenant(tenant_id)
        with self.session() as s:
            u = User(tenant_id=tenant_id, username=username.strip().lower(), password_hash=password_hash,
                     roles=",".join(roles))
            s.add(u)
            return u

    def get_user(self, user_id: str) -> Optional[User]:
        with self.session() as s:
            return s.get(User, user_id)

    def get_user_by_name(self, username: str) -> Optional[User]:
        with self.session() as s:
            return s.scalar(select(User).where(User.username == (username or "").strip().lower()))

    def list_users(self, tenant_id: Optional[str] = None) -> List[User]:
        with self.session() as s:
            q = select(User).order_by(User.username)
            if tenant_id:
                q = q.where(User.tenant_id == tenant_id)
            return list(s.scalars(q))

    def update_user(self, user_id: str, **fields: Any) -> None:
        if "roles" in fields and not isinstance(fields["roles"], str):
            fields["roles"] = ",".join(fields["roles"])
        with self.session() as s:
            s.execute(update(User).where(User.id == user_id).values(**fields))

    def count_users(self) -> int:
        with self.session() as s:
            return int(s.scalar(select(func.count()).select_from(User)) or 0)

    # ------------------------------------------------------------------- tokens
    def create_token(self, user_id: str, name: str = "", ttl_s: Optional[float] = None) -> str:
        token = "rag_" + secrets.token_urlsafe(32)
        with self.session() as s:
            s.add(ApiToken(user_id=user_id, token_hash=hash_token(token), name=name,
                           expires_at=(time.time() + ttl_s) if ttl_s else None))
        return token  # shown once; only its hash is stored

    def resolve_token(self, token: str) -> Optional[User]:
        if not token:
            return None
        with self.session() as s:
            t = s.scalar(select(ApiToken).where(ApiToken.token_hash == hash_token(token)))
            if t is None or t.revoked or (t.expires_at and t.expires_at < time.time()):
                return None
            u = s.get(User, t.user_id)
            return u if u is not None and u.active else None

    def revoke_tokens(self, user_id: str) -> None:
        with self.session() as s:
            s.execute(update(ApiToken).where(ApiToken.user_id == user_id).values(revoked=True))

    # ---------------------------------------------------------------- documents
    def register_version(self, *, tenant_id: str, owner_id: str, name: str, content_hash: str, filename: str,
                         mime: str = "", size_bytes: int = 0, page_count: int = 0,
                         chunks: Iterable[Tuple[int, str, int, str, int]] = (), sensitivity: str = "internal",
                         allowed_roles: Sequence[str] = (), pii_detected: bool = False, storage_key: str = "",
                         meta: Optional[Dict[str, Any]] = None) -> Tuple[Document, DocumentVersion, str]:
        """Register an indexed file. Returns (document, version, status) with status:
        "new" (first version), "new_version" (same name, new content), "duplicate" (identical content already
        indexed in this tenant: nothing is stored twice)."""
        self.ensure_tenant(tenant_id)
        chunks = list(chunks)
        with self.session() as s:
            dup = s.execute(select(Document, DocumentVersion).join(DocumentVersion)
                            .where(Document.tenant_id == tenant_id, Document.status == "active",
                                   DocumentVersion.content_hash == content_hash,
                                   DocumentVersion.superseded.is_(False))).first()
            if dup is not None:
                return dup[0], dup[1], "duplicate"
            doc = s.scalar(select(Document).where(Document.tenant_id == tenant_id, Document.name == name))
            status = "new_version"
            if doc is None:
                doc = Document(tenant_id=tenant_id, owner_id=owner_id, name=name, sensitivity=sensitivity,
                               allowed_roles=",".join(allowed_roles))
                s.add(doc)
                s.flush()
                status = "new"
            elif doc.status == "deleted":
                doc.status, doc.sensitivity, doc.allowed_roles = "active", sensitivity, ",".join(allowed_roles)
                status = "new"
            if pii_detected and doc.sensitivity in ("public", "internal"):
                doc.sensitivity = "confidential"  # documents with PII are never treated as public
            s.execute(update(DocumentVersion).where(DocumentVersion.document_id == doc.id)
                      .values(superseded=True))
            version = (doc.current_version or 0) + 1
            ver = DocumentVersion(document_id=doc.id, version=version, content_hash=content_hash,
                                  filename=filename, mime=mime, size_bytes=size_bytes, page_count=page_count,
                                  chunk_count=len(chunks), pii_detected=pii_detected, storage_key=storage_key,
                                  meta=meta or {})
            s.add(ver)
            s.flush()
            s.add_all(Chunk(version_id=ver.id, node_index=i, type=t, page=p, section=(sec or "")[:255], chars=c)
                      for i, t, p, sec, c in chunks)
            doc.current_version, doc.updated_at = version, time.time()
            return doc, ver, status

    def get_document(self, doc_id: str) -> Optional[Document]:
        with self.session() as s:
            return s.get(Document, doc_id)

    def list_documents(self, tenant_id: str, include_deleted: bool = False) -> List[Document]:
        with self.session() as s:
            q = select(Document).where(Document.tenant_id == tenant_id).order_by(Document.updated_at.desc())
            if not include_deleted:
                q = q.where(Document.status == "active")
            return list(s.scalars(q))

    def versions(self, doc_id: str) -> List[DocumentVersion]:
        with self.session() as s:
            return list(s.scalars(select(DocumentVersion).where(DocumentVersion.document_id == doc_id)
                                  .order_by(DocumentVersion.version)))

    def current_version(self, doc_id: str) -> Optional[DocumentVersion]:
        with self.session() as s:
            return s.scalar(select(DocumentVersion).where(DocumentVersion.document_id == doc_id,
                                                          DocumentVersion.superseded.is_(False)))

    def document_for_hash(self, content_hash: str, tenant_id: Optional[str] = None,
                          current_only: bool = True) -> Optional[Document]:
        with self.session() as s:
            q = (select(Document).join(DocumentVersion).where(DocumentVersion.content_hash == content_hash,
                                                              Document.status == "active"))
            if current_only:
                q = q.where(DocumentVersion.superseded.is_(False))
            if tenant_id:
                q = q.where(Document.tenant_id == tenant_id)
            return s.scalars(q).first()

    def hash_known(self, content_hash: str) -> bool:
        with self.session() as s:
            return s.scalar(select(func.count()).select_from(DocumentVersion)
                            .where(DocumentVersion.content_hash == content_hash)) > 0

    def set_permissions(self, doc_id: str, sensitivity: Optional[str] = None,
                        allowed_roles: Optional[Sequence[str]] = None) -> None:
        values: Dict[str, Any] = {"updated_at": time.time()}
        if sensitivity:
            values["sensitivity"] = sensitivity
        if allowed_roles is not None:
            values["allowed_roles"] = ",".join(allowed_roles)
        with self.session() as s:
            s.execute(update(Document).where(Document.id == doc_id).values(**values))

    def delete_document(self, doc_id: str) -> List[str]:
        """Delete a document with all versions and chunk rows. Returns content hashes that no other
        document still uses, so the caller can drop their vector-store files (no stale vectors)."""
        with self.session() as s:
            hashes = set(s.scalars(select(DocumentVersion.content_hash)
                                   .where(DocumentVersion.document_id == doc_id)))
            ver_ids = list(s.scalars(select(DocumentVersion.id).where(DocumentVersion.document_id == doc_id)))
            if ver_ids:
                s.execute(delete(Chunk).where(Chunk.version_id.in_(ver_ids)))
            s.execute(delete(DocumentVersion).where(DocumentVersion.document_id == doc_id))
            s.execute(update(Document).where(Document.id == doc_id)
                      .values(status="deleted", current_version=0, updated_at=time.time()))
            still_used = set(s.scalars(select(DocumentVersion.content_hash)
                                       .where(DocumentVersion.content_hash.in_(hashes)))) if hashes else set()
            return sorted(hashes - still_used)

    def chunk_count(self, version_id: int) -> int:
        with self.session() as s:
            return int(s.scalar(select(func.count()).select_from(Chunk).where(Chunk.version_id == version_id)))

    # --------------------------------------------------------------------- jobs
    def create_job(self, tenant_id: str, user_id: str, filename: str) -> IngestionJob:
        with self.session() as s:
            job = IngestionJob(tenant_id=tenant_id, user_id=user_id, filename=filename)
            s.add(job)
            return job

    def update_job(self, job_id: str, **fields: Any) -> None:
        with self.session() as s:
            s.execute(update(IngestionJob).where(IngestionJob.id == job_id).values(**fields))

    def get_job(self, job_id: str) -> Optional[IngestionJob]:
        with self.session() as s:
            return s.get(IngestionJob, job_id)

    def list_jobs(self, tenant_id: str, status: Optional[str] = None, limit: int = 200) -> List[IngestionJob]:
        with self.session() as s:
            q = select(IngestionJob).where(IngestionJob.tenant_id == tenant_id)
            if status:
                q = q.where(IngestionJob.status == status)
            return list(s.scalars(q.order_by(IngestionJob.created_at.desc()).limit(limit)))

    # -------------------------------------------------------------------- audit
    def audit(self, tenant_id: str, user_id: str, action: str, target: str = "",
              detail: Optional[Dict[str, Any]] = None, request_id: str = "") -> None:
        try:
            with self.session() as s:
                s.add(AuditEvent(tenant_id=tenant_id, user_id=user_id, action=action, target=target[:255],
                                 detail=detail or {}, request_id=request_id))
        except Exception:  # auditing must never break a request; failure is logged
            log.exception("audit write failed (%s)", action)

    def list_audit(self, tenant_id: Optional[str] = None, action: Optional[str] = None,
                   limit: int = 500) -> List[AuditEvent]:
        with self.session() as s:
            q = select(AuditEvent).order_by(AuditEvent.ts.desc()).limit(limit)
            if tenant_id:
                q = q.where(AuditEvent.tenant_id == tenant_id)
            if action:
                q = q.where(AuditEvent.action == action)
            return list(s.scalars(q))

    # --------------------------------------------------------------- evaluation
    def save_evaluation(self, tenant_id: str, name: str, config: Dict, metrics: Dict, report: str) -> EvaluationRun:
        with self.session() as s:
            run = EvaluationRun(tenant_id=tenant_id, name=name, config=config, metrics=metrics, report=report)
            s.add(run)
            return run

    def list_evaluations(self, tenant_id: str, limit: int = 50) -> List[EvaluationRun]:
        with self.session() as s:
            return list(s.scalars(select(EvaluationRun).where(EvaluationRun.tenant_id == tenant_id)
                                  .order_by(EvaluationRun.created_at.desc()).limit(limit)))

    # ----------------------------------------------------------------- feedback
    def add_feedback(self, tenant_id: str, user_id: str, message_id: str, value: int,
                     reasons: Sequence[str] = (), comment: str = "") -> None:
        with self.session() as s:
            s.add(Feedback(tenant_id=tenant_id, user_id=user_id, message_id=message_id, value=value,
                           reasons=list(reasons), comment=comment))

    def feedback_summary(self, tenant_id: str) -> Dict[str, int]:
        with self.session() as s:
            rows = s.execute(select(Feedback.value, func.count()).where(Feedback.tenant_id == tenant_id)
                             .group_by(Feedback.value)).all()
        return {("up" if v > 0 else "down"): int(c) for v, c in rows}

    # ------------------------------------------------------------ model configs
    def set_model_chain(self, tenant_id: str, role: str, entries: Sequence[str]) -> None:
        with self.session() as s:
            s.execute(delete(ModelConfig).where(ModelConfig.tenant_id == tenant_id, ModelConfig.role == role))
            s.add_all(ModelConfig(tenant_id=tenant_id, role=role, entry=e, priority=i) for i, e in enumerate(entries))

    def model_chains(self, tenant_id: str) -> Dict[str, List[str]]:
        with self.session() as s:
            rows = s.scalars(select(ModelConfig).where(ModelConfig.tenant_id == tenant_id,
                                                       ModelConfig.enabled.is_(True))
                             .order_by(ModelConfig.role, ModelConfig.priority))
            out: Dict[str, List[str]] = {}
            for r in rows:
                out.setdefault(r.role, []).append(r.entry)
            return out

    # ------------------------------------------------------------------ policy
    def get_policy(self, tenant_id: str) -> Dict[str, Any]:
        with self.session() as s:
            p = s.get(PolicySetting, tenant_id)
            return dict(p.config) if p else {}

    def set_policy(self, tenant_id: str, config: Dict[str, Any]) -> None:
        self.ensure_tenant(tenant_id)
        with self.session() as s:
            p = s.get(PolicySetting, tenant_id)
            if p is None:
                s.add(PolicySetting(tenant_id=tenant_id, config=config))
            else:
                p.config, p.updated_at = dict(config), time.time()


_REPOS: Dict[str, Repository] = {}
_LOCK = threading.Lock()


def get_repository(settings: Settings) -> Repository:
    url = settings.db_url
    with _LOCK:
        repo = _REPOS.get(url)
        if repo is None:
            repo = _REPOS[url] = Repository(url)
        return repo


def reset_repositories() -> None:
    with _LOCK:
        for r in _REPOS.values():
            r.engine.dispose()
        _REPOS.clear()
