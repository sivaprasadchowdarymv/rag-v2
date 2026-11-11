"""
Ingestion service: validate -> store original -> parse/chunk/embed (rag.pipeline.get_index) -> register
document version (dedup, versioning, sensitivity) -> clean up superseded vectors.

    svc = IngestionService(settings)
    res = svc.ingest(data, "report.docx", principal)            # synchronous (UI)
    job = svc.submit(data, "report.docx", principal)            # background thread pool
    svc.wait(job, timeout=60); svc.repo.get_job(job)            # status, progress, attempts, error

Retries: transient failures (embedding/network/IO) are retried INGEST_RETRIES times with
exponential backoff; validation errors (bad/unsafe file) fail at once. Every job — and so every
failed document — is recorded in the ingestion_jobs table with its error message.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from config.settings import Settings, get_logger
from database.repository import get_repository
from ingestion.validation import IngestionError, validate_upload
from rag.models import DocumentIndex
from rag.pdf_parser import PdfProcessingError
from rag.pipeline import clear_cache, get_index
from security.pii_detector import contains_pii
from storage.cache import compute_doc_id, sanitize_filename
from storage.objects import make_object_store, object_key

log = get_logger("ingestion")
ProgressFn = Optional[Callable[[float, str], None]]
PII_SCAN_CHARS = 200_000


@dataclass
class IngestResult:
    index: DocumentIndex
    document_id: str
    version: int
    status: str                       # new | new_version | duplicate
    warnings: List[str] = field(default_factory=list)
    superseded_hashes: List[str] = field(default_factory=list)
    job_id: str = ""
    sensitivity: str = "internal"
    pii_detected: bool = False


class IngestionService:
    _pools: Dict[int, ThreadPoolExecutor] = {}
    _futures: Dict[str, Future] = {}
    _lock = threading.RLock()

    def __init__(self, settings: Settings, repository=None, object_store=None):
        self.settings = settings
        self.repo = repository or get_repository(settings)
        self.store = object_store if object_store is not None else make_object_store(settings)

    # ------------------------------------------------------------ synchronous
    def ingest(self, data: bytes, filename: str, principal, sensitivity: str = "internal",
               allowed_roles: Sequence[str] = (), progress: ProgressFn = None, vision=None,
               job_id: Optional[str] = None) -> IngestResult:
        from security.auth import has_permission

        if not has_permission(principal, "upload"):  # enforced here, so UI, API and jobs all obey it
            self.repo.audit(principal.tenant_id, principal.user_id, "upload_denied", sanitize_filename(filename))
            raise IngestionError("Your role is not allowed to upload documents. Ask an editor or admin.")
        filename = sanitize_filename(filename)
        job_id = job_id or self.repo.create_job(principal.tenant_id, principal.user_id, filename).id
        t0 = time.time()
        self.repo.update_job(job_id, status="running", started_at=t0, message="Validating")

        def report(f: float, msg: str) -> None:
            self.repo.update_job(job_id, progress=round(min(max(f, 0.0), 1.0), 3), message=msg[:250])
            if progress:
                progress(f, msg)

        attempts, last_exc = 0, None
        try:
            info = validate_upload(data, filename, self.settings)  # permanent errors: no retry
            while attempts <= self.settings.ingest_retries:
                attempts += 1
                self.repo.update_job(job_id, attempts=attempts)
                try:
                    result = self._ingest_once(data, filename, info, principal, sensitivity, allowed_roles,
                                               report, vision)
                    result.job_id = job_id
                    from monitoring.metrics import observe_ingestion
                    observe_ingestion("succeeded", info.kind, time.time() - t0)
                    self.repo.update_job(job_id, status="succeeded", progress=1.0, finished_at=time.time(),
                                         document_id=result.document_id,
                                         message=f"{result.status} (v{result.version})")
                    self.repo.audit(principal.tenant_id, principal.user_id, "document_ingested",
                                    result.document_id, {"status": result.status, "version": result.version,
                                                         "format": info.kind, "pii": result.pii_detected})
                    return result
                except (IngestionError, PdfProcessingError) as exc:
                    if getattr(exc, "permanent", True):
                        raise
                    last_exc = exc
                except Exception as exc:  # transient (IO, embedding service, ...): retry with backoff
                    last_exc = exc
                    log.warning("ingestion attempt %d of %s failed: %s", attempts, filename, type(exc).__name__)
                if attempts <= self.settings.ingest_retries:
                    time.sleep(min(0.25 * 2 ** (attempts - 1), 4.0))
            raise IngestionError("The file could not be processed after several attempts. Please try again later.",
                                 permanent=False) from last_exc
        except PdfProcessingError as exc:
            from monitoring.metrics import observe_ingestion
            observe_ingestion("failed", filename.rsplit(".", 1)[-1].lower(), time.time() - t0)
            self.repo.update_job(job_id, status="failed", finished_at=time.time(), error=exc.user_message[:2000],
                                 message="Failed")
            self.repo.audit(principal.tenant_id, principal.user_id, "document_failed", filename,
                            {"error": exc.user_message[:300]})
            raise

    def _ingest_once(self, data, filename, info, principal, sensitivity, allowed_roles, report,
                     vision) -> IngestResult:
        content_hash = compute_doc_id(data)
        key = ""
        if self.settings.keep_originals:
            key = object_key(principal.tenant_id, content_hash, filename)
            if not self.store.exists(key):
                self.store.put(key, data)
        index, warnings = get_index(data, filename, self.settings, progress=report, vision=vision)
        sample = "\n".join(n.raw_content for n in index.nodes)[:PII_SCAN_CHARS]
        pii = contains_pii(sample, min_severity=2)
        chunks = [(i, n.type, int(n.page or 0), n.section, len(n.raw_content)) for i, n in enumerate(index.nodes)]
        doc, ver, status = self.repo.register_version(
            tenant_id=principal.tenant_id, owner_id=principal.user_id, name=filename, content_hash=content_hash,
            filename=filename, mime=info.mime, size_bytes=info.size_bytes, page_count=index.page_count,
            chunks=chunks, sensitivity=sensitivity, allowed_roles=allowed_roles, pii_detected=pii,
            storage_key=key, meta={"format": info.kind, "title": index.doc_meta.get("title", "")})
        superseded: List[str] = []
        if status == "new_version":  # drop vectors of older versions nobody else uses (no stale retrieval)
            for v in self.repo.versions(doc.id):
                if v.superseded and v.content_hash != content_hash \
                        and self.repo.document_for_hash(v.content_hash) is None:
                    clear_cache(v.content_hash, self.settings)
                    superseded.append(v.content_hash)
        try:  # 2.1: metadata profile (deterministic, local); a failure never fails the upload
            from storage.metadata_store import MetadataStore

            MetadataStore(self.settings).profile(index, {"size_bytes": info.size_bytes, "sensitivity": doc.sensitivity,
                                                         "uploaded_at": ver.created_at, "owner": doc.owner_id})
        except Exception:
            log.warning("metadata profile could not be built for %s", filename)
        if status == "new_version":  # cached answers built on an older version are never served again
            from ops.answer_cache import invalidate_docs

            invalidate_docs({v.content_hash for v in self.repo.versions(doc.id) if v.content_hash != content_hash})
        if pii:
            warnings = warnings + ["Personal data was detected in this document, so it is classified as "
                                   "confidential (local models only unless an admin allows cloud use)."]
        return IngestResult(index, doc.id, ver.version, status, warnings, superseded, "", doc.sensitivity, pii)

    # ----------------------------------------------------------- background
    def _pool(self) -> ThreadPoolExecutor:
        with self._lock:
            n = self.settings.ingest_workers
            if n not in self._pools:
                self._pools[n] = ThreadPoolExecutor(max_workers=n, thread_name_prefix="ingest")
            return self._pools[n]

    def submit(self, data: bytes, filename: str, principal, sensitivity: str = "internal",
               allowed_roles: Sequence[str] = (), on_done: Optional[Callable[[IngestResult], None]] = None) -> str:
        job = self.repo.create_job(principal.tenant_id, principal.user_id, sanitize_filename(filename))

        def run() -> Optional[IngestResult]:
            try:
                res = self.ingest(data, filename, principal, sensitivity, allowed_roles, job_id=job.id)
            except PdfProcessingError:
                return None  # recorded as a failed job
            except Exception:  # unexpected: record it, never kill the worker
                log.exception("background ingestion crashed")
                self.repo.update_job(job.id, status="failed", error="internal error", finished_at=time.time())
                return None
            if on_done:
                on_done(res)
            return res

        with self._lock:
            self._futures[job.id] = self._pool().submit(run)
        return job.id

    def wait(self, job_id: str, timeout: float = 120.0) -> Optional[IngestResult]:
        fut = self._futures.get(job_id)
        return fut.result(timeout=timeout) if fut else None

    # -------------------------------------------------------------- delete
    def delete(self, document_id: str, principal) -> List[str]:
        doc = self.repo.get_document(document_id)
        if doc is None or doc.tenant_id != principal.tenant_id:
            raise PermissionError("document not found")
        if not (principal.has("admin") or (principal.has("editor") and doc.owner_id == principal.user_id)):
            raise PermissionError("not allowed to delete this document")
        keys = [v.storage_key for v in self.repo.versions(document_id) if v.storage_key]
        hashes = [v.content_hash for v in self.repo.versions(document_id)]
        orphans = self.repo.delete_document(document_id)
        from ops.answer_cache import invalidate_docs

        invalidate_docs(hashes)  # no cached answer may cite a deleted document
        for h in orphans:  # vectors + parsed nodes of every version: nothing stale remains
            clear_cache(h, self.settings)
        for k in keys:
            h = k.rsplit("/", 1)[-1].split(".")[0]
            if h in orphans:
                self.store.delete(k)
        self.repo.audit(principal.tenant_id, principal.user_id, "document_deleted", document_id,
                        {"name": doc.name, "versions_removed": len(keys)})
        return orphans
