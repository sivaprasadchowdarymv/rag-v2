"""
REST API (FastAPI) — the same RAGService / IngestionService / PolicyEngine as the Streamlit UI, so every
guardrail and permission check is enforced server-side regardless of client.

    uvicorn api.main:app --host 0.0.0.0 --port 8000

Auth: AUTH_MODE=users → "Authorization: Bearer <token>" (POST /v1/auth/token). AUTH_MODE=none/password →
single local principal; bind the API to localhost in that mode (docker-compose does).
"""
from __future__ import annotations

import time
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from api.schemas import (MetadataUpdate, PermissionsRequest, QueryRequest, QueryResponse, RetrievalRequest,
                         SourceOut, TokenRequest, TokenResponse, UserCreate)
from config.logging_config import new_request_id, request_context
from config.settings import Settings, get_logger, load_settings, setup_logging
from core.service import AskRequest, Principal, RAGService
from database.repository import get_repository
from ingestion.jobs import IngestionService
from rag.pdf_parser import PdfProcessingError
from core.documents import readable_indexes
from security.auth import AuthError, AuthService, require

log = get_logger("api")


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or load_settings()
    setup_logging(settings.log_level, settings.log_format)
    app = FastAPI(title="RAG Ultimate API", version="2.2.1")
    app.state.settings = settings
    if settings.api_cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(settings.api_cors_origins),
                           allow_methods=["GET", "POST", "DELETE", "PATCH"], allow_headers=["*"])
    auth = AuthService(settings)
    auth.bootstrap_admin()
    if settings.otel_endpoint:
        from monitoring.tracing import configure
        configure(settings.otel_endpoint, "rag-ultimate-api")

    @app.middleware("http")
    async def request_id_mw(request: Request, call_next):
        rid = request.headers.get("x-request-id", "")[:32] or new_request_id()
        t0 = time.perf_counter()
        with request_context(rid, path=request.url.path):
            try:
                response = await call_next(request)
            except Exception:
                log.exception("unhandled API error")
                response = JSONResponse({"detail": "internal error", "request_id": rid}, status_code=500)
            ms = (time.perf_counter() - t0) * 1000
            response.headers["x-request-id"] = rid
            try:
                from monitoring.metrics import observe_http
                observe_http(request.method, request.url.path, response.status_code, ms)
            except Exception:
                pass
            log.info("%s %s %s %.0fms", request.method, request.url.path, response.status_code, ms)
            return response

    def principal(authorization: str = Header(default="")) -> Principal:
        if settings.auth_mode != "users":
            return Principal()
        token = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
        p = auth.principal_from_token(token)
        if p is None:
            raise HTTPException(401, "missing or invalid token", headers={"WWW-Authenticate": "Bearer"})
        return p

    def allow(p: Principal, perm: str) -> None:
        try:
            require(p, perm)
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc

    # ------------------------------------------------------------- health
    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz():
        from llm.model_router import ModelRouter
        db = get_repository(settings).health()
        providers = ModelRouter(settings).health()
        ready = db
        return JSONResponse({"status": "ready" if ready else "not_ready", "database": db, "providers": providers},
                            status_code=200 if ready else 503)

    @app.get("/metrics")
    def metrics():
        from monitoring.metrics import render
        if not settings.metrics_enabled:
            raise HTTPException(404, "metrics disabled")
        return PlainTextResponse(render(), media_type="text/plain; version=0.0.4")

    # --------------------------------------------------------------- auth
    @app.post("/v1/auth/token", response_model=TokenResponse)
    def token(body: TokenRequest):
        try:
            p = auth.authenticate(body.username, body.password)
        except AuthError as exc:
            raise HTTPException(401, exc.user_message) from exc
        return TokenResponse(token=auth.issue_token(p), user_id=p.user_id, tenant_id=p.tenant_id, roles=list(p.roles))

    @app.post("/v1/users")
    def create_user(body: UserCreate, p: Principal = Depends(principal)):
        allow(p, "manage_users")
        try:
            u = auth.create_user(body.username, body.password, body.roles, p.tenant_id, actor=p)
        except AuthError as exc:
            raise HTTPException(400, exc.user_message) from exc
        return {"id": u.id, "username": u.username, "roles": u.role_list}

    # ---------------------------------------------------------- documents
    @app.get("/v1/documents")
    def list_documents(p: Principal = Depends(principal)):
        return [{"id": d.id, "name": d.name, "version": d.current_version, "sensitivity": d.sensitivity,
                 "pages": ix.page_count} for d, ix in readable_indexes(settings, p)]

    @app.post("/v1/documents")
    async def upload(file: UploadFile = File(...), sensitivity: str = Form("internal"), allowed_roles: str = Form(""),
                     p: Principal = Depends(principal)):
        allow(p, "upload")
        if sensitivity not in ("public", "internal", "confidential", "restricted"):
            raise HTTPException(422, "invalid sensitivity")
        data = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
        roles = [r for r in allowed_roles.split(",") if r.strip()]
        try:
            res = IngestionService(settings).ingest(data, file.filename or "upload", p, sensitivity, roles)
        except PdfProcessingError as exc:
            raise HTTPException(422, exc.user_message) from exc
        return {"id": res.document_id, "version": res.version, "status": res.status, "job_id": res.job_id,
                "sensitivity": res.sensitivity, "warnings": res.warnings}

    @app.delete("/v1/documents/{doc_id}")
    def delete(doc_id: str, p: Principal = Depends(principal)):
        allow(p, "delete_own")
        try:
            removed = IngestionService(settings).delete(doc_id, p)
        except PermissionError as exc:
            raise HTTPException(404 if "not found" in str(exc) else 403, str(exc)) from exc
        return {"deleted": doc_id, "vector_files_removed": len(removed)}

    @app.patch("/v1/documents/{doc_id}/permissions")
    def permissions(doc_id: str, body: PermissionsRequest, p: Principal = Depends(principal)):
        allow(p, "relabel")
        repo = get_repository(settings)
        doc = repo.get_document(doc_id)
        if doc is None or doc.tenant_id != p.tenant_id:
            raise HTTPException(404, "document not found")
        if not (p.has("admin") or doc.owner_id == p.user_id):
            raise HTTPException(403, "only the owner or an admin can change permissions")
        repo.set_permissions(doc_id, body.sensitivity, body.allowed_roles)
        repo.audit(p.tenant_id, p.user_id, "permissions_changed", doc_id, body.model_dump())
        return {"ok": True}

    # ------------------------------------------------------- metadata (2.1)
    def _meta_view(doc, ix, p):
        from storage.metadata_store import MetadataStore, scope_for

        ver = get_repository(settings).current_version(doc.id)
        info = {"size_bytes": getattr(ver, "size_bytes", 0), "sensitivity": doc.sensitivity,
                "uploaded_at": getattr(ver, "created_at", 0), "owner": doc.owner_id}
        v = MetadataStore(settings).view(ix, scope_for(p, auth_mode=settings.auth_mode), info)
        return {"id": doc.id, "name": doc.name, "version": doc.current_version, **v}

    @app.get("/v1/documents/{doc_id}/metadata")
    def get_metadata(doc_id: str, p: Principal = Depends(principal)):
        allow(p, "read")
        pairs = readable_indexes(settings, p, [doc_id])
        if not pairs:
            raise HTTPException(404, "document not found")
        return _meta_view(*pairs[0], p)

    @app.patch("/v1/documents/{doc_id}/metadata")
    def edit_metadata(doc_id: str, body: MetadataUpdate, p: Principal = Depends(principal)):
        allow(p, "relabel")
        from storage.metadata_store import MetadataError, MetadataStore, scope_for

        pairs = readable_indexes(settings, p, [doc_id])
        if not pairs:
            raise HTTPException(404, "document not found")
        doc, ix = pairs[0]
        if not (p.has("admin") or doc.owner_id == p.user_id):
            raise HTTPException(403, "only the owner or an admin can edit metadata")
        changes = body.model_dump(exclude_none=True)
        try:
            MetadataStore(settings).update_user(ix.doc_id, scope_for(p, auth_mode=settings.auth_mode), changes,
                                                actor=p.user_id)
        except MetadataError as exc:
            raise HTTPException(422, str(exc)) from exc
        get_repository(settings).audit(p.tenant_id, p.user_id, "metadata_changed", doc_id,
                                       {"fields": sorted(changes)})
        return _meta_view(doc, ix, p)

    @app.get("/v1/metadata/search")
    def search_metadata(q: str = "", tag: str = "", category: str = "", part_number: str = "",
                        p: Principal = Depends(principal)):
        allow(p, "read")
        from rag.metadata_intel import matches
        from storage.metadata_store import MetadataStore, scope_for

        store, scope, out = MetadataStore(settings), scope_for(p, auth_mode=settings.auth_mode), []
        for doc, ix in readable_indexes(settings, p):
            prof, usr = store.profile(ix), store.user(ix.doc_id, scope)
            if matches(prof, usr, query=q[:200], tags=[t for t in tag.split(",") if t], category=category,
                       part_number=part_number):
                out.append({"id": doc.id, "name": doc.name, "version": doc.current_version})
        return out

    @app.get("/v1/jobs/{job_id}")
    def job(job_id: str, p: Principal = Depends(principal)):
        j = get_repository(settings).get_job(job_id)
        if j is None or j.tenant_id != p.tenant_id:
            raise HTTPException(404, "job not found")
        return {k: getattr(j, k) for k in ("id", "status", "progress", "attempts", "message", "error", "filename")}

    # -------------------------------------------------------------- query
    @app.post("/v1/query", response_model=QueryResponse)
    def query(body: QueryRequest, p: Principal = Depends(principal)):
        allow(p, "ask")
        pairs = readable_indexes(settings, p, body.document_ids)
        req = AskRequest(question=body.question, indexes=[ix for _, ix in pairs], session_key=body.session_id,
                         metadata_filter=body.filters, model_override=body.model or "", llm_mode=body.llm_mode or "")
        r = RAGService(settings).ask(req, p)
        return QueryResponse(
            request_id=r.request_id, kind=r.kind, answer=r.answer,
            sources=[SourceOut(label=s.label, document=s.doc_name, page=s.page, section=s.section,
                               snippet=s.snippet[:400]) for s in r.sources],
            warnings=r.warnings, verification=r.verification or {}, routing=r.routing or {},
            agent_trace=r.agent_trace or [], security=r.security or {}, provider=r.provider, model=r.model,
            tokens=r.tokens, cost_usd=r.cost, latency_s=r.latency, cached=bool(r.cached),
            cache={k: v for k, v in (r.cache_info or {}).items() if not k.startswith("_")})

    @app.post("/v1/retrieval/debug")
    def retrieval_debug(body: RetrievalRequest, p: Principal = Depends(principal)):
        allow(p, "read")
        from retrieval.hybrid_search import HybridSearcher, MetadataFilter
        indexes = [ix for _, ix in readable_indexes(settings, p)]
        _, debug = HybridSearcher(settings).search(body.query, indexes, body.k, body.mode,
                                                   MetadataFilter.from_dict(body.filters))
        return debug

    # ---------------------------------------------------------- cache (2.1)
    @app.get("/v1/cache/stats")
    def cache_stats(p: Principal = Depends(principal)):
        allow(p, "manage_policy")
        from ops import answer_cache
        return {"answers": answer_cache.for_settings(settings).stats(tenant=p.tenant_id),
                "retrieval": answer_cache.retrieval_cache_stats()}

    @app.delete("/v1/cache")
    def cache_clear(p: Principal = Depends(principal)):
        allow(p, "manage_policy")
        from ops import answer_cache
        n = answer_cache.for_settings(settings).clear(tenant=p.tenant_id)  # only this tenant's answers
        get_repository(settings).audit(p.tenant_id, p.user_id, "cache_cleared", "answers", {"removed": n})
        return {"removed": n}

    @app.get("/v1/audit")
    def audit(limit: int = 200, p: Principal = Depends(principal)):
        allow(p, "view_audit")
        return [{"ts": e.ts, "user": e.user_id, "action": e.action, "target": e.target, "detail": e.detail,
                 "request_id": e.request_id} for e in get_repository(settings).list_audit(p.tenant_id, limit=limit)]

    return app


def __getattr__(name: str):  # lazy module-level `app` for `uvicorn api.main:app`
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
