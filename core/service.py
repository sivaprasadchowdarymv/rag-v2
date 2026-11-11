"""
RAGService: the single server-side entry point for answering questions.

Both the Streamlit UI and the FastAPI server call this class, so every
security, rate-limit, routing and audit rule is applied in one place,
independent of the user interface.

    service = RAGService(settings)
    result = service.ask(AskRequest(question="...", indexes=[...]), principal)

Dependencies (router factory, limiter, policy engine, repository, workflow)
are injected so tests can replace any of them.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from config.logging_config import request_context
from config.settings import Settings, get_logger
from llm.model_router import ModelRouter
from ops import telemetry
from ops.guard import LIMITER, RateLimiter
from rag.models import AgentStep, DocumentIndex, QueryResult

log = get_logger("service")


@dataclass
class Principal:
    """Who is asking. `roles` drive RBAC; `tenant_id` scopes every lookup."""

    user_id: str = "anonymous"
    tenant_id: str = "default"
    roles: Sequence[str] = ("admin",)  # auth_mode=none: the single local user owns everything
    username: str = "local"

    def has(self, *roles: str) -> bool:
        return any(r in self.roles for r in roles)


@dataclass
class AskRequest:
    question: str
    indexes: Sequence[DocumentIndex]
    memory: str = ""
    previous_question: Optional[str] = None
    preferences: str = ""
    session_key: str = "default"
    on_step: Optional[Callable[[AgentStep], None]] = None
    metadata_filter: Optional[dict] = None
    model_override: str = ""        # "provider:model" chosen in the UI (still policy-checked)
    llm_mode: str = ""              # local | cloud | hybrid ("" = settings)
    extra: dict = field(default_factory=dict)


class RAGService:
    def __init__(self, settings: Settings, router_factory: Optional[Callable[[Settings], ModelRouter]] = None,
                 limiter: Optional[RateLimiter] = None, policy=None, repository=None, workflow=None):
        self.settings = settings
        self.router_factory = router_factory or ModelRouter
        self.limiter = limiter or LIMITER
        self._policy = policy
        self._repository = repository
        self._workflow = workflow

    # ------------------------------------------------------------ dependencies
    @property
    def policy(self):
        if self._policy is None:
            from security.policy_engine import PolicyEngine

            self._policy = PolicyEngine.from_settings(self.settings, self.repository)
        return self._policy

    @property
    def repository(self):
        if self._repository is None:
            from database.repository import get_repository

            self._repository = get_repository(self.settings)
        return self._repository

    @property
    def workflow(self):
        if self._workflow is None:
            from agents.workflow import AgentWorkflow

            self._workflow = AgentWorkflow(self.settings)
        return self._workflow

    # ---------------------------------------------------------------- answering
    def _rate_limited(self, question: str, wait: float) -> QueryResult:
        telemetry.record(self.settings.data_dir, question=question, latency_s=0, tokens=0, llm_calls=0, provider="",
                         cached=False, error=False, kind="rate_limited", rate_limited=True)
        return QueryResult(kind="not_found", answer=f"⏳ Too many questions right now. Please wait about "
                                                    f"{int(wait) + 1} seconds and ask again.", question=question)

    def ask(self, req: AskRequest, principal: Optional[Principal] = None) -> QueryResult:
        principal = principal or Principal()
        from core.model_config import apply_tenant_overrides

        self.settings = s = apply_tenant_overrides(self.settings, principal.tenant_id, self._repository)
        from config.logging_config import request_id as bound_request_id

        # reuse the id bound by the API middleware (X-Request-ID) so one request has one id end to end
        with request_context(bound_request_id() or None, user=principal.user_id, tenant=principal.tenant_id) as rid:
            wait = self.limiter.allow(f"{principal.tenant_id}:{req.session_key}", s.rate_limit_per_min,
                                      s.global_rate_limit_per_min)
            if wait:
                return self._rate_limited(req.question, wait)
            from monitoring import metrics
            from monitoring.tracing import trace_request

            t0, error, result = time.perf_counter(), False, None
            try:
                with trace_request() as spans:
                    result = self._answer(req, principal)
                result.request_id = rid
                metrics.observe_query(result)
                log.info("query done kind=%s provider=%s model=%s tokens=%d cost=%.6f latency=%.2fs spans=%s",
                         result.kind, result.provider or "-", result.model or "-", result.tokens or 0,
                         result.cost or 0.0, time.perf_counter() - t0,
                         ",".join(f"{sp['name']}:{sp['ms']:.0f}ms" for sp in spans))  # never question text
                return result
            except Exception:
                error = True
                metrics.observe_error("service")
                log.exception("ask failed")
                raise
            finally:
                r = result
                telemetry.record(s.data_dir, question=req.question, latency_s=time.perf_counter() - t0,
                                 tokens=getattr(r, "tokens", 0) or 0, llm_calls=getattr(r, "llm_calls", 0) or 0,
                                 provider=getattr(r, "provider", "") or "", cached=bool(getattr(r, "cached", False)),
                                 error=error or (r is not None and r.used_llm is False and r.kind == "answer"
                                                 and any("without an AI answer" in w for w in r.warnings)),
                                 kind=getattr(r, "kind", "error"), cost_usd=getattr(r, "cost", 0.0) or 0.0,
                                 model=getattr(r, "model", "") or "",
                                 saved_tokens=int((getattr(r, "cache_info", None) or {}).get("saved_tokens") or 0)
                                 if getattr(r, "cached", False) else 0,
                                 saved_cost=float((getattr(r, "cache_info", None) or {}).get("saved_cost") or 0.0)
                                 if getattr(r, "cached", False) else 0.0,
                                 verdict=str((getattr(r, "verification", None) or {}).get("verdict") or "")
                                 if getattr(r, "kind", "") == "answer" else "",
                                 tenant=principal.tenant_id, user=principal.user_id)

    def _answer(self, req: AskRequest, principal: Principal) -> QueryResult:
        """Input policy -> authorised documents -> agent workflow -> output policy -> audit."""
        from monitoring.tracing import span

        with span("policy.input"):
            decision = self.policy.check_input(req.question, principal)  # audits blocked/flagged input itself
        if decision.blocked:
            return QueryResult(kind=decision.kind, answer=decision.message, question=req.question,
                               warnings=decision.warnings, security={"input": decision.as_dict()})
        question = decision.text  # possibly redacted
        with span("authorize", requested=len(req.indexes)):
            indexes = self.authorized_indexes(req.indexes, principal)
        req.extra = {**(req.extra or {}), "cache_scope": self.cache_scope(req, principal)}
        if req.metadata_filter:  # 2.1: document-level metadata criteria (tags, category, part number) -> doc_ids
            from storage.metadata_store import resolve_doc_filter, scope_for

            req.metadata_filter = resolve_doc_filter(
                req.metadata_filter, indexes, self.settings,
                scope_for(principal, str((req.extra or {}).get("workspace", "")), self.settings.auth_mode))
        with span("workflow", documents=len(indexes)):
            result = self.workflow.run(question, indexes, self._router(req, indexes), req)
        with span("policy.output"):
            out = self.policy.check_output(result, indexes, principal)
        if decision.warnings:
            out.warnings = list(dict.fromkeys(decision.warnings + out.warnings))
        out.security = {"input": decision.as_dict(), **(out.security or {})}
        if out.security.get("output", {}).get("actions"):
            self.policy.audit(principal, "output_filtered", out.security["output"])
        return out

    def _router(self, req: AskRequest, indexes: Sequence[DocumentIndex]) -> ModelRouter:
        import dataclasses

        s = self.settings
        over = {}
        if req.llm_mode and req.llm_mode != s.llm_mode:
            over["llm_mode"] = req.llm_mode
        if req.model_override:
            over["manual_model"] = req.model_override
        s = dataclasses.replace(s, **over) if over else s
        router = self.router_factory(s)
        sensitive = any(self.policy.is_sensitive(ix) for ix in indexes)
        if hasattr(router, "set_sensitivity"):
            router.set_sensitivity(sensitive)
        return router

    # -------------------------------------------------------------- answer cache
    def cache_scope(self, req: AskRequest, principal: Principal) -> str:
        """Security boundary of cached answers: tenant first (used for per-tenant clearing), then the user (or,
        with ANSWER_CACHE_SCOPE=tenant, everyone with the same roles), the private workspace (AUTH_MODE=none has
        one anonymous principal for every workspace link), the model choice and a fingerprint of the tenant's
        security policy. Document versions, filters and answer settings are added by the pipeline."""
        from ops.answer_cache import fingerprint

        s = self.settings
        who = principal.user_id if s.answer_cache_scope == "user" else "*"
        try:
            policy_fp = fingerprint(sorted(self.policy.config(principal.tenant_id).items()))[:16]
        except Exception:  # unreadable policy: unique scope = no sharing
            import secrets

            policy_fp = secrets.token_hex(8)
        sec = (s.allow_sensitive_cloud, tuple(s.cloud_providers_allowed))
        return "|".join([principal.tenant_id, f"u={who}", "r=" + ",".join(sorted(principal.roles)),
                         f"ws={(req.extra or {}).get('workspace', '')}", f"m={req.model_override}",
                         f"mode={req.llm_mode or s.llm_mode}", f"p={policy_fp}", f"s={fingerprint(sec)[:8]}"])

    # ----------------------------------------------------------- authorisation
    def authorized_indexes(self, indexes: Sequence[DocumentIndex], principal: Principal) -> List[DocumentIndex]:
        """Retrieval-time authorisation: drop every index the principal may not read."""
        allowed = []
        for ix in indexes:
            if self.policy.can_read(principal, ix.doc_id):
                allowed.append(ix)
            else:
                self.policy.audit(principal, "retrieval_denied", {"doc_id": ix.doc_id})
        return allowed
