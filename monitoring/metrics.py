"""
Prometheus metrics (own registry; exposed at GET /metrics of the API). No labels ever contain user text,
document names or PII: only kinds, providers, models, stages, statuses and HTTP route templates.
"""
from __future__ import annotations

import re
from typing import Optional

try:
    from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
    AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    AVAILABLE = False

_LAT = (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 40, 90)
_ID = re.compile(r"/[0-9a-f]{16,}|/\d+")

if AVAILABLE:
    REGISTRY = CollectorRegistry(auto_describe=True)
    HTTP = Counter("rag_http_requests_total", "HTTP requests", ["method", "route", "status"], registry=REGISTRY)
    HTTP_LAT = Histogram("rag_http_request_seconds", "HTTP latency", ["route"], buckets=_LAT, registry=REGISTRY)
    QUERIES = Counter("rag_queries_total", "Answered questions", ["kind", "provider"], registry=REGISTRY)
    QUERY_LAT = Histogram("rag_query_seconds", "End-to-end question latency", buckets=_LAT, registry=REGISTRY)
    STAGE_LAT = Histogram("rag_stage_seconds", "Pipeline stage latency", ["stage"], buckets=_LAT, registry=REGISTRY)
    TOKENS = Counter("rag_llm_tokens_total", "LLM tokens", ["provider", "model"], registry=REGISTRY)
    COST = Counter("rag_llm_cost_usd_total", "LLM cost (USD)", ["provider", "model"], registry=REGISTRY)
    LLM = Counter("rag_llm_calls_total", "LLM calls by outcome", ["provider", "model", "outcome"], registry=REGISTRY)
    GUARD = Counter("rag_guardrail_events_total", "Guardrail decisions", ["event"], registry=REGISTRY)
    INGEST = Counter("rag_ingestion_jobs_total", "Ingestion jobs", ["status", "format"], registry=REGISTRY)
    INGEST_LAT = Histogram("rag_ingestion_seconds", "Ingestion duration", buckets=_LAT + (180, 600),
                           registry=REGISTRY)
    ERRORS = Counter("rag_errors_total", "Unhandled errors", ["component"], registry=REGISTRY)


def _route(path: str) -> str:
    return _ID.sub("/{id}", path)[:80]


def observe_http(method: str, path: str, status: int, ms: float) -> None:
    if AVAILABLE:
        HTTP.labels(method, _route(path), str(status)).inc()
        HTTP_LAT.labels(_route(path)).observe(ms / 1000)


def observe_query(result) -> None:
    if not AVAILABLE or result is None:
        return
    QUERIES.labels(result.kind or "unknown", result.provider or "none").inc()
    QUERY_LAT.observe(float(result.latency or 0.0))
    for stage, ms in (result.stage_latency or {}).items():
        if stage != "total":
            STAGE_LAT.labels(stage).observe(ms / 1000)
    if result.tokens:
        TOKENS.labels(result.provider or "none", (result.model or "")[:60]).inc(result.tokens)
    if result.cost:
        COST.labels(result.provider or "none", (result.model or "")[:60]).inc(result.cost)


def observe_llm(entry: str, outcome: str) -> None:
    if AVAILABLE:
        provider, _, model = entry.partition(":")
        LLM.labels(provider, model[:60], outcome).inc()


def observe_guardrail(event: str) -> None:
    if AVAILABLE:
        GUARD.labels(event).inc()


def observe_ingestion(status: str, fmt: str, seconds: Optional[float] = None) -> None:
    if AVAILABLE:
        INGEST.labels(status, fmt or "unknown").inc()
        if seconds is not None:
            INGEST_LAT.observe(seconds)


def observe_error(component: str) -> None:
    if AVAILABLE:
        ERRORS.labels(component).inc()


def render() -> str:
    return generate_latest(REGISTRY).decode() if AVAILABLE else "# prometheus_client not installed\n"
