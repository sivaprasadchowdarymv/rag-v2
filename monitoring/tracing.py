"""
Lightweight pipeline tracing.

    with span("retrieval", docs=3):
        ...
    spans = current_spans()      # [{"name", "ms", "attrs", "parent"}] for this request

Spans are always kept in-process for the current request (logged as a summary without any user text).
When OTEL_EXPORTER_OTLP_ENDPOINT is set and opentelemetry-sdk (+ OTLP exporter) is installed, every span is
also exported to an OpenTelemetry collector (Jaeger, Tempo, Grafana Cloud, ...). Attributes must never
contain questions, answers, document text or PII.
"""
from __future__ import annotations

import contextlib
import contextvars
import time
from typing import Any, Dict, Iterator, List, Optional

from config.settings import get_logger

log = get_logger("tracing")
_SPANS: contextvars.ContextVar[Optional[List[Dict[str, Any]]]] = contextvars.ContextVar("spans", default=None)
_STACK: contextvars.ContextVar[tuple] = contextvars.ContextVar("span_stack", default=())
_TRACER = None


def configure(endpoint: str = "", service: str = "rag-ultimate", exporter=None):
    """Enable OpenTelemetry export. `exporter` lets tests inject an in-memory exporter."""
    global _TRACER
    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
    except ImportError:
        if endpoint:
            log.warning("OTEL endpoint set but opentelemetry-sdk is not installed; tracing stays in-process")
        return None
    if exporter is None:
        if not endpoint:
            return None
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        except ImportError:
            log.warning("opentelemetry-exporter-otlp-proto-http is not installed; tracing stays in-process")
            return None
        exporter, processor = OTLPSpanExporter(endpoint=endpoint.rstrip("/") + "/v1/traces"), BatchSpanProcessor
    else:
        processor = SimpleSpanProcessor
    provider = TracerProvider(resource=Resource.create({"service.name": service}))
    provider.add_span_processor(processor(exporter))
    _TRACER = provider.get_tracer("rag-ultimate")
    return provider


@contextlib.contextmanager
def trace_request() -> Iterator[List[Dict[str, Any]]]:
    spans: List[Dict[str, Any]] = []
    token = _SPANS.set(spans)
    try:
        yield spans
    finally:
        _SPANS.reset(token)


@contextlib.contextmanager
def span(name: str, **attrs: Any) -> Iterator[Dict[str, Any]]:
    stack = _STACK.get()
    rec: Dict[str, Any] = {"name": name, "attrs": {k: v for k, v in attrs.items()
                                                   if isinstance(v, (int, float, bool, str))},
                           "parent": stack[-1] if stack else None}
    tok = _STACK.set(stack + (name,))
    t0 = time.perf_counter()
    otel_cm = _TRACER.start_as_current_span(name, attributes=rec["attrs"]) if _TRACER is not None else None
    otel_span = otel_cm.__enter__() if otel_cm is not None else None
    try:
        yield rec
    except Exception as exc:
        rec["error"] = type(exc).__name__
        if otel_span is not None:
            otel_span.record_exception(exc)
        raise
    finally:
        rec["ms"] = round((time.perf_counter() - t0) * 1000, 2)
        if otel_span is not None:
            for k, v in rec["attrs"].items():
                otel_span.set_attribute(k, v)
        if otel_cm is not None:
            otel_cm.__exit__(None, None, None)
        _STACK.reset(tok)
        bucket = _SPANS.get()
        if bucket is not None:
            bucket.append(rec)


def current_spans() -> List[Dict[str, Any]]:
    return list(_SPANS.get() or [])
