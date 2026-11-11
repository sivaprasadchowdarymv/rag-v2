"""
Structured logging: request IDs, JSON or text output, redaction of secrets and PII.

    from config.logging_config import request_context, configure
    configure("rag_ultimate", "INFO", "json")
    with request_context(user="u1", tenant="t1"):
        log.info("query answered")      # -> {"request_id": "...", "user": "u1", ...}

Redaction runs on every record (message *and* args) so a stray key, e-mail or
phone number in an exception message never reaches the log sink.
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import re
import sys
import time
import uuid
from typing import Any, Dict, Iterator, Optional

_CTX: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar("log_ctx", default=None)

_REDACTIONS = [
    (re.compile(r"\b(sk-(?:ant-|or-|proj-)?[A-Za-z0-9_\-]{16,}|gsk_[A-Za-z0-9]{16,}|hf_[A-Za-z0-9]{16,}"
                r"|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,})\b"), "[SECRET]"),
    (re.compile(r"(?i)\b(password|passwd|secret|api[_-]?key|token)\s*[=:]\s*\S+"), r"\1=[SECRET]"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),

]
_PHONE = re.compile(r"(?<![\w.])(?:\+|\()?\d[\d\s().-]{8,18}\d(?![\w.])")


def _phone(m: "re.Match[str]") -> str:
    digits = sum(ch.isdigit() for ch in m.group())
    return "[PHONE]" if 10 <= digits <= 13 else m.group()


def redact(text: str) -> str:
    for pattern, repl in _REDACTIONS:
        text = pattern.sub(repl, text)
    return _PHONE.sub(_phone, text)


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def current_context() -> Dict[str, Any]:
    return dict(_CTX.get() or {})


def request_id() -> str:
    return str((_CTX.get() or {}).get("request_id", ""))


@contextlib.contextmanager
def request_context(request_id: Optional[str] = None, **fields: Any) -> Iterator[str]:
    """Bind a request ID (and e.g. user/tenant ids, never names or text) to all logs in this block."""
    rid = request_id or new_request_id()
    token = _CTX.set({**(_CTX.get() or {}), "request_id": rid, **{k: v for k, v in fields.items() if v is not None}})
    try:
        yield rid
    finally:
        _CTX.reset(token)


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:  # bad format args: keep the raw template
            msg = str(record.msg)
        record.msg, record.args = redact(msg), ()
        for k, v in current_context().items():
            if not hasattr(record, k):
                setattr(record, k, v)
        return True


class JsonFormatter(logging.Formatter):
    _STD = set(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {"message", "asctime"}

    def format(self, record: logging.LogRecord) -> str:
        out: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname, "logger": record.name, "message": record.getMessage(),
        }
        out.update({k: v for k, v in vars(record).items() if k not in self._STD and not k.startswith("_")})
        if record.exc_info:
            out["exception"] = redact(self.formatException(record.exc_info))
        return json.dumps(out, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)-5s %(name)s  %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        rid = getattr(record, "request_id", "")
        text = f"{base}  [rid={rid}]" if rid else base
        return redact(text) if record.exc_info else text


def configure(logger_name: str, level: str = "INFO", fmt: str = "text") -> logging.Logger:
    logger = logging.getLogger(logger_name)
    handler = next((h for h in logger.handlers if getattr(h, "_rag_handler", False)), None)
    if handler is None:
        handler = logging.StreamHandler(sys.stdout)
        handler._rag_handler = True  # type: ignore[attr-defined]
        handler.addFilter(RedactingFilter())
        logger.addHandler(handler)
        logger.propagate = False
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    return logger
