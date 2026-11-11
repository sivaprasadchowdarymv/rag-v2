"""Per-request system telemetry: latency, tokens, cost, cache hits, errors.

Kept in memory (bounded) and appended to data/telemetry/requests.jsonl.
No question text is stored, only a short hash.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional

_LOCK = threading.Lock()
_RECENT: Deque[Dict] = deque(maxlen=2000)


def record(data_dir: Path, *, question: str, latency_s: float, tokens: int, llm_calls: int, provider: str,
           cached: bool, error: bool, kind: str, cost_usd: float = 0.0, rate_limited: bool = False,
           model: str = "", saved_tokens: int = 0, saved_cost: float = 0.0, verdict: str = "",
           tenant: str = "", user: str = "") -> Dict:
    """`cost_usd` is the real cost reported by OpenRouter (usage.cost), not an estimate."""
    row = {"ts": round(time.time(), 3), "q": hashlib.sha256(question.encode()).hexdigest()[:10],
           "latency_s": round(latency_s, 3), "tokens": int(tokens), "llm_calls": int(llm_calls),
           "provider": provider or "", "model": model or "", "cached": cached, "error": error, "kind": kind,
           "rate_limited": rate_limited, "cost": round(float(cost_usd or 0.0), 6)}
    if saved_tokens or saved_cost:  # 2.1: answer-cache savings (tokens/cost the cached answer originally used)
        row.update(saved_tokens=int(saved_tokens), saved_cost=round(float(saved_cost), 6))
    if verdict:
        row["verdict"] = verdict
    if tenant:  # hashed: counts active users without storing identities
        row["tenant"] = hashlib.sha256(tenant.encode()).hexdigest()[:10]
    if user:
        row["user"] = hashlib.sha256(user.encode()).hexdigest()[:10]
    with _LOCK:
        _RECENT.append(row)
        try:
            path = Path(data_dir) / "telemetry" / "requests.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
        except OSError:
            pass
    return row


def _pct(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return round(v[lo] + (v[hi] - v[lo]) * (k - lo), 3)


def summary(rows: Optional[List[Dict]] = None) -> Dict:
    rows = list(_RECENT) if rows is None else rows
    served = [r for r in rows if not r.get("rate_limited")]
    lat = [r["latency_s"] for r in served if not r["error"]]
    n = len(served)
    return {"requests": len(rows), "p50_latency_s": _pct(lat, 0.50), "p95_latency_s": _pct(lat, 0.95),
            "avg_latency_s": round(sum(lat) / len(lat), 3) if lat else None,
            "avg_tokens": round(sum(r["tokens"] for r in served) / n, 1) if n else None,
            "total_tokens": sum(r["tokens"] for r in served),
            "total_cost": round(sum(r["cost"] for r in served), 6),
            "cost_per_request": round(sum(r["cost"] for r in served) / n, 6) if n else None,
            "cache_hit_rate": round(sum(r["cached"] for r in served) / n, 3) if n else None,
            "error_rate": round(sum(r["error"] for r in served) / n, 3) if n else None,
            "rate_limited": sum(1 for r in rows if r.get("rate_limited")),
            # 2.1 additions (absent in old rows -> counted as 0)
            "cache_hits": sum(1 for r in served if r.get("cached")),
            "cache_saved_tokens": sum(int(r.get("saved_tokens") or 0) for r in served),
            "cache_saved_cost": round(sum(float(r.get("saved_cost") or 0.0) for r in served), 6),
            "verified_rate": _rate([r for r in served if r.get("verdict")], lambda r: r["verdict"] in
                                   ("verified", "partial")),
            "verdicts": sum(1 for r in served if r.get("verdict")),
            "active_users": len({r["user"] for r in served if r.get("user")}),
            "blocked": sum(1 for r in served if r.get("kind") in ("blocked", "out_of_scope")),
            "abstained": sum(1 for r in served if r.get("kind") == "abstained")}


def _rate(rows: List[Dict], pred) -> Optional[float]:
    return round(sum(1 for r in rows if pred(r)) / len(rows), 3) if rows else None


def _day_start() -> float:
    t = time.localtime()
    return time.mktime((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, 0, 0, -1))


def spent_today(data_dir: Path) -> float:
    """USD spent on answers since local midnight (all workspaces on this server)."""
    start = _day_start()
    path = Path(data_dir) / "telemetry" / "requests.jsonl"
    total = 0.0
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("ts", 0) >= start:
                    total += float(row.get("cost") or 0.0)
    except OSError:
        with _LOCK:
            total = sum(float(r.get("cost") or 0.0) for r in _RECENT if r.get("ts", 0) >= start)
    return round(total, 6)


def reset() -> None:
    with _LOCK:
        _RECENT.clear()
