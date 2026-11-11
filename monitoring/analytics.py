"""Usage analytics from telemetry rows (no question text): per-model cost/tokens, per-day volume, failure rate."""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def load_rows(data_dir: Path, days: int = 30) -> List[Dict]:
    path = Path(data_dir) / "telemetry" / "requests.jsonl"
    if not path.exists():
        return []
    cutoff = time.time() - days * 86400
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("ts", 0) >= cutoff:
                rows.append(r)
    return rows


def by_model(rows: List[Dict]) -> List[Dict]:
    agg: Dict[str, Dict] = defaultdict(lambda: {"requests": 0, "tokens": 0, "cost": 0.0, "errors": 0, "latency": 0.0})
    for r in rows:
        if r.get("rate_limited"):
            continue
        key = f"{r.get('provider') or 'none'}:{r.get('model') or '-'}"
        a = agg[key]
        a["requests"] += 1
        a["tokens"] += r.get("tokens", 0)
        a["cost"] += r.get("cost", 0.0)
        a["errors"] += 1 if r.get("error") else 0
        a["latency"] += r.get("latency_s", 0.0)
    return [{"model": k, "requests": v["requests"], "tokens": v["tokens"], "cost_usd": round(v["cost"], 6),
             "error_rate": round(v["errors"] / v["requests"], 3) if v["requests"] else 0.0,
             "avg_latency_s": round(v["latency"] / v["requests"], 3) if v["requests"] else 0.0}
            for k, v in sorted(agg.items(), key=lambda kv: -kv[1]["requests"])]


def by_day(rows: List[Dict]) -> List[Dict]:
    agg: Dict[str, Dict] = defaultdict(lambda: {"requests": 0, "cost": 0.0, "tokens": 0})
    for r in rows:
        day = time.strftime("%Y-%m-%d", time.localtime(r.get("ts", 0)))
        agg[day]["requests"] += 1
        agg[day]["cost"] += r.get("cost", 0.0)
        agg[day]["tokens"] += r.get("tokens", 0)
    return [{"day": d, **v} for d, v in sorted(agg.items())]
