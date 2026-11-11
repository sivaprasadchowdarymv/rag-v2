"""Performance guards: retrieval latency on a larger synthetic document and concurrent API queries.

Budgets are generous (CI machines vary); they catch regressions such as accidental O(n²) scoring or
re-downloading models per request, not micro-optimisations."""
from __future__ import annotations

import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from agents.orchestrator import plan
from agents.retrieval_stage import run_retrieval


def _big_doc(n_sections: int = 150) -> bytes:
    parts = []
    for i in range(n_sections):
        parts.append(f"# Section {i}\n\nComponent C{i} has a rated voltage of {i % 40 + 5} V and a maximum current of "
                     f"{(i % 9) + 1}.{i % 10} A. Thermal limit is {60 + i % 50} °C. Maintenance interval {100 + i} h.\n")
    return "\n".join(parts).encode()


def test_retrieval_latency_budget(make_settings, server):
    from rag.pipeline import get_index
    s = make_settings(max_chunk_tokens=120)
    t0 = time.perf_counter()
    index, _ = get_index(_big_doc(), "big.md", s)
    ingest_s = time.perf_counter() - t0
    assert len(index.nodes) >= 150 and ingest_s < 30
    lat = []
    for i in range(30):
        t = time.perf_counter()
        out = run_retrieval(plan(f"What is the maximum current of component C{i * 5}?"), [index], s)
        lat.append((time.perf_counter() - t) * 1000)
        assert out.fusion.evidence
    p50, p95 = statistics.median(lat), sorted(lat)[int(0.95 * (len(lat) - 1))]
    print(f"\nretrieval over {len(index.nodes)} nodes: p50={p50:.1f} ms p95={p95:.1f} ms; ingestion {ingest_s:.2f} s")
    assert p95 < 1500, f"retrieval p95 {p95:.0f} ms"


def test_concurrent_api_queries(make_settings, server, sample_pdf):
    from api.main import create_app
    s = make_settings(rate_limit_per_min=0, global_rate_limit_per_min=0, answer_cache_ttl=0)
    c = TestClient(create_app(s))
    assert c.post("/v1/documents", files={"file": ("LM7805X.pdf", sample_pdf)}).status_code == 200
    qs = ["What is the peak output current?", "What is the dropout voltage?", "What is the function of pin 2?",
          "What is the typical quiescent current?"] * 4

    def one(q):
        t = time.perf_counter()
        r = c.post("/v1/query", json={"question": q, "session_id": q[:20]})
        return r.status_code, time.perf_counter() - t

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        res = list(pool.map(one, qs))
    wall = time.perf_counter() - t0
    assert all(code == 200 for code, _ in res)
    p95 = sorted(t for _, t in res)[int(0.95 * (len(res) - 1))]
    print(f"\n{len(qs)} concurrent queries: wall {wall:.2f} s, p95 {p95:.2f} s")
    assert p95 < 20
