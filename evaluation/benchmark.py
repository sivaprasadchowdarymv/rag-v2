"""
Benchmark runner: baseline (1.x) vs upgraded (2.0) on the controlled corpus — retrieval and, when a model is
reachable, generation. Results are stored in the evaluation_runs table and exported as JSON / CSV / Markdown.

    python -m evaluation.benchmark                      # retrieval only (no LLM needed)
    python -m evaluation.benchmark --generation         # + answers with the configured providers
    python -m evaluation.benchmark --out data/evaluations

Configurations
    baseline   1.x retrieval: single query, no rewriting, no parent-child; answers via answer_pro
    upgraded   2.0: rewriting + multi-query RRF + parent-child; answers via RAGService (agents + guardrails)

Nothing is estimated: a metric that could not be computed (e.g. generation without a model, dense retrieval
without an embedding model) is reported as null with the reason in "notes".
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import io
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.orchestrator import plan
from agents.rag_pro import answer_pro
from agents.retrieval_stage import run_retrieval
from config.settings import Settings, get_logger, load_settings
from evaluation import corpus
from evaluation import metrics as M
from llm.model_router import ModelRouter
from rag.pipeline import get_index

log = get_logger("evaluation")
DATASET = Path(__file__).parent / "test_datasets" / "benchmark_v1.json"
CONFIGS = ("baseline", "upgraded")


def config_settings(settings: Settings, name: str) -> Settings:
    if name == "baseline":
        return dataclasses.replace(settings, query_rewrite=False, multi_query=1, parent_child=False,
                                   context_compression=False)
    return dataclasses.replace(settings, query_rewrite=True, multi_query=max(2, settings.multi_query),
                               parent_child=True)


def load_dataset(path: Path = DATASET) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_indexes(settings: Settings) -> Dict[str, Any]:
    out, timings = {}, {}
    for name, data in corpus.files().items():
        t0 = time.perf_counter()
        index, warnings = get_index(data, name, settings)
        timings[name] = round(time.perf_counter() - t0, 3)
        out[name] = index
    return {"indexes": out, "ingestion_s": timings,
            "dense": all(ix.missing_embeddings == 0 for ix in out.values())}


def run(settings: Optional[Settings] = None, generation: bool = False, k: int = 5,
        dataset_path: Path = DATASET, service_factory=None) -> Dict[str, Any]:
    settings = settings or load_settings()
    ds = load_dataset(dataset_path)
    built = build_indexes(settings)
    indexes = list(built["indexes"].values())
    notes: List[str] = []
    if not built["dense"]:
        notes.append("Dense retrieval unavailable (embedding model not loaded): lexical retrieval only.")
    report: Dict[str, Any] = {"dataset": ds["name"], "k": k, "questions": len(ds["questions"]),
                              "ingestion_s": built["ingestion_s"], "configs": {}, "notes": notes,
                              "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    router_ok = ModelRouter(settings).has_role("master")
    if generation and not router_ok:
        notes.append("Generation metrics not run: no LLM provider is configured/reachable.")
        generation = False
    for cfg in CONFIGS:
        s = config_settings(settings, cfg)
        rows = []
        for q in ds["questions"]:
            exp = q["expected_source"]
            t0 = time.perf_counter()
            out = run_retrieval(plan(q["question"]), indexes, s)
            r_ms = (time.perf_counter() - t0) * 1000
            ranked = [(e.doc_name, int(e.page) if isinstance(e.page, int) else 0) for e in out.fusion.evidence]
            row: Dict[str, Any] = {"id": q["id"], "difficulty": q["difficulty"], "retrieval_ms": round(r_ms, 1),
                                   **M.retrieval_metrics(ranked, exp["file"], exp["pages"], k),
                                   "context_relevance": M.context_relevance([e.content for e in out.fusion.evidence],
                                                                            q["relevant_context"], q["expected_answer"])}
            if generation:
                row.update(_generate(q, s, indexes, cfg, service_factory))
            rows.append(row)
        report["configs"][cfg] = {"rows": rows, "summary": _summarise(rows, generation)}
    report["comparison"] = _compare(report["configs"])
    return report


def _generate(q: Dict, s: Settings, indexes, cfg: str, service_factory) -> Dict[str, Any]:
    exp = q["expected_source"]
    t0 = time.perf_counter()
    failed = 0.0
    try:
        if cfg == "baseline":
            res = answer_pro(q["question"], indexes, s, ModelRouter(s))
        else:
            from core.service import AskRequest, RAGService
            svc = service_factory(s) if service_factory else RAGService(s)
            res = svc.ask(AskRequest(q["question"], indexes, session_key=f"eval-{q['id']}"))
    except Exception as exc:  # recorded as a failure, never hidden
        log.warning("generation failed for %s: %s", q["id"], type(exc).__name__)
        return {"answer_correct": 0.0, "failed": 1.0, "latency_s": round(time.perf_counter() - t0, 3)}
    if not res.used_llm and res.kind == "answer":
        failed = 1.0
    refs = {s_.ref_num: s_.full_text for s_ in res.sources}
    cited = [(s_.doc_name, int(s_.page) if isinstance(s_.page, int) else 0) for s_ in res.sources]
    ver = res.verification or {}
    return {"answer_correct": M.answer_correct(res.answer, q["expected_answer"]),
            "faithfulness": M.faithfulness(res.answer, refs),
            "answer_relevance": (res.metrics or {}).get("answer_relevance"),
            "citation_correctness": M.citation_correctness(cited, exp["file"], exp["pages"]),
            "hallucination": M.hallucinated(res.answer, refs, ver.get("invalid_citations") or []),
            "abstained": 1.0 if res.kind in ("abstained", "not_found") else 0.0,
            "latency_s": round(time.perf_counter() - t0, 3), "tokens": res.tokens, "cost_usd": res.cost,
            "failed": failed, "model": f"{res.provider}:{res.model}"}


def _summarise(rows: List[Dict], generation: bool) -> Dict[str, Any]:
    keys = ["precision@k", "recall@k", "hit@k", "mrr", "ndcg@k", "context_relevance"]
    if generation:
        keys += ["answer_correct", "faithfulness", "answer_relevance", "citation_correctness", "hallucination",
                 "abstained", "failed"]
    out = {k: M.mean(r.get(k) for r in rows) for k in keys}
    out["retrieval_ms_p50"] = M.pct([r["retrieval_ms"] for r in rows], 50)
    out["retrieval_ms_p95"] = M.pct([r["retrieval_ms"] for r in rows], 95)
    if generation:
        out["latency_s_p50"] = M.pct([r.get("latency_s") for r in rows], 50)
        out["latency_s_p95"] = M.pct([r.get("latency_s") for r in rows], 95)
        out["tokens_per_query"] = M.mean(r.get("tokens") for r in rows)
        out["cost_per_query_usd"] = M.mean(r.get("cost_usd") for r in rows)
    by_diff: Dict[str, List[Dict]] = {}
    for r in rows:
        by_diff.setdefault(r["difficulty"], []).append(r)
    out["recall@k_by_difficulty"] = {d: M.mean(x["recall@k"] for x in rs) for d, rs in sorted(by_diff.items())}
    return out


def _compare(configs: Dict[str, Dict]) -> Dict[str, Any]:
    if not {"baseline", "upgraded"} <= set(configs):
        return {}
    b, u = configs["baseline"]["summary"], configs["upgraded"]["summary"]
    return {k: {"baseline": b[k], "upgraded": u[k], "delta": round(u[k] - b[k], 4)}
            for k in b if isinstance(b.get(k), (int, float)) and isinstance(u.get(k), (int, float))}


def to_markdown(rep: Dict[str, Any]) -> str:
    lines = [f"# Evaluation report — {rep['dataset']}", "", f"Created {rep['created']} · {rep['questions']} questions · "
             f"k = {rep['k']}", ""]
    for n in rep["notes"]:
        lines.append(f"> **Note:** {n}")
    lines += ["", "## Baseline vs upgraded", "", "| Metric | Baseline | Upgraded | Δ |", "|---|---|---|---|"]
    for k, v in rep["comparison"].items():
        lines.append(f"| {k} | {v['baseline']} | {v['upgraded']} | {v['delta']:+} |")
    lines += ["", "## Ingestion time (s)", "", "| File | Seconds |", "|---|---|"]
    lines += [f"| {f} | {t} |" for f, t in rep["ingestion_s"].items()]
    for cfg, data in rep["configs"].items():
        lines += ["", f"## Per question — {cfg}", ""]
        cols = [c for c in data["rows"][0] if c not in ("model",)]
        lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in data["rows"]:
            lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def to_csv(rep: Dict[str, Any]) -> str:
    buf = io.StringIO()
    rows = [{"config": cfg, **r} for cfg, d in rep["configs"].items() for r in d["rows"]]
    cols = sorted({k for r in rows for k in r}, key=lambda c: (c != "config", c != "id", c))
    w = csv.DictWriter(buf, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def save(rep: Dict[str, Any], settings: Settings, out_dir: Optional[Path] = None, name: str = "benchmark",
         tenant_id: str = "default") -> Dict[str, str]:
    out_dir = Path(out_dir or Path(settings.data_dir) / "evaluations")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    paths = {}
    for ext, text in (("json", json.dumps(rep, indent=2, default=str)), ("md", to_markdown(rep)), ("csv", to_csv(rep))):
        p = out_dir / f"{name}-{stamp}.{ext}"
        p.write_text(text, encoding="utf-8")
        paths[ext] = str(p)
    try:
        from database.repository import get_repository
        get_repository(settings).save_evaluation(tenant_id, name, {"k": rep["k"], "dataset": rep["dataset"]},
                                                 {"comparison": rep["comparison"], "notes": rep["notes"],
                                                  "summary": {c: d["summary"] for c, d in rep["configs"].items()}},
                                                 to_markdown(rep))
    except Exception:
        log.exception("could not store the evaluation run")
    return paths


def main() -> None:  # pragma: no cover - CLI
    ap = argparse.ArgumentParser(description="Baseline vs upgraded benchmark")
    ap.add_argument("--generation", action="store_true", help="also generate answers (needs a model)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    s = load_settings()
    rep = run(s, generation=args.generation, k=args.k)
    paths = save(rep, s, Path(args.out) if args.out else None)
    print(to_markdown(rep)[:3000])
    print("saved:", paths)


if __name__ == "__main__":  # pragma: no cover
    main()
