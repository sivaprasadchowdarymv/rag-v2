"""
Run the planned specialist agents over the selected documents, then rerank
and fuse their evidence. Every stage is timed; every agent's calls,
results, failures and latency are recorded for the metrics page.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from agents.orchestrator import Plan
from agents.specialists import SPECIALISTS, QueryContext
from concepts.expand import expand
from concepts.graph import get_graph
from config.settings import Settings, get_logger
from rag.models import DocumentIndex, Evidence
from retrieval.evidence_fusion import FusionResult, fuse, merge, rrf
from retrieval import query_rewriter
from retrieval.hybrid_search import MetadataFilter, compress, parent_text
from retrieval.reranker import rerank

log = get_logger("retrieval_stage")
RERANK_POOL = 20


@dataclass
class RetrievalOutcome:
    fusion: FusionResult
    agent_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    latency_ms: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    reranked: bool = False
    debug: Dict = field(default_factory=dict)


def run_agents(agents: Sequence[str], query: str, indexes: Sequence[DocumentIndex], settings: Settings,
               describe: Optional[Callable] = None, ctx: Optional[QueryContext] = None):
    ctx = ctx or QueryContext(query, settings)
    groups: List[List[Evidence]] = []
    stats: Dict[str, Dict[str, float]] = {}
    for name in agents:
        fn = SPECIALISTS[name]
        st = stats.setdefault(name, {"calls": 0, "results": 0, "failures": 0, "latency_ms": 0.0})
        for index in indexes:
            t0 = time.perf_counter()
            st["calls"] += 1
            try:
                found = fn(index, ctx, describe) if name == "figure_agent" else fn(index, ctx)
                groups.append(found)
                st["results"] += len(found)
            except Exception:  # one failing agent must never break the answer
                log.exception("%s failed on %s", name, index.filename)
                st["failures"] += 1
            st["latency_ms"] += (time.perf_counter() - t0) * 1000
    return groups, stats, ctx


def _filtered(groups: List[List[Evidence]], flt: Optional[MetadataFilter]) -> List[List[Evidence]]:
    if flt is None:
        return groups
    out = []
    for g in groups:
        keep = []
        for e in g:
            if flt.doc_ids is not None and e.doc_id not in flt.doc_ids:
                continue
            if flt.types is not None and e.type not in flt.types and not (e.type == "pin" and "row" in flt.types):
                continue
            if flt.pages is not None and isinstance(e.page, int) and not flt.pages[0] <= e.page <= flt.pages[1]:
                continue
            if flt.section_contains and flt.section_contains.lower() not in (e.section or "").lower():
                continue
            if flt.ocr_only and not (e.meta or {}).get("ocr"):
                continue
            keep.append(e)
        out.append(keep)
    return out


def run_retrieval(plan: Plan, indexes: Sequence[DocumentIndex], settings: Settings,
                  describe: Optional[Callable] = None, chooser: Optional[Callable] = None,
                  filters: Optional[MetadataFilter] = None, rewriter: Optional[Callable] = None) -> RetrievalOutcome:
    t0 = time.perf_counter()
    if filters is not None and filters.doc_ids is not None:  # pre-filter whole documents
        indexes = [ix for ix in indexes if ix.doc_id in filters.doc_ids]
    groups, stats, ctx = run_agents(plan.agents, plan.retrieval_query, indexes, settings, describe)
    queries = [plan.retrieval_query]
    if settings.query_rewrite and settings.multi_query > 1:
        queries = query_rewriter.variants(plan.retrieval_query, settings.multi_query,
                                          rewriter if settings.llm_query_rewrite else None)
        for qi, q in enumerate(queries[1:], start=1):  # each variant votes in RRF under its own name
            extra, _, _ = run_agents(plan.agents, q, indexes, settings, describe)
            for g in extra:
                for e in g:
                    e.ranks = {f"{k}@q{qi}": v for k, v in e.ranks.items()}
            groups += extra
    groups = _filtered(groups, filters)
    t_retrieval = (time.perf_counter() - t0) * 1000

    # Concept/entity resolution -> selective cross-modal expansion (deterministic, cached graph).
    t_concepts = 0.0
    if getattr(settings, "concept_linking", False) and indexes:
        try:
            if chooser is not None:
                for ix in indexes:
                    get_graph(ix, chooser)
            existing = {e.key for g in groups for e in g}
            linked, cstats = expand(plan.retrieval_query, plan.intents, indexes, existing,
                                    settings.concept_expand_max, settings.concept_min_conf,
                                    settings.max_chars_per_chunk)
            t_concepts = cstats["latency_ms"]
            if linked:
                groups.append(linked)
            stats["concept_linker"] = {"calls": 1, "results": cstats["added"], "failures": 0,
                                       "latency_ms": t_concepts, "concepts": cstats["concepts"]}
        except Exception:  # linking is an enhancement: never break retrieval
            log.exception("concept linking failed")
            stats["concept_linker"] = {"calls": 1, "results": 0, "failures": 1, "latency_ms": 0.0}

    # Rerank the best candidates by RRF with the cross-encoder.
    pool = merge(groups)
    rrf(pool)
    pool.sort(key=lambda e: e.score, reverse=True)
    pool = pool[:RERANK_POOL]
    scores, t_rerank, warn = rerank(plan.retrieval_query, [e.content for e in pool], settings)
    rerank_scores = {e.key: s for e, s in zip(pool, scores)} if scores else None

    t1 = time.perf_counter()
    budget_items = settings.max_context_chunks + (2 if len(indexes) > 1 else 0)
    fused = fuse(groups, budget_items, settings.max_context_chars, rerank_scores)
    if "figure" in plan.intents and not any(e.type == "figure" for e in fused.evidence):
        # A figure question must keep its best figure: text rerankers score short captions low,
        # and the figure (and its image) is exactly what the user asked about.
        best = next((e for g in groups for e in g if e.type == "figure"), None)
        if best is not None:
            if len(fused.evidence) >= budget_items:
                fused.evidence.pop()
            best.confidence = 0.5
            fused.evidence.append(best)
    by_doc = {ix.doc_id: ix for ix in indexes}
    if settings.parent_child or settings.context_compression:
        used = sum(len(e.content) for e in fused.evidence)
        for e in fused.evidence[:2]:  # small-to-big for the strongest text evidence only (token budget)
            ix = by_doc.get(e.doc_id)
            if settings.parent_child and e.type == "text" and ix is not None and 0 <= e.node_index < len(ix.nodes):
                parent = parent_text(ix, ix.nodes[e.node_index], settings.max_chars_per_chunk * 2)
                if len(parent) > len(e.content) and used + len(parent) - len(e.content) <= settings.max_context_chars:
                    used += len(parent) - len(e.content)
                    e.content, e.meta = parent, {**(e.meta or {}), "parent_expanded": True}
        if settings.context_compression:
            for e in fused.evidence:
                e.content = compress(e.content, plan.retrieval_query, settings.max_chars_per_chunk)
    t_fusion = (time.perf_counter() - t1) * 1000

    if "concept_linker" in stats:  # what actually reached the LLM context because of linking
        kept = [e for e in fused.evidence if set(e.ranks) == {"concept"}]
        stats["concept_linker"].update(kept=len(kept), extra_tokens=round(sum(len(e.content) for e in kept) / 4))
    warnings = list(dict.fromkeys(ctx.warnings + ([warn] if warn else [])))
    lat = {"retrieval": t_retrieval, "rerank": t_rerank, "fusion": t_fusion}
    if t_concepts:
        lat["concepts"] = t_concepts
    selected = {e.key for e in fused.evidence}
    debug = {"original_query": plan.retrieval_query, "queries": queries,
             "filters": None if filters is None else {k: (sorted(v) if isinstance(v, set) else v)
                                                     for k, v in vars(filters).items() if v},
             "candidates": [{"key": e.key, "doc": e.doc_name, "page": e.page, "type": e.type, "agent": e.agent,
                             "ranks": dict(e.ranks), "rrf": round(e.score, 5),
                             "rerank": None if rerank_scores is None else rerank_scores.get(e.key),
                             "selected": e.key in selected, "text": e.excerpt[:200]} for e in pool],
             "selected": [{"key": e.key, "doc": e.doc_name, "page": e.page, "type": e.type,
                           "confidence": e.confidence, "chars": len(e.content),
                           "parent_expanded": bool((e.meta or {}).get("parent_expanded"))} for e in fused.evidence]}
    return RetrievalOutcome(fused, stats, lat, warnings, reranked=bool(scores), debug=debug)
