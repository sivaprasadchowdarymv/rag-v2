"""
Controlled multi-agent workflow (deterministic routing, bounded, traced).

    User query
      → QueryUnderstandingAgent    task type, intents, follow-up rewrite, search variants          (rules, 0 tokens)
      → orchestrator decision      no documents · summarisation · question answering
      → DocumentRetrievalAgent     specialists → concept links → rerank → fusion                 (local, 0 tokens)
        or SummarizationAgent      section-level evidence across the whole document                (local, 0 tokens)
      → ReasoningAgent             cited answer, ReAct tools only for calculations/comparisons   (1 LLM call typ.)
      → CitationVerificationAgent  numbers vs cited sources, mis-citations, claim support; ≤1 regeneration
      → AnswerQualityAgent         relevance, grounding, citations → accept | warn | abstain        (0 tokens)
      → final response

Bounds: every agent runs with a timeout (AGENT_TIMEOUT_S); the ReAct loop is limited by AGENT_MAX_STEPS,
regeneration by MAX_REGENERATIONS; the workflow itself is a fixed sequence (no agent can call another,
so loops are impossible). Each agent gets only the context it needs. A failing agent has a fallback:
understanding → default plan; summarisation → normal retrieval; reasoning → evidence-only answer;
quality → accept with a warning. The trace (agent, status, latency, short summary — never document text)
is returned in QueryResult.agent_trace.

Retrieval, reasoning and verification are the existing agents (agents/retrieval_stage.py, agents/master.py,
agents/verifier.py) coordinated by agents/rag_pro.answer_pro; their trace entries are built from the
measured stage latencies and reports of that run, not estimated.
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from agents.orchestrator import Plan, plan as make_plan
from agents.rag_pro import answer_pro
from agents.retrieval_stage import RetrievalOutcome
from config.settings import Settings, get_logger
from rag.models import AgentStep, DocumentIndex, Evidence, QueryResult
from rag.prompts import NOT_FOUND
from retrieval import query_rewriter
from retrieval.evidence_fusion import FusionResult
from retrieval.hybrid_search import MetadataFilter

log = get_logger("workflow")

_SUMMARY = re.compile(r"\b(summar\w*|overview|key (?:points|features|takeaways)|what is (?:this|the) document about|"
                      r"tl;?dr|outline)\b", re.I)
_GREETING = re.compile(r"^\s*(hi|hello|hey|thanks?|thank you|good (?:morning|evening))\W*$", re.I)
_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="agent")


@dataclass
class Understanding:
    task: str                    # question | summarization | calculation | comparison | figure | greeting
    plan: Plan
    variants: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return f"task={self.task}; intents={','.join(self.plan.intents)}; variants={len(self.variants)}"


class AgentTimeout(Exception):
    pass


def _timed(fn: Callable[[], Any], timeout: float) -> Any:
    import contextvars

    ctx = contextvars.copy_context()  # keep request id, log context and trace spans inside the agent thread
    fut = _POOL.submit(ctx.run, fn)
    try:
        return fut.result(timeout=timeout)
    except FutureTimeout as exc:  # the thread cannot be killed; its result is discarded
        fut.cancel()
        raise AgentTimeout(f"timed out after {timeout:.0f}s") from exc


class QueryUnderstandingAgent:
    name = "Query Understanding Agent"

    def run(self, question: str, previous: Optional[str], n_docs: int, settings: Settings) -> Understanding:
        p = make_plan(question, previous, n_docs, settings.agent_mode)
        if _GREETING.match(question):
            task = "greeting"
        elif _SUMMARY.search(question) and not p.is_follow_up:
            task = "summarization"
        elif "calculation" in p.intents:
            task = "calculation"
        elif "comparison" in p.intents or "multi_document" in p.intents:
            task = "comparison"
        elif "figure" in p.intents:
            task = "figure"
        else:
            task = "question"
        n = settings.multi_query if settings.query_rewrite else 1
        return Understanding(task, p, query_rewriter.variants(p.retrieval_query, n))


class SummarizationAgent:
    """Whole-document evidence: document card + the first chunk of each section (bounded)."""

    name = "Summarization Agent"

    def run(self, indexes: Sequence[DocumentIndex], settings: Settings) -> RetrievalOutcome:
        from agents.specialists import QueryContext, document_agent

        groups: List[List[Evidence]] = []
        per_doc = max(1, settings.summarize_max_sections // max(len(indexes), 1))
        budget = settings.max_context_chars
        for ix in indexes:
            ev = document_agent(ix, QueryContext("summary", settings))
            seen, picked = set(), []
            for i, n in enumerate(ix.nodes):
                if n.type != "text" or n.section in seen or len(picked) >= per_doc:
                    continue
                seen.add(n.section)
                picked.append(Evidence(key=f"{ix.doc_id}#n{i}", doc_id=ix.doc_id, doc_name=ix.filename, type="text",
                                       page=n.page, section=n.section, content=n.content[: settings.max_chars_per_chunk],
                                       excerpt=n.raw_content, agent="summarization_agent", node_index=i,
                                       parent_ctx=n.parent_ctx, ranks={"section": len(picked) + 1}))
            groups.append(ev + picked)
        items: List[Evidence] = []
        used = 0
        for g in groups:  # interleave documents fairly within the budget
            for e in g:
                if used + len(e.content) > budget and items:
                    break
                items.append(e)
                used += len(e.content)
        for k, e in enumerate(items):
            e.confidence = round(max(0.3, 1.0 - 0.05 * k), 2)
        stats = {"summarization_agent": {"calls": len(indexes), "results": len(items), "failures": 0,
                                         "latency_ms": 0.0}}
        return RetrievalOutcome(FusionResult(items, [], 0, len(items)), stats, {"retrieval": 0.0}, [],
                                debug={"original_query": "document summary", "queries": ["document summary"],
                                       "candidates": [], "selected": [{"key": e.key, "doc": e.doc_name,
                                                                        "page": e.page, "type": e.type}
                                                                       for e in items]})


class AnswerQualityAgent:
    """Deterministic final check: grounded + cited + relevant → accept; otherwise warn or abstain."""

    name = "Answer Quality Agent"
    ABSTAIN = ("I could not find enough reliable evidence in your documents to answer this. "
               "Try rephrasing the question or check the sources below.")

    def run(self, result: QueryResult, settings: Settings) -> Dict[str, Any]:
        if result.kind != "answer" or not result.used_llm:
            return {"action": "accept", "score": None, "issues": []}
        ver = result.verification or {}
        m = result.metrics or {}
        issues = []
        if ver.get("verdict") not in ("verified", "partial"):
            issues.append("not grounded")
        if ver.get("invalid_citations"):
            issues.append("invalid citations")
        if not result.sources:
            issues.append("no sources")
        relevance = m.get("answer_relevance")
        if relevance is not None and relevance < 0.15:
            issues.append("low relevance")
        score = round(1.0 - 0.25 * len(issues), 2)
        grounded = ver.get("grounded", True)
        action = "accept" if not issues else ("abstain" if settings.abstain_ungrounded and not grounded
                                              and ver.get("unsupported_claims") else "warn")
        return {"action": action, "score": score, "issues": issues}


class AgentWorkflow:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.understanding = QueryUnderstandingAgent()
        self.summarizer = SummarizationAgent()
        self.quality = AnswerQualityAgent()

    def run(self, question: str, indexes: Sequence[DocumentIndex], router, req) -> QueryResult:
        s = self.settings
        trace: List[Dict[str, Any]] = []

        def record(agent: str, status: str, t0: float, summary: str = "") -> None:
            trace.append({"agent": agent, "status": status, "ms": round((time.perf_counter() - t0) * 1000, 1),
                          "summary": summary[:200]})

        # 1. Query understanding (fallback: default task)
        t0 = time.perf_counter()
        try:
            u = _timed(lambda: self.understanding.run(question, req.previous_question, len(indexes), s), 10.0)
            record(self.understanding.name, "ok", t0, u.summary())
        except Exception as exc:
            log.warning("understanding failed: %s", type(exc).__name__)
            u = Understanding("question", make_plan(question))
            record(self.understanding.name, "failed", t0, "fallback: default plan")

        if u.task == "greeting":
            res = QueryResult(kind="answer", answer="Hello! Ask me anything about your documents.", question=question)
            res.agent_trace = trace
            return res
        if not indexes:
            res = QueryResult(kind="not_found", question=question,
                              answer="There are no documents you can access yet. Upload a document first.")
            res.agent_trace = trace + [{"agent": "Orchestrator", "status": "skipped", "ms": 0.0,
                                        "summary": "no authorised documents"}]
            return res

        # 2. Summarisation agent supplies document-wide evidence (fallback: normal retrieval)
        override = None
        if u.task == "summarization":
            t0 = time.perf_counter()
            try:
                override = _timed(lambda: self.summarizer.run(indexes, s), 30.0)
                record(self.summarizer.name, "ok", t0, f"{len(override.fusion.evidence)} section evidence items")
            except Exception as exc:
                record(self.summarizer.name, "failed", t0, f"fallback to retrieval ({type(exc).__name__})")

        # 3-5. Retrieval → reasoning → citation verification (bounded inside answer_pro)
        flt = MetadataFilter.from_dict(req.metadata_filter) if req.metadata_filter else None
        t0 = time.perf_counter()
        try:
            result = _timed(lambda: answer_pro(question, indexes, s, router, memory=req.memory,
                                               previous_question=req.previous_question, preferences=req.preferences,
                                               on_step=req.on_step, filters=flt, retrieval_override=override,
                                               cache_scope=str((getattr(req, "extra", None) or {}).get(
                                                   "cache_scope", ""))),
                            s.agent_timeout_s)
        except AgentTimeout:
            record("Reasoning Agent", "timeout", t0, f"exceeded {s.agent_timeout_s:.0f}s")
            result = QueryResult(kind="not_found", question=question, answer=NOT_FOUND,
                                 warnings=["The answer took too long and was stopped. Please try again or ask a "
                                           "narrower question."])
            result.agent_trace = trace
            return result
        lat = result.stage_latency or {}
        if result.cached:
            ci = result.cache_info or {}
            trace.append({"agent": "Answer Cache", "status": "hit", "ms": round((time.perf_counter() - t0) * 1000, 1),
                          "summary": f"{ci.get('match', 'exact')} match, verified {int(ci.get('age_s', 0))} s ago; "
                                     f"saved ~{ci.get('saved_tokens', 0)} tokens"})
        if override is None:
            trace.append({"agent": "Document Retrieval Agent", "status": "ok",
                          "ms": round(sum(lat.get(k, 0.0) for k in ("retrieval", "concepts", "rerank", "fusion")), 1),
                          "summary": f"{len(result.sources)} cited / "
                                     f"{len((result.retrieval_debug or {}).get('selected', []))} evidence items"})
        trace.append({"agent": "Reasoning Agent", "status": "ok" if result.used_llm else "fallback",
                      "ms": round(lat.get("master", 0.0), 1),
                      "summary": f"{result.provider}:{result.model} calls={result.llm_calls}" if result.used_llm
                      else "no model answered: evidence-only response"})
        if result.verification:
            trace.append({"agent": "Citation Verification Agent", "status": "ok",
                          "ms": round(lat.get("verifier", 0.0), 1),
                          "summary": f"verdict={result.verification.get('verdict')}"
                                     + ("; regenerated once" if result.regenerated else "")})

        # 6. Answer quality (fallback: accept with warning)
        t0 = time.perf_counter()
        try:
            q = self.quality.run(result, s)
            record(self.quality.name, "ok", t0, f"action={q['action']} issues={','.join(q['issues']) or '-'}")
        except Exception:
            q = {"action": "accept", "issues": ["quality check failed"], "score": None}
            record(self.quality.name, "failed", t0, "accepted without quality check")
        if q["action"] == "abstain":
            result.warnings = list(dict.fromkeys(["The draft answer contained claims not supported by the "
                                                  "documents, so it was withheld."] + result.warnings))
            result.answer, result.kind = AnswerQualityAgent.ABSTAIN, "abstained"
        elif q["action"] == "warn" and q["issues"]:
            result.warnings = list(dict.fromkeys(result.warnings + [f"Quality check: {', '.join(q['issues'])}."]))
        ci = result.cache_info if isinstance(result.cache_info, dict) else {}
        stored_key = ci.pop("_key", None)
        if stored_key and (q["action"] != "accept" or q.get("issues")):  # never reuse an answer quality rejected
            from ops import answer_cache

            answer_cache.for_settings(s).discard(stored_key)
            answer_cache.for_settings(s).note_not_cached(f"quality check: {q['action']}")
            result.cache_info = {"status": "skipped", "reason": f"quality check: {q['action']}"}
        result.metrics = {**(result.metrics or {}), "quality_score": q.get("score")}
        result.plan = {**(result.plan or {}), "task": u.task, "variants": u.variants}
        result.agent_trace = trace
        if req.on_step:
            try:
                req.on_step(AgentStep("stage", f"Answer quality: {q['action']}"))
            except Exception:
                pass
        return result
