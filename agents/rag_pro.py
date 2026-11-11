"""
RAG Ultimate pipeline: one question -> verified, cited answer.

    orchestrator (plan, ~1 ms, no LLM) -> specialists (text/table/equation/figure/document)
    -> reranker -> evidence fusion
    -> MASTER: text questions -> Groq (gpt-oss-120b, ReAct tools when needed)
               figure questions with figure evidence -> Qwen 3.8 / Gemma 4 with the images attached
    -> VERIFIER (deterministic claim checks, optional LLM cross-check)
    (-> one bounded regeneration if verification fails) -> metrics + cost

Every stage is timed. The trace records stages and tool ACTIONS only.
"""
from __future__ import annotations

import re
import time
from typing import Callable, Dict, List, Optional, Sequence

from agents.master import EvidenceRegistry, ProToolBox, regenerate, run_master
from agents.orchestrator import plan as make_plan
from agents.retrieval_stage import run_retrieval
from agents.verifier import canonicalize, check, feedback_text, llm_check
from config.settings import Settings, get_logger
from llm.model_router import ModelRouter
from llm.providers import LLMError
from metrics import rag_metrics
from ops import answer_cache, telemetry
from rag.prompts import NOT_FOUND
from rag.models import AgentStep, DocumentIndex, QueryResult
from rag.pipeline import normalize_answer
from retrieval.reranker import support_scores
from storage.cache import DocumentPaths, VisionCache

log = get_logger("rag_pro")
StepFn = Optional[Callable[[AgentStep], None]]
_AGENT_NAMES = {"text_agent": "Text retrieval", "table_agent": "Table retrieval", "equation_agent": "Equation retrieval",
                "figure_agent": "Figure retrieval", "document_agent": "Document metadata"}


def _vision_describer(router: ModelRouter, settings: Settings):
    if not router.has_role("vision"):
        return None

    def describe(fig, path) -> str:
        cache = VisionCache(DocumentPaths(settings.data_dir, path.parent.parent.name).vision_file)
        cached = cache.get(fig.figure_file or "")
        if cached:
            return cached
        prompt = ("Describe this figure from a technical document precisely: axes, units, curves, blocks and "
                  "connections, key values. No speculation.")
        try:
            text = router.chat("vision", [{"role": "user", "content": prompt}], images=[path.read_bytes()],
                               max_tokens=500).result.content
        except LLMError as exc:
            log.info("Vision skipped: %s", exc.kind)
            return ""
        if text:
            cache.set(fig.figure_file or "", text)
        return text

    return describe


def _concept_chooser(router: ModelRouter, settings: Settings):
    """Small-LLM resolver for ambiguous abbreviations only (off by default; results cached in the graph)."""
    if not settings.concept_llm_resolve or not router.has_role("fast"):
        return None

    def choose(context: str, senses: List[str]):
        prompt = ("Which meaning does the abbreviation have in this text? Reply with the number only.\n"
                  + "\n".join(f"{k + 1}. {s}" for k, s in enumerate(senses)) + f"\n\nText: {context}")
        try:
            res = router.chat("fast", [{"role": "user", "content": prompt}], max_tokens=5)
            m = re.search(r"\d+", res.result.content or "")
            return int(m.group()) - 1 if m else None
        except LLMError:
            return None
    return choose


def _figure_images(evidence, indexes: Sequence[DocumentIndex], settings: Settings):
    """[(figure evidence, PNG bytes)] for the top figures, downscaled to IMAGE_MAX_PX (fewer image tokens)."""
    import pymupdf

    by_doc = {i.doc_id: i for i in indexes}
    out = []
    for ev in evidence:
        if ev.type != "figure" or not ev.figure_file or len(out) >= settings.max_answer_images:
            continue
        index = by_doc.get(ev.doc_id)
        path = index.figures_dir / ev.figure_file if index else None
        if not path or not path.exists():
            continue
        data = path.read_bytes()
        try:
            pix = pymupdf.Pixmap(data)
            if pix.n - pix.alpha >= 4:  # CMYK -> RGB
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
            while max(pix.width, pix.height) > settings.image_max_px:
                pix.shrink(1)
            data = pix.tobytes("png")
        except Exception:  # unusual image format: send the original bytes
            pass
        out.append((ev, data))
    return out


def answer_pro(question: str, indexes: Sequence[DocumentIndex], settings: Settings, router: ModelRouter,
               memory: str = "", previous_question: Optional[str] = None, preferences: str = "",
               on_step: StepFn = None, filters=None, retrieval_override=None, cache_scope: str = "") -> QueryResult:
    """`retrieval_override` (a RetrievalOutcome) lets another agent (e.g. summarisation) supply the evidence;
    reasoning, citation verification and regeneration then run exactly as for normal questions.
    `cache_scope` ("<tenant>|<user/roles/workspace/policy…>", built by RAGService) isolates cached answers per
    security boundary; "" (direct calls, tests) keeps the 1.x behaviour of one process-wide scope."""
    t_start = time.perf_counter()
    question = (question or "").strip()
    steps: List[AgentStep] = []
    warnings: List[str] = []
    latency: Dict[str, float] = {}

    def stage(text: str) -> None:
        step = AgentStep("stage", text)
        steps.append(step)
        if on_step:
            try:
                on_step(step)
            except Exception:
                pass

    # 1. Plan
    plan = make_plan(question, previous_question, len(indexes), settings.agent_mode)
    latency["router"] = plan.latency_ms
    stage(f"Query classified: {', '.join(plan.intents)}" + (" (follow-up)" if plan.is_follow_up else ""))

    cache = answer_cache.for_settings(settings)
    doc_ids = [i.doc_id for i in indexes]
    ckey = None  # cache context (security scope + document versions + every setting that shapes the answer)
    skip_reason = ("follow-up question (depends on the conversation)" if plan.is_follow_up
                   else "document-wide evidence (summary)" if retrieval_override is not None
                   else "" if cache.active else "cache disabled")
    if not skip_reason:
        ckey = answer_cache.answer_key(  # retrieval settings are part of the key
            "", doc_ids, [getattr(i, "embed_model", "") for i in indexes],
            settings.agent_mode, settings.rerank, settings.max_context_chars, settings.models_for("master"),
            settings.verify_with_llm, preferences, settings.multi_query, settings.parent_child,
            settings.context_compression, repr(sorted(vars(filters).items())) if filters is not None else "",
            cache_scope, settings.models_for("verifier"), settings.top_k, settings.max_context_chunks,
            settings.max_chars_per_chunk, settings.llm_temperature, settings.max_output_tokens,
            settings.reasoning_effort, settings.max_regenerations, settings.agent_max_steps, settings.image_answers,
            settings.query_rewrite, settings.llm_query_rewrite, settings.concept_linking, settings.support_threshold,
            settings.abstain_ungrounded, settings.llm_mode, settings.manual_model)
        hit, match, info = cache.lookup(ckey, question, allowed_doc_ids=doc_ids)
        if hit is not None:
            note = ("Answered from cache (same question, verified earlier)" if match == "exact"
                    else "Answered from cache (equivalent question, verified earlier)")
            hit.cached, hit.steps = True, [AgentStep("stage", note)]
            hit.latency, hit.llm_calls, hit.tokens, hit.cost = round(time.perf_counter() - t_start, 3), 0, 0, 0.0
            hit.question = question
            hit.cache_info = {"status": "hit", **info}
            if on_step:
                try:  # a display callback must never break answering (same rule as stage())
                    on_step(hit.steps[0])
                except Exception:
                    pass
            return hit

    # 2-4. Specialists → rerank → fusion
    # With image answers on, figures are SEEN by the vision model at answer time, so no separate
    # per-figure description call is needed during retrieval (saves one vision call per figure question).
    describer = None if settings.image_answers else _vision_describer(router, settings)
    from retrieval.query_rewriter import llm_rewriter

    from monitoring.tracing import span

    with span("retrieval", agents=len(plan.agents)):
        retrieval = retrieval_override or run_retrieval(plan, indexes, settings, describer,
                                                        _concept_chooser(router, settings), filters,
                                                        llm_rewriter(router) if settings.llm_query_rewrite else None)
    if len(retrieval.debug.get("queries", [])) > 1:
        stage(f"Query rewritten: {len(retrieval.debug['queries'])} search variants")
    latency.update(retrieval.latency_ms)
    warnings += retrieval.warnings
    for name in plan.agents:
        st = retrieval.agent_stats.get(name, {})
        stage(f"{_AGENT_NAMES[name]}: {int(st.get('results', 0))} candidates"
              + (f" ({int(st['failures'])} failed)" if st.get("failures") else ""))
    cl = retrieval.agent_stats.get("concept_linker")
    if cl and cl.get("concepts"):
        stage(f"Concept linker: {int(cl['concepts'])} concept(s) resolved, {int(cl['results'])} linked item(s) added")
    if retrieval.reranked:
        stage("Evidence reranked")
    fused = retrieval.fusion
    stage(f"Evidence fused: {len(fused.evidence)} items kept, {fused.dropped_duplicates} duplicates removed")

    base = dict(question=question, mode=settings.agent_mode, plan=plan.as_dict(), agent_stats=retrieval.agent_stats,
                conflicts=fused.conflicts)
    use_tools = plan.use_tools or (settings.agent_mode == "agent" and not fused.evidence)
    if not fused.evidence and not use_tools:
        latency["total"] = (time.perf_counter() - t_start) * 1000
        return QueryResult(kind="not_found", answer=NOT_FOUND, steps=steps, warnings=warnings,
                           stage_latency=latency, latency=round(latency["total"] / 1000, 2),
                           retrieval_debug=retrieval.debug, **base)

    registry = EvidenceRegistry()
    evidence_text = registry.render(fused.evidence, settings.max_context_chars)
    toolbox = ProToolBox(indexes, settings, registry)

    # Route: figure question with figure evidence -> Qwen/Gemma sees the images; everything else -> Groq text.
    images, image_note, visual_refs = [], "", set()
    if settings.image_answers and "figure" in plan.intents and router.has_role("vision"):
        picked = _figure_images(fused.evidence, indexes, settings)
        if picked:
            images = [b for _, b in picked]
            labels = [registry.add(ev) for ev, _ in picked]
            visual_refs = {ref.ref_num for ref in labels}
            image_note = ", ".join(f"{ev.item_id} {ref.label}" for (ev, _), ref in zip(picked, labels))
            stage(f"Figure question: {len(images)} image(s) sent to the vision model")

    if hasattr(router, "complexity_hint"):  # cost-aware routing: simple lookups may use a cheaper/faster model
        router.complexity_hint = "high" if (use_tools or len(indexes) > 1 or images) else "low"

    # 5. Master synthesis
    answer, calls, tokens, provider, model, regenerated = "", 0, 0, "", "", False
    outcome = None
    over_budget = (settings.daily_budget_usd > 0
                   and telemetry.spent_today(settings.data_dir) >= settings.daily_budget_usd)
    if over_budget:
        warnings.append(f"Today's AI budget (${settings.daily_budget_usd:.2f}, DAILY_BUDGET_USD) is used up. "
                        "Showing the best matching evidence without an AI answer.")
    else:
        try:
            try:
                outcome = run_master(question, evidence_text, use_tools, router, settings, toolbox, memory,
                                     preferences, fused.conflicts,
                                     on_step=lambda s: (steps.append(s), on_step and on_step(s)),
                                     images=images, image_note=image_note)
            except LLMError as exc:
                if not images:
                    raise
                # No vision model reachable: answer from captions / nearby text with the text model instead.
                stage(f"Vision model unavailable ({exc.kind}); answering from figure captions with the text model")
                warnings.append("The figure images could not be read this time, so the answer uses figure captions "
                                "and nearby text only.")
                images, visual_refs = [], set()
                outcome = run_master(question, evidence_text, use_tools, router, settings, toolbox, memory,
                                     preferences, fused.conflicts,
                                     on_step=lambda s: (steps.append(s), on_step and on_step(s)))
            answer, calls, tokens = outcome.answer, outcome.llm_calls, outcome.tokens
            provider, model = outcome.provider, outcome.model
            latency["master"] = outcome.llm_ms
            if outcome.attempts and outcome.attempts[0][2] != "ok":
                failed = ", ".join(f"{m} ({k})" for _, m, k in outcome.attempts if k != "ok")
                stage(f"Model fallback used: {failed} -> {model}")
        except LLMError as exc:
            warnings.append(f"Showing the best matching evidence without an AI answer.\n\n{exc.user_message}")

    if not answer:
        answer = "\n\n".join(f"{r.label}\n{r.full_text[:300]}" for r in registry.refs[:3]) or NOT_FOUND
    answer = normalize_answer(answer)

    # 6. Verification (+ one bounded regeneration)
    def support(sentence: str, texts: List[str]) -> List[float]:
        return support_scores(sentence, texts, settings)

    def verify(text: str):
        text, cited, bad = canonicalize(text, registry.by_number)
        ref_texts = {r.ref_num: r.full_text for r in registry.refs}
        all_text = "\n".join(ref_texts.values())
        rep = check(text, ref_texts, toolbox.observations, question, bad,
                    support_fn=support if settings.rerank else None, support_threshold=settings.support_threshold,
                    visual_refs=visual_refs)
        if (settings.verify_with_llm and outcome is not None and rep.grounded and not visual_refs
                and router.has_role("verifier")):  # a text-only checker cannot judge values read from images
            rep = llm_check(rep, text, all_text, router)
        return text, cited, rep

    answer, cited, report = verify(answer)
    latency["verifier"] = report.latency_ms
    verifier_cost = report.llm_cost
    if outcome is not None and report.needs_regeneration and settings.max_regenerations > 0:
        stage("Verification failed: regenerating once")
        try:
            outcome = regenerate(outcome, feedback_text(report), router)
            candidate, c_cited, c_report = verify(normalize_answer(outcome.answer))
            latency["master"], latency["verifier"] = outcome.llm_ms, latency["verifier"] + c_report.latency_ms
            calls, tokens, model = outcome.llm_calls, outcome.tokens, outcome.model
            verifier_cost += c_report.llm_cost
            answer, cited, report, regenerated = candidate, c_cited, c_report, True
        except LLMError as exc:
            warnings.append(f"Regeneration skipped: {exc.user_message}")
    if outcome is not None:
        stage({"verified": "Verification passed", "partial": "Verification passed (some citations weakly supported)"}
              .get(report.verdict, "Verification: issues found (see warnings)"))
    if report.invalid_citations:
        warnings.append(f"The answer cites {', '.join(report.invalid_citations)}, which are not retrieved sources.")
    if report.miscited and not report.grounded:
        warnings.append("Some values are cited to the wrong source: " + "; ".join(report.miscited[:3]))
    if report.weak_support and report.grounded:
        warnings.append("Some cited sources only weakly support their sentence. Check: "
                        + " | ".join(report.weak_support[:2]))
    if report.unsupported_claims and not report.grounded:
        warnings.append("Some values could not be found in the evidence. Check them against the sources: "
                        + " | ".join(report.unsupported_claims[:3]))
    if outcome is not None and not cited and NOT_FOUND not in answer.upper():
        warnings.append("The answer has no valid citations. Verify it against the sources below.")

    not_found = NOT_FOUND in answer.upper() and len(answer) < 80
    sources = [] if not_found else (cited or list(registry.refs))
    latency["total"] = (time.perf_counter() - t_start) * 1000
    metrics = {} if not_found else rag_metrics.compute(question, answer, sources, report.as_dict(), settings,
                                                        {k: v for k, v in latency.items() if k != "total"},
                                                        tokens, calls)
    if registry.injection_lines:
        warnings.append(f"{registry.injection_lines} instruction-like line(s) in the documents were treated as data "
                        "(prompt-injection guard).")
    result = QueryResult(
        kind="not_found" if not_found else "answer", answer=answer, sources=sources, metrics=metrics,
        latency=round(latency["total"] / 1000, 2), warnings=list(dict.fromkeys(warnings)),
        used_llm=outcome is not None and bool(outcome.answer), steps=steps, llm_calls=calls, tokens=tokens,
        provider=provider, model=model, stage_latency={k: round(v, 1) for k, v in latency.items()},
        verification=report.as_dict(), regenerated=regenerated,
        cost=round((outcome.cost if outcome is not None else 0.0) + verifier_cost, 6),
        verifier_model=report.llm_model, retrieval_debug=retrieval.debug,
        routing=dict(getattr(router, "last_routing", {}) or {}), **base)
    reason = skip_reason or _not_cacheable(result, report)
    if ckey and not reason:
        stored_key = cache.store(ckey, question, result, [s.doc_id for s in result.sources if s.doc_id] or doc_ids,
                                 tenant=cache_scope.split("|", 1)[0], saved_tokens=result.tokens,
                                 saved_cost=result.cost, saved_seconds=result.latency)
        result.cache_info = {"status": "stored", "_key": stored_key}
    else:
        if reason and reason != "cache disabled":
            cache.note_not_cached(reason)
        result.cache_info = {"status": "skipped" if reason else "miss", "reason": reason}
    return result


def _not_cacheable(result: QueryResult, report) -> str:
    """Why this answer must NOT be reused ("" = safe to cache)."""
    if result.kind != "answer":
        return f"not an answer ({result.kind})"
    if not result.used_llm:
        return "no AI answer (evidence only)"
    if report.verdict not in ("verified", "partial") or not report.grounded:
        return "not verified against the sources"
    if report.invalid_citations:
        return "invalid citations"
    if not result.sources:
        return "no sources"
    if any(("budget" in w.lower() or "without an ai answer" in w.lower()) for w in result.warnings):
        return "degraded answer (budget or provider error)"
    return ""
