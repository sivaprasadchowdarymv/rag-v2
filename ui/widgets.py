"""
Reusable display widgets. Answers are rendered as Markdown + LaTeX (never raw
HTML from the model); every custom HTML snippet escapes its inputs.
"""
from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import streamlit as st

from config.settings import Settings
from rag.models import AgentStep, QueryResult, SourceRef
from storage.cache import DocumentPaths

ICONS = {"TEXT": "📄", "PAGE": "📄", "TABLE": "📊", "ROW": "📊", "PIN": "📌", "EQUATION": "📐",
         "FIGURE": "🖼", "DOC": "🗂"}
_CITE = re.compile(r"\[REF-(\d+):\s*([A-Z]+)\s*p(\d+)\]")
_STAGE_ICON = {"stage": "✓", "action": "🔧", "observation": "👁", "note": "ℹ️", "answer": "✓"}


def esc(v) -> str:
    return html.escape("" if v is None else str(v))


def brand() -> None:
    st.markdown('<div class="brand"><div class="logo">∞</div><div><div class="brand-title">RAG∞ Ultimate</div>'
                '<div class="brand-sub">Verified answers from your documents</div></div></div>', unsafe_allow_html=True)


def hero(title: str, subtitle: str, steps: Optional[List[tuple]] = None) -> None:
    html_steps = ""
    if steps:
        html_steps = '<div class="steps">' + "".join(
            f'<div class="step"><span class="step-n">{i + 1}</span><span class="step-t">{esc(t)}</span>'
            f'<div class="step-d">{esc(d)}</div></div>' for i, (t, d) in enumerate(steps)) + "</div>"
    st.markdown(f'<div class="hero"><div class="hero-title">{esc(title)}</div>'
                f'<div class="hero-sub">{esc(subtitle)}</div>{html_steps}</div>', unsafe_allow_html=True)


def section(label: str) -> None:
    st.markdown(f'<div class="section-label">{esc(label)}</div>', unsafe_allow_html=True)


def status_line(ok: Optional[bool], label: str, value: str) -> str:
    cls = "ok" if ok else ("warn" if ok is None else "bad")
    return f'<div class="status-line"><span class="dot {cls}"></span>{esc(label)}: {esc(value)}</div>'


def format_answer_markdown(answer: str) -> str:
    return _CITE.sub(lambda m: f":blue-background[REF-{m.group(1)} · {m.group(2)} p{m.group(3)}]", answer or "")


def evidence_chip(src: SourceRef) -> str:
    what = f"{src.item_id} p{src.page}" if src.item_id else f"Page {src.page}"
    return f'<span class="chip">{ICONS.get(src.tag, "📄")} {esc(what)}</span>'


def figure_path(settings: Settings, src: SourceRef) -> Optional[Path]:
    if not (src.figure_file and src.doc_id):
        return None
    try:
        p = (DocumentPaths(settings.data_dir, src.doc_id).figures_dir / src.figure_file).resolve()
    except ValueError:
        return None
    return p if p.exists() else None





def verification_pill(result: QueryResult) -> str:
    v = result.verification or {}
    if not v or result.kind != "answer":
        return ""
    verdict = v.get("verdict") or ("verified" if v.get("grounded") else "unverified")
    txt, cls = {"verified": ("✓ Verified", "ok"), "partial": ("≈ Partly verified", "warn"),
                "unverified": ("⚠ Check sources", "bad")}.get(verdict, ("", ""))
    if not txt:
        return ""
    if result.regenerated:
        txt += " · corrected once"
    return f'<span class="badge {cls}">{esc(txt)}</span>'


def _words(text: str):
    import time as _t

    for i, w in enumerate(text.split(" ")):
        yield ("" if i == 0 else " ") + w
        if i % 4 == 0:
            _t.sleep(0.01)


_VERDICT_TEXT = {
    "verified": "every cited fact and number was found in the quoted passages",
    "partially_verified": "most facts were found in the passages; some could only be partly confirmed",
    "unverified": "some statements could not be confirmed in the passages, so read the evidence yourself",
}
_INPUT_FINDINGS = {
    "pii": "personal data (for example a card or ID number) was found in your question and handled by the "
           "privacy policy ({policy})",
    "prompt_injection": "the text tried to change the assistant's rules or extract hidden information",
    "harmful_content": "the request asked for harmful content ({cats})",
    "topic": "the question is about a topic your organisation has restricted or that is outside the allowed scope",
}


def explain_result(result: QueryResult) -> List[tuple]:
    """Plain-English story of how this answer was produced: [(icon, title, text)], built only from what
    actually happened (security decisions, agent trace, retrieval debug, routing, verification)."""
    out: List[tuple] = []
    sec = result.security or {}
    findings = [f for f in (sec.get("input") or {}).get("findings", []) if f.get("type") != "decision"]
    if result.kind in ("blocked", "out_of_scope"):
        why = "; ".join(_INPUT_FINDINGS.get(f.get("type"), f.get("type", "a safety rule")).format(
            policy=f.get("policy", ""), cats=", ".join(f.get("categories", []))) for f in findings)
        out.append(("🛡️", "Stopped by the safety check",
                     f"Your question was checked on the server before anything else happened: {why or 'a policy rule matched'}. "
                     "It was NOT sent to any AI model and no documents were searched. The event is recorded in the "
                     "audit log."))
        return out
    if findings:
        out.append(("🛡️", "Safety check: passed with a note",
                    "; ".join(_INPUT_FINDINGS.get(f.get("type"), f.get("type", "")).format(
                        policy=f.get("policy", ""), cats=", ".join(f.get("categories", []))) for f in findings) + "."))
    else:
        out.append(("🛡️", "Safety check: passed",
                    "Your question was checked for instruction attacks, harmful requests, restricted topics and "
                    "personal data. Nothing was found."))
    trace = {t.get("agent", ""): t for t in result.agent_trace}
    qu = trace.get("Query Understanding Agent")
    if qu:
        parts = dict(kv.split("=", 1) for kv in qu.get("summary", "").split("; ") if "=" in kv)
        what = parts.get("intents", "").replace(",", ", ")
        out.append(("🧠", "Understood the question",
                    "Worked out what kind of request this is" + (f" ({parts['task']})" if parts.get("task") else "")
                    + (f" and what to look for: {what}" if what else "")
                    + (f", then rewrote it into {parts['variants']} search phrasing(s)" if parts.get("variants") else "")
                    + "."))
    rd = result.retrieval_debug or {}
    docs = sorted({s.doc_name for s in result.sources if s.doc_name})
    queries = rd.get("queries") or []
    cands = rd.get("candidates") or []
    search = (f"Searched with {len(queries)} phrasing(s)" if queries else "Searched your documents") + \
        " using meaning (dense vectors), keywords (BM25) and approximate text matching, merged the rankings (RRF)"
    search += " and re-ranked the best candidates" if any(c.get("rerank") is not None for c in cands) else ""
    search += f". {len(result.sources)} passage(s) were cited" + (f" from {', '.join(docs)}." if docs else ".")
    out.append(("🔎", "Found the evidence", search))
    reasoning = trace.get("Reasoning Agent", {})
    if result.used_llm and result.model:
        r = result.routing or {}
        why = []
        if r.get("mode"):
            why.append(f"mode {r['mode']}")
        if r.get("sensitive"):
            why.append("confidential documents, so only local models were allowed")
        if r.get("strategy"):
            why.append(f"{r['strategy']} strategy")
        out.append(("✍️", "Wrote the answer",
                    f"The AI model {result.model.split('/', 1)[-1]} ({result.provider or 'local'}) wrote the answer "
                    f"using ONLY the evidence above" + (f" — chosen because of {', '.join(why)}" if why else "") + "."))
    else:
        out.append(("📄", "Showed the evidence (no AI model)",
                    "No AI model answered (" + (reasoning.get("summary") or "none connected") + "), so the app shows "
                    "the best-matching passages instead of a written answer. Connect Ollama or a cloud key to get "
                    "written answers."))
    v = result.verification or {}
    if v.get("verdict"):
        extra = []
        if v.get("unsupported_claims"):
            extra.append(f"{len(v['unsupported_claims'])} unsupported statement(s)")
        if v.get("miscited"):
            extra.append(f"{len(v['miscited'])} wrong citation(s)")
        out.append(("✅" if v["verdict"] == "verified" else "⚠️", "Checked the answer",
                    f"The citation checker compared the answer with its sources: "
                    f"{_VERDICT_TEXT.get(v['verdict'], v['verdict'])}" + (f" ({'; '.join(extra)})" if extra else "")
                    + (". A second AI model also reviewed it." if v.get("llm_checked") else ".")))
    qa = trace.get("Answer Quality Agent")
    if qa:
        action = "accept" if "accept" in qa.get("summary", "") else ("abstain" if "abstain" in qa.get("summary", "")
                                                                       else "warn")
        out.append(("🎯", "Quality decision", {
            "accept": "The answer was good enough to show as is.",
            "warn": "The answer is shown with a warning: double-check it against the evidence.",
            "abstain": "The evidence was too weak, so the app refused to guess instead of inventing an answer.",
        }[action]))
    actions = (sec.get("output") or {}).get("actions") or []
    if actions:
        out.append(("🔒", "Final output check", "Before display the answer was cleaned: " + ", ".join(map(str, actions)) + "."))
    return out


def render_explanation(result: QueryResult) -> None:
    steps = explain_result(result)
    if not steps:
        return
    with st.expander("🧑‍🏫 How this answer was made (plain English)"):
        for i, (icon, title, text) in enumerate(steps, 1):
            st.markdown(f"**{i}. {icon} {title}** — {text}")
        if result.request_id:
            st.caption(f"Request id {result.request_id} (find it in the server logs and Agent traces).")


def render_result(result: QueryResult, settings: Settings, show_evidence: bool = True, stream: bool = False) -> None:
    """`stream=True` types the (already verified and policy-checked) answer out word by word: tokens are never
    shown before verification and the output guard have run."""
    for w in dict.fromkeys(result.warnings):
        st.warning(w)
    if result.kind in ("blocked", "out_of_scope"):
        st.error(result.answer)
        if st.session_state.get("beginner_help", True):
            render_explanation(result)
        return
    if result.kind == "abstained":
        st.info(result.answer)
    elif stream:
        st.write_stream(_words(format_answer_markdown(result.answer)))
    else:
        st.markdown(format_answer_markdown(result.answer))
    if result.sources:
        st.markdown("".join(dict.fromkeys(evidence_chip(s) for s in result.sources)), unsafe_allow_html=True)
    meta = [f"{result.latency} s"]
    if result.model:
        who = {"groq": "Groq", "openrouter": "OpenRouter"}.get(result.provider, result.provider)
        meta.append(f"{who}: {result.model.split('/', 1)[-1]}" if who else result.model.split("/", 1)[-1])
    if result.verifier_model:
        meta.append(f"checked by {result.verifier_model.split('/', 1)[-1]}")
    if result.cost:
        meta.append(f"${result.cost:.4f}")
    if result.llm_calls:
        meta.append(f"{result.llm_calls} LLM call{'s' if result.llm_calls != 1 else ''}")
    if result.tokens:
        meta.append(f"~{result.tokens:,} tokens")
    cache_badge = ('<span class="badge cache" title="Reused a verified answer: no AI call, no cost">⚡ From cache</span>'
                   if result.cached else "")
    st.markdown(f'{verification_pill(result)}{cache_badge}<span class="meta">{esc(" · ".join(meta))}</span>',
                unsafe_allow_html=True)
    render_summary(result)
    if show_evidence and result.sources:
        with st.expander(f"Evidence ({len(result.sources)})"):
            render_sources(result.sources, settings)
    if st.session_state.get("beginner_help", True):
        render_explanation(result)
    trace = [s for s in result.steps if s.kind in _STAGE_ICON]
    if trace or result.agent_trace:
        with st.expander("Agent activity"):
            if result.agent_trace:
                st.markdown("".join(f'<div class="trace-line">{"✓" if t["status"] == "ok" else "⚠"} '
                                    f'<b>{esc(t["agent"])}</b> · {t["ms"]:.0f} ms · {esc(t["summary"])}</div>'
                                    for t in result.agent_trace), unsafe_allow_html=True)
            render_trace(trace)
    if result.routing and result.routing.get("chosen"):
        with st.expander("Why this model"):
            r = result.routing
            st.markdown(f"Chosen **{esc(r['chosen'])}** · mode {esc(r.get('mode'))} · strategy "
                        f"{esc(r.get('strategy'))} · {'confidential data: local only' if r.get('sensitive') else 'no confidential data'}")
            if r.get("excluded"):
                st.caption(" · ".join(f"{k}: {v}" for k, v in r["excluded"].items()))


def render_sources(sources: Iterable[SourceRef], settings: Settings) -> None:
    for src in sources:
        head = f"{ICONS.get(src.tag, '📄')} **REF-{src.ref_num}** · {src.tag} · page {src.page}"
        if src.item_id:
            head += f" · {src.item_id}"
        st.markdown(head)
        details = [x for x in (src.doc_name, f"Section: {src.section}" if src.section else "",
                               f"Path: {src.parent_ctx}" if src.parent_ctx else "",
                               f"Found by: {src.agent.replace('_', ' ')}" if src.agent else "",
                               f"Confidence: {src.score:.2f}" if isinstance(src.score, (int, float)) else "") if x]
        st.markdown(f'<div class="src-meta">{"<br>".join(esc(d) for d in details)}</div>', unsafe_allow_html=True)
        fig = figure_path(settings, src)
        if fig is not None:
            st.image(str(fig), width=360)
        st.markdown(f'<div class="src-body">{esc(src.snippet)}</div>', unsafe_allow_html=True)
        st.divider()


def describe_step(step: AgentStep) -> str:
    if step.kind == "action":
        if step.tool == "search_datasheet":
            where = step.args.get("source", "any")
            return f'Search {"" if where == "any" else where + " "}for "{step.args.get("query", "")}"'
        if step.tool == "read_page":
            return f"Read page {step.args.get('page')}"
        if step.tool == "calculate":
            return f"Calculate {step.args.get('expression', '')}"
        return f"{step.tool}"
    if step.kind == "observation":
        return f"{step.tool} result: {step.text[:160]}"
    return step.text


def render_trace(steps: List[AgentStep]) -> None:
    """Actions and stages only; no model reasoning (chain-of-thought) is ever shown."""
    for s in steps:
        if s.kind == "thought":  # legacy results may contain these; never display them
            continue
        st.markdown(f'<div class="trace-line">{_STAGE_ICON.get(s.kind, "•")} {esc(describe_step(s))}</div>',
                    unsafe_allow_html=True)


def _color(pct: float) -> str:
    return "var(--ok)" if pct >= 70 else "var(--warn)" if pct >= 40 else "var(--bad)"


def metric_cards(values: Dict[str, Optional[float]]) -> None:
    """values: label -> 0..1 (shown as %), or None to skip."""
    cells = []
    for label, v in values.items():
        if v is None:
            continue
        pct = max(0, min(100, int(round(v * 100))))
        cells.append(f'<div class="metric"><div class="metric-name">{esc(label)}</div>'
                     f'<div class="metric-val" style="color:{_color(pct)}">{pct}%</div>'
                     f'<div class="meter"><span style="width:{pct}%;background:{_color(pct)}"></span></div></div>')
    if cells:
        st.markdown(f'<div class="metrics">{"".join(cells)}</div>', unsafe_allow_html=True)


def doc_card(name: str, stats: Dict, meta: Dict, active: bool) -> str:
    scanned = meta.get("scanned_pages") or []
    badges = '<span class="badge ok">✓ Indexed</span>' + ('<span class="badge info">In chat</span>' if active else "")
    if scanned:
        badges += f'<span class="badge warn">{len(scanned)} scanned page(s), no OCR</span>'
    title = meta.get("title") or ""
    stats_html = "".join(f'<span class="stat"><b>{stats[k]}</b> {label}</span>' for k, label in
                         (("pages", "pages"), ("table", "tables"), ("equation", "equations"), ("figure", "figures")))
    return (f'<div class="card"><div class="card-title">📄 {esc(name)}</div>'
            f'<div class="card-sub">{esc(title) if title else "&nbsp;"}</div>'
            f'<div class="stat-row">{stats_html}</div><div style="margin-top:.6rem">{badges}</div></div>')


def missing_card(name: str) -> str:
    return (f'<div class="card"><div class="card-title">📄 {esc(name)}</div>'
            '<div class="card-sub">The saved index is not available (storage reset or settings changed). '
            'Upload the PDF again to restore it.</div><span class="badge warn">Needs re-upload</span></div>')



# =============================================================================
# 2.1 additions: answer execution summary, KPI tiles, empty states, onboarding, metadata
# =============================================================================
_METHOD = {"rerank": "hybrid search + rerank", "hybrid": "hybrid search (meaning + keywords + fuzzy)"}


def answer_summary(result: QueryResult) -> List[tuple]:
    """[(label, value)] factual execution summary built only from what the pipeline recorded (no reasoning)."""
    if result.kind in ("blocked", "out_of_scope"):
        return []
    rd = result.retrieval_debug or {}
    cands = [c for c in (rd.get("candidates") or []) if isinstance(c, dict)] \
        if isinstance(rd.get("candidates"), list) else []
    searched = sorted({c.get("doc") for c in cands if c.get("doc")} | {s.doc_name for s in result.sources if s.doc_name})
    n_ev = len(rd.get("selected") or []) or len(result.sources)
    reranked = any(c.get("rerank") is not None for c in cands)
    queries = len(rd.get("queries") or [])
    dense_off = any("keyword search" in w.lower() or "semantic search is off" in w.lower() for w in result.warnings)
    method = ("whole-document summary" if (result.plan or {}).get("task") == "summarization"
              else "keyword + fuzzy search (meaning search unavailable)" if dense_off
              else _METHOD["rerank"] if reranked else _METHOD["hybrid"])
    if reranked and dense_off:
        method += " + rerank"
    if queries > 1:
        method += f", {queries} phrasings"
    v = (result.verification or {}).get("verdict")
    ver = {"verified": "✓ verified", "partial": "≈ partly verified", "unverified": "⚠ not verified"}.get(v or "", "—")
    if result.kind == "abstained":
        ver = "withheld (weak evidence)"
    ci = result.cache_info or {}
    if result.cached:
        cache = f"hit ({ci.get('match', 'exact')} match)"
    elif ci.get("status") == "stored":
        cache = "miss · saved for reuse"
    elif ci.get("status") == "skipped":
        cache = f"not cached: {ci.get('reason', '')}"[:60]
    else:
        cache = "miss"
    model = (result.model.split("/", 1)[-1] if result.model else "no AI model (evidence only)") if not result.cached \
        else "reused verified answer"
    cited = sorted({s.doc_name for s in result.sources if s.doc_name})
    return [("Documents searched", ", ".join(searched) or "—"), ("Evidence found", f"{n_ev} passage(s)"),
            ("Retrieval", method), ("Model", model), ("Cache", cache),
            ("Sources cited", f"{len(result.sources)} from {', '.join(cited)}" if cited else "—"),
            ("Verification", ver)]


def render_summary(result: QueryResult) -> None:
    rows = answer_summary(result)
    if not rows:
        return
    cells = "".join(f'<div class="ans-cell"><div class="ans-k">{esc(k)}</div><div class="ans-v">{esc(v)}</div></div>'
                    for k, v in rows)
    with st.expander("📋 How this answer was produced (summary)", expanded=False):
        st.markdown(f'<div class="ans-sum" role="list">{cells}</div>', unsafe_allow_html=True)
        st.caption("Factual execution record (documents, evidence, method, model, cache, verification). "
                   "The model's internal reasoning is never shown.")


def kpi_tiles(items: List[tuple]) -> None:
    """items: (label, value, note, tone) with tone in ok|warn|bad|info|''."""
    cells = []
    for label, value, note, tone in items:
        cells.append(f'<div class="kpi {esc(tone)}"><div class="kpi-label">{esc(label)}</div>'
                     f'<div class="kpi-val">{esc(value)}</div>'
                     + (f'<div class="kpi-note">{esc(note)}</div>' if note else "") + "</div>")
    st.markdown(f'<div class="kpis">{"".join(cells)}</div>', unsafe_allow_html=True)


def empty_state(icon: str, title: str, text: str) -> None:
    st.markdown(f'<div class="empty"><div class="empty-icon" aria-hidden="true">{esc(icon)}</div>'
                f'<div class="empty-title">{esc(title)}</div><div class="empty-text">{esc(text)}</div></div>',
                unsafe_allow_html=True)


def onboarding(items: List[tuple], title: str = "Get started in 4 steps") -> int:
    """items: (done, text). Renders a progress checklist; returns how many are done."""
    done = sum(1 for d, _ in items if d)
    pct = int(round(100 * done / max(len(items), 1)))
    rows = "".join(f'<div class="ob-item{" done" if d else ""}"><span class="ob-check">{"✓" if d else i + 1}</span>'
                   f'<span class="ob-text">{esc(t)}</span></div>' for i, (d, t) in enumerate(items))
    st.markdown(f'<div class="onboard" role="region" aria-label="{esc(title)}"><div class="onboard-title">{esc(title)}</div>'
                f'<div class="onboard-sub">{done} of {len(items)} done</div>'
                f'<div class="ob-bar"><span style="width:{pct}%"></span></div>{rows}</div>', unsafe_allow_html=True)
    return done


def conf_badge(c: Optional[float]) -> str:
    if c is None:
        return ""
    cls = "hi" if c >= 0.8 else "mid" if c >= 0.5 else "lo"
    return f'<span class="conf {cls}" title="extraction confidence">{int(round(c * 100))}%</span>'


def source_chip(source: str) -> str:
    label = {"user": "you", "file": "file", "auto": "auto"}.get(source, source)
    cls = "user" if source == "user" else "auto"
    return f'<span class="chip {cls}" title="where this value comes from">{esc(label)}</span>'
