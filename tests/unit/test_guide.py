"""In-app Guide & Test Lab: the pass/fail logic never passes what it could not check; plain-English explanations."""
from __future__ import annotations

from rag.models import QueryResult, SourceRef
from ui.guide import CASES, evaluate
from ui.help_content import PAGE_HELP
from ui.widgets import explain_result

CASE = {c.id: c for c in CASES}


def _src(doc, page):
    return SourceRef(1, f"[REF-1: TABLE p{page}]", "TABLE", page, "", "", "", "", "table", {}, 1.0, 0, 0,
                     doc_name=doc)


def test_answer_case_passes_only_with_value_and_source():
    r = QueryResult("answer", "The peak output current is 2.2 A [REF-1: TABLE p2]", sources=[_src("LM7805X.pdf", 2)],
                    used_llm=True)
    assert evaluate(CASE["A1"], r).status == "pass"
    wrong_page = QueryResult("answer", "2.2 A", sources=[_src("LM7805X.pdf", 3)], used_llm=True)
    assert evaluate(CASE["A1"], wrong_page).status == "fail"
    wrong_value = QueryResult("answer", "1.5 A", sources=[_src("LM7805X.pdf", 2)], used_llm=True)
    assert evaluate(CASE["A1"], wrong_value).status == "fail"


def test_llm_only_case_is_skipped_not_passed_without_a_model():
    r = QueryResult("answer", "[REF-1] PD = (VIN - VOUT) x IOUT", sources=[_src("LM7805X.pdf", 3)], used_llm=False)
    out = evaluate(CASE["A5"], r)
    assert out.status == "skip" and "AI model" in out.note


def test_rate_limited_is_skipped():
    out = evaluate(CASE["S1"], QueryResult("not_found", "⏳ Too many questions right now. Please wait"))
    assert out.status == "skip"


def test_safety_cases_need_the_recorded_finding():
    sec = {"input": {"blocked": True, "findings": [{"type": "prompt_injection"}, {"type": "decision"}]}}
    assert evaluate(CASE["S1"], QueryResult("blocked", "not processed", security=sec)).status == "pass"
    assert evaluate(CASE["S1"], QueryResult("blocked", "not processed", security={})).status == "fail"
    assert evaluate(CASE["S1"], QueryResult("answer", "sure", security=sec)).status == "fail"


def test_explanation_for_blocked_and_evidence_only_answers():
    sec = {"input": {"blocked": True, "findings": [{"type": "harmful_content", "categories": ["weapons"]}]}}
    steps = explain_result(QueryResult("blocked", "no", security=sec))
    assert len(steps) == 1 and "NOT sent" in steps[0][2] and "weapons" in steps[0][2]
    r = QueryResult("answer", "x", sources=[_src("LM7805X.pdf", 2)], used_llm=False,
                    verification={"verdict": "verified"},
                    agent_trace=[{"agent": "Reasoning Agent", "summary": "no model answered: evidence-only response"},
                                 {"agent": "Answer Quality Agent", "summary": "action=accept issues=-"}])
    titles = [t for _, t, _ in explain_result(r)]
    assert "Showed the evidence (no AI model)" in titles and "Checked the answer" in titles
    assert titles[0].startswith("Safety check")


def test_every_navigable_page_has_help():
    for key in ("guide", "chat", "library", "validate", "dashboard", "ingestion", "retrieval", "traces",
                "evaluation", "insights", "settings", "models", "security", "users", "audit"):
        assert PAGE_HELP[key]["what"] and PAGE_HELP[key]["how"]
