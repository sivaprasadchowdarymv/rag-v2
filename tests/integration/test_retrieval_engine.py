"""Phase 3: rewriting, multi-query, hybrid modes, RRF, rerank, filters, parent-child, compression, debug trace."""
from __future__ import annotations

import pytest

from agents.orchestrator import plan
from agents.retrieval_stage import run_retrieval
from metrics.benchmark import retrieval_scores
from retrieval import query_rewriter as qr
from retrieval.hybrid_search import MODES, HybridSearcher, MetadataFilter, compress

QUESTIONS = [("What is the peak output current?", [2]), ("What is the typical quiescent current?", [2]),
             ("What is the dropout voltage VDO?", [2]), ("Which equation gives the power dissipation PD?", [3]),
             ("How is the junction temperature TJ calculated?", [3]), ("What is the function of pin 2?", [4]),
             ("What is the absolute maximum input voltage?", [1])]


def test_rewrite_removes_filler_and_expands():
    assert qr.rewrite("Could you please tell me what is the max temp?") == "maximum temperature"
    assert "VDO" in qr.rewrite("What is the dropout voltage VDO?")  # symbols kept
    assert "limit" in qr.expand("max voltage")
    v = qr.variants("Can you tell me the max temp?", 3)
    assert v[0] == "Can you tell me the max temp?" and len(v) == 3 and len(set(map(str.lower, v))) == 3


def test_llm_variant_is_optional_and_safe():
    assert qr.variants("q one", 2, llm=lambda q: 1 / 0) == ["q one"]  # failing LLM never breaks retrieval


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_returns_scored_hits(indexed, mode):
    index, s = indexed
    hits, debug = HybridSearcher(s).search("dropout voltage VDO", [index], k=5, mode=mode)
    assert hits and debug["mode"] == mode and len(debug["hits"]) == len(hits)
    assert all(h.rrf > 0 for h in hits)
    if mode == "rerank":
        assert hits[0].rerank is not None


def test_metadata_filters_are_pre_filters(indexed):
    index, s = indexed
    hits, _ = HybridSearcher(s).search("voltage", [index], k=10, mode="hybrid",
                                       filters=MetadataFilter(types={"text"}, pages=(1, 1)))
    assert hits and all(h.node.type == "text" and h.node.page == 1 for h in hits)
    none, _ = HybridSearcher(s).search("voltage", [index], k=5, filters=MetadataFilter(doc_ids={"other"}))
    assert none == []
    assert MetadataFilter.from_dict({"types": ["table"], "pages": [2, 3]}).pages == (2, 3)


def test_pipeline_filters_and_debug(indexed):
    index, s = indexed
    out = run_retrieval(plan("What is the output voltage?"), [index], s,
                        filters=MetadataFilter(pages=(2, 2)))
    assert out.fusion.evidence and all(e.page == 2 for e in out.fusion.evidence)
    d = out.debug
    assert d["queries"][0] == "What is the output voltage?" and len(d["queries"]) == 2
    assert d["candidates"] and any(c["selected"] for c in d["candidates"]) and d["filters"]["pages"] == (2, 2)
    assert any(any(k.endswith("@q1") for k in c["ranks"]) for c in d["candidates"])  # variant voted in RRF


def test_parent_child_expands_strongest_text(make_settings):
    from rag.pipeline import get_index
    s = make_settings(max_chunk_tokens=100, overlap_tokens=0, max_chars_per_chunk=1200)
    body = " ".join(f"Line {i} describes the cooling loop and its pump." for i in range(12))
    md = f"# Cooling\n\n{body}\n\nThe coolant flow rate is 14 litres per minute.\n\n# Other\n\nUnrelated text here."
    index, _ = get_index(md.encode(), "cool.md", s)
    assert sum(1 for n in index.nodes if n.section == "Cooling") >= 2
    out = run_retrieval(plan("What is the coolant flow rate?"), [index], s)
    top = out.fusion.evidence[0]
    assert top.meta.get("parent_expanded") and "litres per minute" in top.content and "Line 0" in top.content
    assert sum(len(e.content) for e in out.fusion.evidence) <= s.max_context_chars


def test_compression_keeps_relevant_sentences():
    text = "The sky is blue. The dropout voltage is 2.0 V at 1 A. Bananas are yellow. Output noise is low."
    out = compress(text, "dropout voltage", 40)
    assert "dropout voltage is 2.0 V" in out and "Bananas" not in out


def test_upgraded_retrieval_not_worse_than_baseline(indexed, make_settings):
    """Controlled dataset: legacy single-query retrieval vs rewrite + multi-query + parent-child (same models)."""
    index, s = indexed
    base = make_settings(query_rewrite=False, multi_query=1, parent_child=False)
    agg = {"base": [], "new": []}
    for q, pages in QUESTIONS:
        for name, st in (("base", base), ("new", s)):
            ranked = [e.page for e in run_retrieval(plan(q), [index], st).fusion.evidence]
            agg[name].append(retrieval_scores(ranked, pages, 5))
    mean = {n: {m: sum(r[m] for r in rows) / len(rows) for m in rows[0]} for n, rows in agg.items()}
    assert mean["new"]["recall@k"] >= mean["base"]["recall@k"] and mean["new"]["mrr"] >= mean["base"]["mrr"] - 0.05
