"""Phase 7: evaluation metrics, benchmark runner (baseline vs upgraded), storage and exports."""
from __future__ import annotations

import json

from evaluation import benchmark, metrics as M, ragas_evaluator


def test_retrieval_metrics_known_values():
    ranked = [("a.pdf", 3), ("a.pdf", 2), ("b.pdf", 2), ("a.pdf", 2)]
    m = M.retrieval_metrics(ranked, "a.pdf", [2], k=3)
    assert m["precision@k"] == 1 / 3 and m["recall@k"] == 1.0 and m["mrr"] == 0.5
    assert abs(m["ndcg@k"] - 0.6309) < 1e-3
    assert M.retrieval_metrics([("x", 1)], "a.pdf", [2], 5)["hit@k"] == 0.0


def test_generation_metrics():
    refs = {1: "Dropout voltage VDO = 2.0V", 2: "Peak output current 2.2 A"}
    good = "**Answer:** The dropout voltage is 2.0 V [REF-1: TEXT p2]."
    bad = "**Answer:** The dropout voltage is 9.9 V at room temperature [REF-1: TEXT p2]."
    assert M.answer_correct(good, ["2.0"]) == 1.0 and M.answer_correct(good, ["2.0", "zzz"]) == 0.0
    assert M.faithfulness(good, refs) == 1.0 and M.faithfulness(bad, refs) == 0.0
    assert M.hallucinated(bad, refs, []) == 1.0 and M.hallucinated(good, refs, []) == 0.0
    assert M.hallucinated("NOT FOUND IN DOCUMENT", refs, []) == 0.0
    assert M.citation_correctness([("a", 2), ("a", 5)], "a", [2]) == 0.5 and M.citation_correctness([], "a", [2]) is None
    assert M.context_relevance(["the VDO = 2.0V here", "other"], "VDO = 2.0V", ["2.0"]) == 0.5
    assert M.pct([1, 2, 3, 4, 100], 50) == 3


def test_benchmark_retrieval_and_generation(make_settings, server, tmp_path):
    s = make_settings()
    rep = benchmark.run(s, generation=True)
    assert rep["questions"] == 18 and set(rep["configs"]) == {"baseline", "upgraded"}
    for cfg in rep["configs"].values():
        summ = cfg["summary"]
        assert 0 <= summ["recall@k"] <= 1 and summ["answer_correct"] is not None
        assert summ["failed"] == 0.0 and all("model" in r for r in cfg["rows"])
    assert rep["comparison"]["mrr"]["delta"] == round(rep["comparison"]["mrr"]["upgraded"]
                                                       - rep["comparison"]["mrr"]["baseline"], 4)
    paths = benchmark.save(rep, s, tmp_path)
    assert json.loads(open(paths["json"], encoding="utf-8").read())["dataset"]
    assert "Baseline vs upgraded" in open(paths["md"], encoding="utf-8").read()
    assert open(paths["csv"], encoding="utf-8").read().startswith("config,id")
    from database.repository import get_repository
    runs = get_repository(s).list_evaluations("default")
    assert runs and runs[0].metrics["comparison"]


def test_generation_skipped_without_model(make_settings, server):
    rep = benchmark.run(make_settings(groq_api_key="", openrouter_api_key=""), generation=True)
    assert any("Generation metrics not run" in n for n in rep["notes"])
    assert "answer_correct" not in rep["configs"]["upgraded"]["summary"]


def test_ragas_never_invents_scores():
    out = ragas_evaluator.evaluate([{"question": "q", "answer": "a", "contexts": ["c"]}])
    assert out["status"] == "unavailable" and "scores" not in out
