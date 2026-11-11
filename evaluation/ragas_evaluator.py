"""
Optional RAGAS adapter. RAGAS needs an LLM judge and heavy dependencies, so it is NOT installed by default and
never runs implicitly. When `pip install ragas` is done and a judge model is configured, `evaluate()` scores the
same records the built-in metrics use; otherwise it reports why it did not run (it never returns made-up scores).
"""
from __future__ import annotations

from typing import Any, Dict, List


def available() -> bool:
    try:
        import ragas  # noqa: F401
        return True
    except ImportError:
        return False


def evaluate(records: List[Dict[str, Any]], llm=None, embeddings=None) -> Dict[str, Any]:
    """records: [{"question", "answer", "contexts": [..], "ground_truth"}]."""
    if not available():
        return {"status": "unavailable", "reason": "ragas is not installed (pip install ragas)"}
    if llm is None:
        return {"status": "unavailable", "reason": "RAGAS needs a judge LLM; none was provided"}
    from datasets import Dataset  # ragas dependency
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    ds = Dataset.from_list([{"question": r["question"], "answer": r["answer"], "contexts": r["contexts"],
                             "ground_truth": r.get("ground_truth", "")} for r in records])
    result = ragas_evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision], llm=llm,
                            embeddings=embeddings)
    return {"status": "ok", "scores": {k: float(v) for k, v in result.items()}}
