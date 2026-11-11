"""
Evaluation metrics (deterministic; computed only from actual runs).

Retrieval (relevance = expected file AND page):
    precision@k, recall@k, hit@k, MRR, NDCG@k
Generation:
    answer_correctness    all expected strings present (case-insensitive)
    faithfulness          share of factual answer sentences supported by their cited source (grounding_guard)
    answer_relevance      question/answer term overlap (embedding similarity when available, see rag_metrics)
    context_relevance     share of retrieved items that contain the relevant context or expected answer
    citation_correctness  share of cited sources that are the expected file+page
    hallucination         1 if the answer has unsupported factual sentences or invalid citations
Operational (aggregated): latency p50/p95, retrieval latency, tokens, cost per query, failure rate, ingestion time.
"""
from __future__ import annotations

import math
import re
import statistics
from typing import Dict, Iterable, Optional, Sequence, Tuple

from security import grounding_guard

Item = Tuple[str, int]  # (filename, page)


def _rel(item: Item, expected_file: str, expected_pages: Sequence[int]) -> bool:
    return item[0] == expected_file and int(item[1] or 0) in set(expected_pages)


def retrieval_metrics(ranked: Sequence[Item], expected_file: str, expected_pages: Sequence[int], k: int) -> Dict[str, float]:
    top = list(ranked)[:k]
    rel = [_rel(it, expected_file, expected_pages) for it in top]
    relevant_total = len(set(expected_pages))
    found_pages = {int(it[1]) for it, r in zip(top, rel) if r}
    rr = next((1.0 / (i + 1) for i, it in enumerate(ranked) if _rel(it, expected_file, expected_pages)), 0.0)
    seen, dcg = set(), 0.0
    for i, (it, r) in enumerate(zip(top, rel)):
        if r and it not in seen:
            seen.add(it)
            dcg += 1.0 / math.log2(i + 2)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(relevant_total, k)))
    return {"precision@k": sum(rel) / max(len(top), 1), "recall@k": len(found_pages) / max(relevant_total, 1),
            "hit@k": 1.0 if any(rel) else 0.0, "mrr": rr, "ndcg@k": dcg / idcg if idcg else 0.0}


def answer_correct(answer: str, expected: Sequence[str]) -> float:
    low = (answer or "").lower()
    return 1.0 if expected and all(e.lower() in low for e in expected) else 0.0


def faithfulness(answer: str, sources_by_ref: Dict[int, str]) -> Optional[float]:
    rep = grounding_guard.check(answer, sources_by_ref)
    return rep.support if rep.factual else None


def context_relevance(contexts: Sequence[str], relevant_context: str, expected: Sequence[str]) -> float:
    if not contexts:
        return 0.0
    needles = [relevant_context.lower()] + [e.lower() for e in expected]
    norm = lambda t: re.sub(r"\s+", " ", t.lower())
    return sum(1 for c in contexts if any(n and norm(n) in norm(c) for n in needles)) / len(contexts)


def citation_correctness(cited: Sequence[Item], expected_file: str, expected_pages: Sequence[int]) -> Optional[float]:
    if not cited:
        return None
    return sum(1 for c in cited if _rel(c, expected_file, expected_pages)) / len(cited)


def hallucinated(answer: str, sources_by_ref: Dict[int, str], invalid_citations: Sequence[str]) -> float:
    if not answer or "NOT FOUND IN DOCUMENT" in answer.upper():
        return 0.0
    rep = grounding_guard.check(answer, sources_by_ref)
    return 1.0 if (rep.unsupported or invalid_citations) else 0.0


def pct(values: Sequence[float], p: float) -> Optional[float]:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    idx = min(len(vals) - 1, max(0, int(round(p / 100 * (len(vals) - 1)))))
    return round(vals[idx], 4)


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return round(statistics.fmean(vals), 4) if vals else None
