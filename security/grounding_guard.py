"""
Grounding guard: server-side, final check that an answer is backed by its cited evidence.

    report = check(answer, sources)
    report.unsupported  -> factual sentences whose citation does not support them
    report.uncited      -> factual sentences without any citation
    report.support      -> fraction of factual sentences supported (claim-to-source matching)

Claim-to-source matching is lexical: content words and numbers of a sentence must appear in the
source(s) it cites (numbers must match exactly). It complements agents/verifier.py (which also uses a
cross-encoder) and runs on every answer, including cached ones, so ungrounded text cannot bypass it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

CITE = re.compile(r"\[REF-(\d+)[^\]]*\]")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z*$\[])|\n+")
_WORD = re.compile(r"[a-zA-Zµ°Ω]{4,}")
_NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w])")
_STOP = {"this", "that", "with", "from", "which", "where", "when", "what", "have", "will", "into", "they", "their",
         "there", "these", "those", "than", "then", "also", "only", "each", "such", "used", "uses", "using", "value",
         "values", "document", "answer", "explanation", "equations", "shows", "means", "about", "under", "over"}
NOT_FOUND = "NOT FOUND IN DOCUMENT"


@dataclass
class GroundingReport:
    factual: int = 0
    supported: int = 0
    uncited: List[str] = field(default_factory=list)
    unsupported: List[str] = field(default_factory=list)

    @property
    def support(self) -> float:
        return round(self.supported / self.factual, 3) if self.factual else 1.0

    @property
    def grounded(self) -> bool:
        return not self.unsupported and self.support >= 0.5

    def as_dict(self) -> dict:
        return {"factual_sentences": self.factual, "supported": self.supported, "support": self.support,
                "uncited": self.uncited[:5], "unsupported": self.unsupported[:5], "grounded": self.grounded}


def _is_factual(sentence: str) -> bool:
    s = sentence.strip().strip("*").strip()
    return len(s) > 25 and bool(_NUM.search(CITE.sub("", s)) or len(_WORD.findall(s)) >= 5)


def check(answer: str, sources_by_ref: Dict[int, str], min_overlap: float = 0.35) -> GroundingReport:
    rep = GroundingReport()
    if not answer or NOT_FOUND in answer.upper():
        return rep
    for sent in _SENT.split(answer):
        if not _is_factual(sent) or sent.strip().startswith("$$"):
            continue
        rep.factual += 1
        refs = [int(n) for n in CITE.findall(sent)]
        if not refs:
            rep.uncited.append(sent.strip()[:160])
            continue
        source = " ".join(sources_by_ref.get(r, "") for r in refs).lower()
        if not source:
            rep.unsupported.append(sent.strip()[:160])
            continue
        body = CITE.sub(" ", sent)
        nums = [n for n in _NUM.findall(body) if len(n.strip("-")) > 0]
        nums_ok = all(re.search(rf"(?<![\d.]){re.escape(n)}(?![\d])", source) for n in nums)
        words = {w.lower() for w in _WORD.findall(body)} - _STOP
        overlap = (sum(1 for w in words if w in source) / len(words)) if words else 1.0
        if nums_ok and overlap >= min_overlap:
            rep.supported += 1
        else:
            rep.unsupported.append(sent.strip()[:160])
    return rep


def sources_map(sources: Sequence) -> Dict[int, str]:
    return {int(getattr(s, "ref_num", 0)): str(getattr(s, "full_text", "") or getattr(s, "snippet", ""))
            for s in sources}
