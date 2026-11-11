"""
Query rewriting, expansion and multi-query generation (deterministic, 0 tokens by default).

    rewrite("Can you please tell me what's the max temp?")
        -> "maximum temperature"                      (filler removed, abbreviations expanded)
    variants(q, n=3)
        -> [original, keyword rewrite, synonym expansion]

Optional (LLM_QUERY_REWRITE=true): one extra paraphrase from the "fast" model.
Variants are searched separately and merged with Reciprocal Rank Fusion, so a
bad variant can only add candidates, never remove the original's results.
"""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional

_FILLER = re.compile(
    r"^\s*(?:hi|hello|hey)?[,!\s]*(?:could|can|would|will)\s+you\s+(?:please\s+)?(?:tell|show|give|explain(?:\s+to)?)"
    r"\s+(?:me\s+)?|^\s*(?:please\s+)?(?:tell|show|give)\s+me\s+|^\s*i\s+(?:want|need|would\s+like)\s+to\s+know\s+"
    r"|\bplease\b|\bkindly\b|\bthanks?(?:\s+you)?\b|\bin\s+the\s+(?:document|datasheet|file|pdf)s?\b", re.I)
_STOP = {"what", "whats", "what's", "which", "who", "whom", "is", "are", "was", "were", "the", "a", "an", "of",
         "for", "to", "in", "on", "at", "and", "or", "does", "do", "did", "me", "about", "this", "that", "it",
         "its", "be", "by", "with", "there", "how", "much", "many", "can", "could", "would", "should", "i", "you",
         "your", "my", "our", "we", "they", "give", "tell", "show", "explain", "describe", "list"}
ABBREVIATIONS: Dict[str, str] = {
    "max": "maximum", "min": "minimum", "temp": "temperature", "typ": "typical", "spec": "specification",
    "specs": "specifications", "freq": "frequency", "volt": "voltage", "curr": "current", "approx": "approximately",
    "config": "configuration", "info": "information", "qty": "quantity", "no.": "number", "num": "number",
    "avg": "average", "req": "requirement", "reqs": "requirements", "dept": "department", "mgmt": "management",
    "doc": "document", "docs": "documents", "pwr": "power", "gnd": "ground", "vin": "input voltage",
    "vout": "output voltage", "iout": "output current", "abs": "absolute",
}
SYNONYMS: Dict[str, List[str]] = {
    "maximum": ["max", "limit", "highest"], "minimum": ["min", "lowest"], "price": ["cost"], "cost": ["price"],
    "error": ["fault", "failure"], "fault": ["error"], "install": ["setup", "installation"],
    "setup": ["install", "configuration"], "warranty": ["guarantee"], "temperature": ["thermal"],
    "power": ["dissipation", "watts"], "current": ["amperes"], "voltage": ["volts"], "pin": ["terminal"],
    "summary": ["overview"], "overview": ["summary"], "revenue": ["sales", "income"], "employee": ["staff"],
}
_TOKEN = re.compile(r"[A-Za-z0-9µΩ°.+\-_/]+")


def _tokens(text: str) -> List[str]:
    return [t.strip(".") for t in _TOKEN.findall(text) if t.strip(".")]


def expand_abbreviations(text: str) -> str:
    out = []
    for tok in _tokens(text):
        low = tok.lower()
        out.append(ABBREVIATIONS.get(low, tok) if not tok.isupper() or low in ("max", "min", "typ") else tok)
    return " ".join(out)


def rewrite(question: str) -> str:
    """Keyword form of the question: filler and stop words removed, abbreviations expanded, symbols kept."""
    q = _FILLER.sub(" ", question or "")
    kept = [t for t in _tokens(expand_abbreviations(q)) if t.lower() not in _STOP]
    return " ".join(kept).strip() or (question or "").strip()


def expand(question: str, max_terms: int = 6) -> str:
    """Rewrite plus a few synonyms of its words (exact symbols are never altered)."""
    base = rewrite(question)
    extra: List[str] = []
    for t in base.lower().split():
        for syn in SYNONYMS.get(t, []):
            if syn not in base.lower() and syn not in extra:
                extra.append(syn)
    return (base + " " + " ".join(extra[:max_terms])).strip()


def variants(question: str, n: int = 2, llm: Optional[Callable[[str], str]] = None) -> List[str]:
    """Up to n distinct queries, the original first."""
    out = [question.strip()]
    for cand in (rewrite(question), expand(question)):
        if len(out) >= n:
            break
        if cand and cand.lower() not in {o.lower() for o in out}:
            out.append(cand)
    if llm is not None and len(out) < n + 1:
        try:
            para = (llm(question) or "").strip().split("\n")[0][:300]
            if para and para.lower() not in {o.lower() for o in out}:
                out.append(para)
        except Exception:  # optional enhancement; never fail retrieval
            pass
    return out


def llm_rewriter(router) -> Optional[Callable[[str], str]]:
    """Paraphrase with the cheap "fast" role (only when LLM_QUERY_REWRITE=true)."""
    if router is None or not router.has_role("fast"):
        return None

    def run(q: str) -> str:
        prompt = ("Rewrite this question as one search query for a document search engine. Keep every symbol, "
                  "number and part name. Reply with the query only.\nQuestion: " + q)
        return router.chat("fast", [{"role": "user", "content": prompt}], max_tokens=60).result.content
    return run
