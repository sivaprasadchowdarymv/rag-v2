"""
Output guard: what leaves the server.

    * secrets       API keys / tokens / private keys / passwords in the answer → redacted, always
    * prompt leak   long verbatim overlap with the system prompt → answer withheld
    * PII           PII_OUTPUT_POLICY: block | redact | mask | allow | authorized
                    (authorized = full values for admins and users with the "pii" role, masked for others)
    * disclosure    sources from documents the principal may not read are removed (defence in depth;
                    retrieval already filters them) and the event is audited
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Set

from security import pii_detector

SECRET = re.compile(
    r"(?:\b(?:sk-(?:ant-|or-|proj-)?[A-Za-z0-9_\-]{16,}|gsk_[A-Za-z0-9]{16,}|hf_[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16}|"
    r"xox[baprs]-[A-Za-z0-9-]{10,}|ghp_[A-Za-z0-9]{30,}|rag_[A-Za-z0-9_\-]{30,})\b"
    r"|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)"
    r"|(?i:\b(?:password|passwd|api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*\S{6,}))")
PROMPT_LEAK_MESSAGE = "The answer was withheld because it repeated internal instructions."


def _shingles(text: str, n: int = 8) -> Set[str]:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {" ".join(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


@dataclass
class OutputReport:
    text: str
    actions: List[str] = field(default_factory=list)
    pii: dict = field(default_factory=dict)
    blocked: bool = False

    def as_dict(self) -> dict:
        return {"actions": self.actions, "pii": self.pii, "blocked": self.blocked}


def leaks_prompt(answer: str, prompts: Iterable[str], threshold: int = 3) -> bool:
    ans = _shingles(answer)
    if not ans:
        return False
    return any(len(ans & _shingles(p)) >= threshold for p in prompts if p)


def guard_text(text: str, pii_policy: str, pii_authorized: bool, system_prompts: Sequence[str] = ()) -> OutputReport:
    rep = OutputReport(text or "")
    if SECRET.search(rep.text):
        rep.text = SECRET.sub("[REDACTED SECRET]", rep.text)
        rep.actions.append("secret_redacted")
    if system_prompts and leaks_prompt(rep.text, system_prompts):
        rep.text, rep.blocked = PROMPT_LEAK_MESSAGE, True
        rep.actions.append("prompt_leak_blocked")
        return rep
    policy = pii_policy
    if policy == "authorized":
        policy = "allow" if pii_authorized else "mask"
    res = pii_detector.apply(rep.text, policy)
    if res.entities and policy != "allow":
        rep.pii = res.summary()
        if res.blocked:
            rep.text, rep.blocked = "The answer contains personal data and was withheld by policy.", True
            rep.actions.append("pii_blocked")
        else:
            rep.text = res.text
            rep.actions.append(f"pii_{policy}ed" if policy != "mask" else "pii_masked")
    return rep
