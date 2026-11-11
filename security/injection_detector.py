"""
Prompt-injection detection for user questions (direct) and document text (indirect).

Score 0..1 from weighted category hits; categories:
    instruction_override   "ignore previous instructions", "from now on you will", "new instructions:"
    prompt_extraction      "reveal / repeat / print your system prompt / hidden instructions"
    role_hijack            "you are now DAN", "developer mode", "jailbreak", "pretend you have no rules"
    delimiter_injection    "system:" at line start, <system>, [INST], "### assistant"
    tool_misuse            requests to run shell/code, call tools on everything, exfiltrate via tools
    exfiltration           "send ... to http(s)://", Markdown images with query strings, "email the contents"
    obfuscation            zero-width / bidi control characters, base64 blobs that decode to the above

Limitations: this is a heuristic, deterministic filter. Paraphrased or novel attacks
can evade it (false negatives) and unusual but benign wording can trigger it (false
positives). It is one layer: retrieved text is also labelled untrusted in the system
prompt, the model has no write tools, and outputs are checked (output_guard).
"""
from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass, field
from typing import Dict, List

_I = re.IGNORECASE | re.MULTILINE
_PATTERNS: Dict[str, List["re.Pattern[str]"]] = {
    "instruction_override": [
        re.compile(r"\b(ignore|disregard|forget|override|bypass|skip)\b[\w\s,]{0,30}?\b(previous|prior|above|earlier|"
                   r"preceding|all|any|your|the|system|original)\b[\w\s]{0,20}?\b(instructions?|prompts?|rules?|"
                   r"messages?|directions?|guidelines?|polic(?:y|ies)|guardrails?|context)\b", _I),
        re.compile(r"\bfrom now on,?\s+(?:you|act|respond|answer|ignore)\b", _I),
        re.compile(r"\bnew\s+(?:instructions?|rules?|task)\s*:", _I),
        re.compile(r"\b(?:do not|don't|never)\s+(?:follow|obey)\s+(?:your|the|any)\s+(?:rules|instructions)", _I),
    ],
    "prompt_extraction": [
        re.compile(r"\b(reveal|show|print|repeat|output|display|leak|tell me|give me|what (?:is|are|were))\b[\w\s,]{0,30}?"
                   r"\b(system|initial|hidden|original|developer|secret)\s+(prompt|instructions?|message|rules)\b", _I),
        re.compile(r"\b(repeat|print|output)\s+(?:everything|all|the text)\s+(?:above|before)\b", _I),
        re.compile(r"\b(print|reveal|show|leak|give me)\s+(?:your|the)\s+(?:api\s+)?(?:keys?|secrets?|credentials?|"
                   r"tokens?|passwords?)\b", _I),
    ],
    "role_hijack": [
        re.compile(r"\byou\s+are\s+now\s+(?!able\b)(?:an?\s+|in\s+|the\s+)?\w+", _I),
        re.compile(r"\b(?:DAN|do anything now|developer mode|god mode|jailbreak(?:ed)?|unfiltered mode)\b", _I),
        re.compile(r"\bpretend\s+(?:that\s+)?you\s+(?:are|have)\s+(?:no|an?\s+(?:ai|assistant)\s+without)", _I),
        re.compile(r"\bact\s+as\s+(?:an?\s+)?(?:unrestricted|evil|uncensored|system|root|admin)", _I),
    ],
    "delimiter_injection": [
        re.compile(r"^\s*(?:system|assistant|developer)\s*:", _I),
        re.compile(r"<\s*/?\s*(?:system|instructions?|im_start|im_end)\s*>|\[/?INST\]|<<SYS>>", _I),
        re.compile(r"^\s*#{2,}\s*(?:system|instruction|assistant)\b", _I),
    ],
    "tool_misuse": [
        re.compile(r"\b(?:run|execute|eval)\s+(?:this\s+)?(?:shell|bash|python|code|command|script)\b", _I),
        re.compile(r"\b(?:os\.system|subprocess|rm\s+-rf|curl\s+https?://|wget\s+https?://|powershell)\b", _I),
        re.compile(r"\b(?:call|use|invoke)\s+(?:the\s+)?(?:read_page|search_datasheet|calculate|tool)s?\b[\w\s]{0,40}"
                   r"\b(?:every|all|each|send|post|upload|exfiltrate)\b", _I),
    ],
    "exfiltration": [
        re.compile(r"\b(?:send|post|upload|forward|exfiltrate|leak)\b[\w\s]{0,40}?\bto\s+(?:https?://|\S+@\S+\.\w+)", _I),
        re.compile(r"!\[[^\]]*\]\(\s*https?://[^)]*[?&][^)]*=", _I),
        re.compile(r"\bemail\s+(?:the|all|these)\s+(?:contents?|documents?|data)\b", _I),
    ],
}
_WEIGHTS = {"instruction_override": 0.6, "prompt_extraction": 0.6, "role_hijack": 0.5, "delimiter_injection": 0.5,
            "tool_misuse": 0.5, "exfiltration": 0.6, "obfuscation": 0.3}
_HIDDEN = re.compile("[\u200b\u200c\u200d\u2060\ufeff\u202a-\u202e\u2066-\u2069]")
_B64 = re.compile(r"\b[A-Za-z0-9+/]{24,}={0,2}")
BLOCK_THRESHOLD = 0.5


@dataclass
class InjectionResult:
    score: float = 0.0
    categories: List[str] = field(default_factory=list)
    matches: List[str] = field(default_factory=list)

    @property
    def detected(self) -> bool:
        return self.score >= BLOCK_THRESHOLD

    def as_dict(self) -> dict:
        return {"score": round(self.score, 2), "categories": self.categories}


def _scan(text: str, out: InjectionResult) -> None:
    for cat, patterns in _PATTERNS.items():
        for p in patterns:
            m = p.search(text)
            if m:
                if cat not in out.categories:
                    out.categories.append(cat)
                out.matches.append(m.group(0)[:80])
                break


def detect(text: str) -> InjectionResult:
    out = InjectionResult()
    if not text:
        return out
    if _HIDDEN.search(text):
        out.categories.append("obfuscation")
        text = _HIDDEN.sub("", text)
    _scan(text, out)
    for blob in _B64.findall(text)[:5]:  # encoded payloads
        try:
            decoded = base64.b64decode(blob + "=" * (-len(blob) % 4), validate=False).decode("utf-8", "ignore")
        except (binascii.Error, ValueError):
            continue
        if decoded and sum(ch.isprintable() for ch in decoded) > 0.9 * len(decoded):
            before = len(out.categories)
            _scan(decoded, out)
            if len(out.categories) > before and "obfuscation" not in out.categories:
                out.categories.append("obfuscation")
    out.score = min(1.0, sum(_WEIGHTS[c] for c in out.categories))
    return out
