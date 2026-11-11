"""
PII detection and protection (deterministic, offline).

Detected entity types:
    EMAIL, PHONE, CREDIT_CARD (Luhn-checked), IBAN (mod-97 checked), US_SSN, AADHAAR (Verhoeff-checked),
    PAN (Indian tax id), IP_ADDRESS, PASSPORT (keyword-anchored), BANK_ACCOUNT (keyword-anchored), DATE_OF_BIRTH
    (keyword-anchored)

Policies (apply()):
    block   refuse the text            -> PIIResult.blocked = True
    redact  replace with  [EMAIL]      (type kept, value gone)
    mask    keep a hint: j***@example.com, ******1234
    allow   leave unchanged (use only with explicit authorisation)

Limitations (documented in docs/SECURITY_AUDIT.md): regex + checksum detection has
false negatives (names, free-form addresses and unusual formats are NOT detected)
and false positives (long part numbers can look like phone numbers; checksums
reduce but do not remove this). It is a safety net, not a guarantee.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


def _iban_ok(value: str) -> bool:
    s = value.replace(" ", "").upper()
    if not 15 <= len(s) <= 34:
        return False
    s = s[4:] + s[:4]
    num = "".join(str(int(c, 36)) for c in s)
    return int(num) % 97 == 1


_VERHOEFF_D = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
               [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
               [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
               [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
_VERHOEFF_P = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
               [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
               [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]]


def _verhoeff(digits: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
    return c == 0


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def _phone_ok(s: str) -> bool:
    return 10 <= len(_digits(s)) <= 13


def _ip_ok(s: str) -> bool:
    parts = s.split(".")
    return len(parts) == 4 and all(p.isdigit() and int(p) <= 255 for p in parts) and s not in ("0.0.0.0",)  # nosec B104 - a value comparison, not a bind address


# (type, pattern, validator)
_RULES: List[Tuple[str, "re.Pattern[str]", Optional[Callable[[str], bool]]]] = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), None),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b"), lambda s: 13 <= len(_digits(s)) <= 19
     and _luhn(_digits(s))),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b"), _iban_ok),
    ("AADHAAR", re.compile(r"\b[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}\b"), lambda s: _verhoeff(_digits(s))),
    ("US_SSN", re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"), None),
    ("PAN", re.compile(r"\b[A-Z]{3}[PCHFATBLJG][A-Z]\d{4}[A-Z]\b"), None),
    ("PHONE", re.compile(r"(?<![\w.])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,5}\)[\s.-]?)?\d{2,5}(?:[\s.-]?\d{2,5}){1,3}"
                         r"(?![\w.])"), _phone_ok),
    ("IP_ADDRESS", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), _ip_ok),
    ("PASSPORT", re.compile(r"(?i)\bpassport(?:\s*(?:no\.?|number|#))?\s*[:#]?\s*([A-Z][0-9]{7,8})\b"), None),
    ("BANK_ACCOUNT", re.compile(r"(?i)\b(?:account|a/c|acct)(?:\s*(?:no\.?|number|#))?\s*[:#]?\s*(\d{9,18})\b"),
     None),
    ("DATE_OF_BIRTH", re.compile(r"(?i)\b(?:dob|date of birth|born on)\s*[:\-]?\s*"
                                 r"(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{4}-\d{2}-\d{2})"), None),
]
_SEVERITY = {"CREDIT_CARD": 3, "IBAN": 3, "US_SSN": 3, "AADHAAR": 3, "PAN": 3, "PASSPORT": 3, "BANK_ACCOUNT": 3,
             "DATE_OF_BIRTH": 2, "EMAIL": 1, "PHONE": 1, "IP_ADDRESS": 1}


@dataclass
class PIIEntity:
    type: str
    start: int
    end: int
    value: str

    @property
    def severity(self) -> int:
        return _SEVERITY.get(self.type, 1)


@dataclass
class PIIResult:
    text: str
    entities: List[PIIEntity] = field(default_factory=list)
    blocked: bool = False
    policy: str = "allow"

    @property
    def types(self) -> List[str]:
        return sorted({e.type for e in self.entities})

    def summary(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for e in self.entities:
            out[e.type] = out.get(e.type, 0) + 1
        return out


def detect(text: str, types: Optional[Sequence[str]] = None) -> List[PIIEntity]:
    found: List[PIIEntity] = []
    taken: List[Tuple[int, int]] = []
    for kind, pattern, ok in _RULES:
        if types and kind not in types:
            continue
        for m in pattern.finditer(text or ""):
            g = 1 if m.groups() else 0
            start, end, value = m.start(g), m.end(g), m.group(g)
            if ok is not None and not ok(value):
                continue
            if any(s < end and start < e for s, e in taken):  # earlier (stronger) rule already owns it
                continue
            taken.append((start, end))
            found.append(PIIEntity(kind, start, end, value))
    return sorted(found, key=lambda e: e.start)


def mask_value(e: PIIEntity) -> str:
    v = e.value
    if e.type == "EMAIL":
        user, _, domain = v.partition("@")
        return f"{user[:1]}***@{domain}"
    digits = _digits(v)
    if len(digits) >= 4:
        return "*" * max(len(digits) - 4, 2) + digits[-4:]
    return "*" * len(v)


def apply(text: str, policy: str = "redact", types: Optional[Sequence[str]] = None,
          min_severity: int = 1) -> PIIResult:
    ents = [e for e in detect(text, types) if e.severity >= min_severity]
    if not ents or policy == "allow":
        return PIIResult(text, ents, False, policy)
    if policy == "block":
        return PIIResult(text, ents, True, policy)
    out, pos = [], 0
    for e in ents:
        out.append(text[pos:e.start])
        out.append(f"[{e.type}]" if policy == "redact" else mask_value(e))
        pos = e.end
    out.append(text[pos:])
    return PIIResult("".join(out), ents, False, policy)


def contains_pii(text: str, min_severity: int = 1) -> bool:
    return any(e.severity >= min_severity for e in detect(text))
