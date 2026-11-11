"""
Harmful-content policy (configurable, deterministic).

Categories: weapons (explosives / chemical / biological agents / firearm manufacture), malware (creating
malicious code), violence (threats, planning harm), self_harm (handled with a supportive response, never a
cold refusal), hate (dehumanising language about protected groups), illegal_drugs (synthesis / production).

HARMFUL_POLICY=block (default) refuses; warn lets the request through with an audit event. Administrators
can add their own blocked terms per tenant (policy config "blocked_terms").

Limitations: keyword/regex rules catch explicit requests only; they do not understand intent and can miss
paraphrases or block legitimate safety research. Technical documents (e.g. a datasheet's "thermal runaway"
or "explosion-proof enclosure") are deliberately NOT matched: patterns need an action + harmful object.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

_I = re.IGNORECASE
_ACT = r"(?:how (?:do i|to|can i)|steps? to|instructions? (?:for|to)|guide to|recipe for|help me|teach me|ways? to)"
RULES: Dict[str, List["re.Pattern[str]"]] = {
    "weapons": [re.compile(_ACT + r"[\w\s]{0,30}\b(?:make|build|synthesi[sz]e|assemble|produce)\b[\w\s]{0,20}"
                           r"\b(?:bomb|explosive|ied|nerve agent|sarin|ricin|anthrax|bioweapon|chemical weapon|"
                           r"pipe bomb|napalm|ghost gun|untraceable gun)\b", _I)],
    "malware": [re.compile(r"\b(?:write|create|build|code|develop|generate)\b[\w\s]{0,20}\b(?:ransomware|keylogger|"
                           r"malware|trojan|botnet|virus that|worm that|credential stealer|rootkit)\b", _I)],
    "violence": [re.compile(r"\b(?:how (?:do i|to|can i)|help me|best way to)\b[\w\s]{0,20}\b(?:kill|murder|poison|"
                            r"hurt|attack|stab|shoot)\s+(?:someone|somebody|people|him|her|them|my\s+(?:wife|husband|"
                            r"partner|boss|neighbou?r|family|parents?|brother|sister|friend|colleague)|"
                            r"an?\s+(?:person|man|woman|child|kid|human))\b", _I),
                 re.compile(r"\bi(?:'m| am)? (?:going to|gonna|will)\s+(?:kill|shoot|stab|hurt)\b", _I)],
    "self_harm": [re.compile(r"\b(?:kill myself|end my life|suicide method|ways to die|hurt myself|self[- ]harm)\b", _I)],
    "hate": [re.compile(r"\b(?:all|those)\s+\w+\s+(?:are|should be)\s+(?:vermin|subhuman|exterminated|animals)\b", _I)],
    "illegal_drugs": [re.compile(_ACT + r"[\w\s]{0,20}\b(?:cook|make|synthesi[sz]e|produce|extract)\b[\w\s]{0,15}"
                                 r"\b(?:meth(?:amphetamine)?|fentanyl|heroin|cocaine|lsd|mdma)\b", _I)],
}
SELF_HARM_RESPONSE = ("It sounds like you might be going through something really difficult. You don't have to face "
                      "it alone — please consider reaching out to someone you trust or a local crisis line "
                      "(in India: Tele-MANAS 14416; elsewhere: findahelpline.com). I'm not able to help with that "
                      "request here, but I can help you with your documents.")
REFUSAL = "This request falls outside what this assistant can help with (content policy)."


@dataclass
class HarmResult:
    categories: List[str] = field(default_factory=list)
    matches: List[str] = field(default_factory=list)

    @property
    def detected(self) -> bool:
        return bool(self.categories)

    @property
    def message(self) -> str:
        return SELF_HARM_RESPONSE if "self_harm" in self.categories else REFUSAL


def detect(text: str, extra_terms: Sequence[str] = ()) -> HarmResult:
    out = HarmResult()
    for cat, patterns in RULES.items():
        for p in patterns:
            m = p.search(text or "")
            if m:
                out.categories.append(cat)
                out.matches.append(m.group(0)[:80])
                break
    low = (text or "").lower()
    for term in extra_terms:
        t = term.strip().lower()
        if t and re.search(rf"\b{re.escape(t)}\b", low):
            out.categories.append("custom")
            out.matches.append(t)
            break
    return out
