"""
Topic restrictions configured by administrators.

    restricted_topics   any match → refused (e.g. "salary", "medical advice", "politics")
    allowed_topics      if set, a question must touch at least one allowed topic (domain boundary);
                        greetings and questions about the document itself are always allowed
    out_of_scope_message  shown when a question is outside the boundary

A topic is a comma-free phrase; it matches by whole words (case-insensitive), and a topic can list
synonyms with "|" (e.g. "hr|human resources|leave policy").
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

DEFAULT_OUT_OF_SCOPE = ("This assistant only answers questions within its configured scope. "
                        "Please ask about the topics your administrator has enabled.")
_ALWAYS = re.compile(r"\b(document|file|page|section|summar\w*|this (?:pdf|deck|sheet)|uploaded)\b", re.I)


def _matches(topic: str, text: str) -> bool:
    return any(t.strip() and re.search(rf"\b{re.escape(t.strip().lower())}\b", text)
               for t in topic.split("|"))


@dataclass
class TopicDecision:
    allowed: bool
    reason: str = ""
    topic: str = ""
    message: str = ""


def check(question: str, allowed: Sequence[str] = (), restricted: Sequence[str] = (),
          out_of_scope_message: Optional[str] = None) -> TopicDecision:
    low = (question or "").lower()
    for topic in restricted:
        if _matches(topic, low):
            return TopicDecision(False, "restricted", topic, f"Questions about '{topic.split('|')[0]}' are not "
                                                             "permitted by your organisation's policy.")
    if allowed and not _ALWAYS.search(low) and not any(_matches(t, low) for t in allowed):
        return TopicDecision(False, "out_of_scope", "", out_of_scope_message or DEFAULT_OUT_OF_SCOPE)
    return TopicDecision(True)
