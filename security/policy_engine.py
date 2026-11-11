"""
Policy engine: every server-side guardrail decision in one place (used by core/service.RAGService).

INPUT  (check_input):  length → prompt injection → harmful content → topic restrictions → PII (block/redact/
                       mask/allow) → decision + audit
OUTPUT (check_output): source authorisation (no disclosure of unreadable documents) → grounding (claim-to-source;
                       abstention) → secrets → system-prompt leakage → PII output policy
ACCESS (can_read / is_sensitive): tenant + document ACL from the database; superseded/deleted versions denied.

Configuration = settings (env) overridden per tenant by the admin UI (policy_settings table):
    injection_policy, harmful_policy, pii_input_policy, pii_output_policy, allowed_topics, restricted_topics,
    out_of_scope_message, blocked_terms, abstain_ungrounded, max_question_chars
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from config.logging_config import request_id
from config.settings import Settings, get_logger
from rag.models import DocumentIndex, QueryResult
from security import grounding_guard, harmful_content, injection_detector, output_guard, pii_detector, topic_guard

log = get_logger("policy")

DEFAULT_MAX_QUESTION = 2000
ABSTAIN_TEXT = ("I could not verify an answer to this in your documents, so I won't guess. "
                "Check the sources below or rephrase the question.")


@dataclass
class Decision:
    text: str
    blocked: bool = False
    message: str = ""
    warnings: List[str] = field(default_factory=list)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    kind: str = "blocked"

    def as_dict(self) -> Dict[str, Any]:
        return {"blocked": self.blocked, "findings": self.findings}


class PolicyEngine:
    def __init__(self, settings: Settings, repository=None, overrides: Optional[Dict[str, Any]] = None):
        self.settings = settings
        self.repo = repository
        self._overrides = overrides

    @classmethod
    def from_settings(cls, settings: Settings, repository=None) -> "PolicyEngine":
        return cls(settings, repository)

    # --------------------------------------------------------------- config
    def config(self, tenant_id: str = "default") -> Dict[str, Any]:
        s = self.settings
        cfg: Dict[str, Any] = {
            "injection_policy": s.injection_policy, "harmful_policy": s.harmful_policy,
            "pii_input_policy": s.pii_input_policy, "pii_output_policy": s.pii_output_policy,
            "allowed_topics": list(s.allowed_topics), "restricted_topics": list(s.restricted_topics),
            "out_of_scope_message": "", "blocked_terms": [], "abstain_ungrounded": s.abstain_ungrounded,
            "max_question_chars": DEFAULT_MAX_QUESTION,
        }
        if self._overrides is not None:
            cfg.update(self._overrides)
        elif self.repo is not None:
            try:
                cfg.update(self.repo.get_policy(tenant_id))
            except Exception:
                log.exception("could not read tenant policy; using defaults")
        return cfg

    # ---------------------------------------------------------------- input
    def check_input(self, question: str, principal) -> Decision:
        cfg = self.config(principal.tenant_id)
        q = (question or "").strip()
        d = Decision(q)
        if len(q) > int(cfg["max_question_chars"]):
            return self._block(d, principal, "too_long", "The question is too long. Please shorten it.")

        inj = injection_detector.detect(q)
        if inj.detected:
            d.findings.append({"type": "prompt_injection", **inj.as_dict()})
            if cfg["injection_policy"] == "block":
                return self._block(d, principal, "prompt_injection",
                                   "This request looks like an attempt to override the assistant's instructions or "
                                   "extract internal information, so it was not processed.")
            d.warnings.append("Parts of this request look like instructions to the assistant; they were ignored.")

        harm = harmful_content.detect(q, cfg.get("blocked_terms") or ())
        if harm.detected:
            d.findings.append({"type": "harmful_content", "categories": harm.categories})
            if cfg["harmful_policy"] == "block" or "self_harm" in harm.categories:
                return self._block(d, principal, "harmful_content", harm.message)
            d.warnings.append("This request touches a sensitive topic.")

        topic = topic_guard.check(q, cfg.get("allowed_topics") or (), cfg.get("restricted_topics") or (),
                                  cfg.get("out_of_scope_message") or None)
        if not topic.allowed:
            d.findings.append({"type": "topic", "reason": topic.reason, "topic": topic.topic})
            d.kind = "out_of_scope"
            return self._block(d, principal, f"topic_{topic.reason}", topic.message, kind="out_of_scope")

        pol = cfg["pii_input_policy"]
        res = pii_detector.apply(q, pol)
        if res.entities:
            d.findings.append({"type": "pii", "entities": res.summary(), "policy": pol})
            if res.blocked:
                return self._block(d, principal, "pii_input",
                                   "Your question contains personal data (" + ", ".join(res.types) + "). "
                                   "Remove it and ask again.")
            if pol in ("redact", "mask"):
                d.text = res.text
                d.warnings.append("Personal data in your question was " + ("redacted" if pol == "redact" else
                                                                           "masked") + " before processing.")
        if d.findings:
            self.audit(principal, "input_flagged", d.as_dict())
        return d

    def _block(self, d: Decision, principal, reason: str, message: str, kind: str = "blocked") -> Decision:
        d.blocked, d.message, d.kind = True, message, kind
        d.findings.append({"type": "decision", "reason": reason})
        self.audit(principal, "input_blocked", d.as_dict())
        return d

    # --------------------------------------------------------------- output
    def check_output(self, result: QueryResult, indexes: Sequence[DocumentIndex], principal) -> QueryResult:
        cfg = self.config(principal.tenant_id)
        actions: List[str] = []
        allowed_ids = {ix.doc_id for ix in indexes}
        leaked = [s for s in result.sources if getattr(s, "doc_id", "") and s.doc_id not in allowed_ids]
        if leaked:
            result.sources = [s for s in result.sources if s not in leaked]
            actions.append("unauthorized_sources_removed")
            self.audit(principal, "disclosure_prevented", {"count": len(leaked)})

        grounding: Dict[str, Any] = {}
        if result.kind == "answer" and result.used_llm and result.sources:
            rep = grounding_guard.check(result.answer, grounding_guard.sources_map(result.sources))
            grounding = rep.as_dict()
            verified = (result.verification or {}).get("verdict") in ("verified", "partial")
            if not rep.grounded and not verified and cfg["abstain_ungrounded"]:
                result.kind, result.answer = "abstained", ABSTAIN_TEXT
                actions.append("abstained_ungrounded")
            elif rep.uncited or rep.unsupported:
                result.warnings = list(dict.fromkeys(result.warnings + [
                    f"{len(rep.unsupported) + len(rep.uncited)} statement(s) could not be matched to a cited "
                    "source; check them against the evidence."]))

        authorized = principal.has("admin", "pii")
        prompts = _system_prompts()
        rep_out = output_guard.guard_text(result.answer, cfg["pii_output_policy"], authorized, prompts)
        if rep_out.actions:
            actions += rep_out.actions
            result.answer = rep_out.text
            if rep_out.blocked:
                result.kind = "blocked"
        if cfg["pii_output_policy"] != "allow" and not (cfg["pii_output_policy"] == "authorized" and authorized):
            policy = "mask" if cfg["pii_output_policy"] in ("authorized", "block") else cfg["pii_output_policy"]
            for s in result.sources:
                for attr in ("snippet", "full_text"):
                    val = getattr(s, attr, "") or ""
                    res = pii_detector.apply(val, policy)
                    if res.entities:
                        setattr(s, attr, res.text)
                        if "source_pii_masked" not in actions:
                            actions.append("source_pii_masked")
        result.security = {**(result.security or {}), "output": {"actions": actions, "pii": rep_out.pii},
                           "grounding": grounding}
        return result

    # --------------------------------------------------------------- access
    def _doc(self, doc_id: str, principal):
        return self.repo.document_for_hash(doc_id, principal.tenant_id) if self.repo else None

    def can_read(self, principal, doc_id: str) -> bool:
        doc = self._doc(doc_id, principal)
        if doc is None:
            if self.repo is not None and self.repo.hash_known(doc_id):
                return False  # superseded version, deleted document or another tenant's file
            return self.settings.auth_mode != "users"  # legacy link workspace (no accounts) only
        roles = [r for r in (doc.allowed_roles or "").split(",") if r]
        return (principal.has("admin") or doc.owner_id == principal.user_id or not roles
                or any(principal.has(r) for r in roles))

    def is_sensitive(self, index: DocumentIndex) -> bool:
        if self.repo is None:
            return False
        doc = self.repo.document_for_hash(index.doc_id)
        return doc is not None and doc.sensitivity in ("confidential", "restricted")

    def audit(self, principal, action: str, detail: Dict[str, Any]) -> None:
        from monitoring.metrics import observe_guardrail

        observe_guardrail(action)
        if self.repo is not None:
            self.repo.audit(principal.tenant_id, principal.user_id, action, detail=detail, request_id=request_id())


def _system_prompts() -> List[str]:
    try:
        from agents.master import _system_prompt
        from rag.prompts import ANSWER_FORMAT

        return [_system_prompt(True, ""), ANSWER_FORMAT]
    except Exception:  # pragma: no cover
        return []
