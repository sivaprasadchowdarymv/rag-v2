# Security Audit — RAG Ultimate 2.0

Scope: application code in this repository, as of release 2.0.0. Method: threat modelling of each layer,
automated adversarial tests (`tests/security/`, 54 tests), static analysis (bandit), dependency audit
(pip-audit). This is a self-assessment; an independent penetration test is recommended before exposing a
deployment to untrusted users.

## Controls by layer

| Layer | Control | Where | Tested by |
|---|---|---|---|
| Upload | extension allow-list, magic-byte sniffing, size limit, executable rejection, macro/encrypted Office rejection, zip-bomb and path-traversal checks, binary-in-text rejection | `ingestion/validation.py` | `tests/integration/test_ingestion_formats.py` (9 malicious-file cases) |
| Upload | PII scan → documents auto-classified **confidential** | `ingestion/jobs.py` | `test_pii_document_is_confidential` |
| Input | prompt-injection detector (override, extraction, role hijack, delimiters, tool misuse, exfiltration, zero-width/bidi and base64 obfuscation) — block or warn | `security/injection_detector.py` | 15 attack variants + 9 benign false-positive checks |
| Input | harmful-content rules (weapons, malware, violence, self-harm → supportive response, hate, drugs) + tenant blocked terms | `security/harmful_content.py` | 6 positive + 6 false-positive checks |
| Input | topic restrictions and domain boundary | `security/topic_guard.py` | `test_topic_restrictions`, `test_policy_overrides_per_tenant` |
| Input | PII in questions: block / redact / mask / allow (default redact, before any model call) | `security/pii_detector.py`, policy engine | `test_input_pii_is_redacted_before_the_llm` (asserts the fake provider never received it) |
| Retrieval | documents treated as **untrusted data**: instruction-like lines quoted as data, system prompt says so | `ops/guard.py`, `agents/master.py` | `test_indirect_injection_in_document_is_neutralised` |
| Retrieval | tenant + document ACL at retrieval time; superseded/deleted versions denied | `core/service.authorized_indexes`, `security/policy_engine.can_read` | `tests/security/test_auth_tenancy.py` (service **and** REST API) |
| Routing | confidential/PII documents never sent to cloud models unless `ALLOW_SENSITIVE_CLOUD=true` (+ optional provider allow-list) | `llm/model_router.py` | `test_mode_and_sensitivity_exclusions`, `test_sensitive_without_local_model_refuses_cloud`, `test_service_routes_confidential_documents_locally` |
| Generation | model has read-only tools (search, read page, calculator); no network/file/shell tools | `agents/master.py` | — (by construction) |
| Output | grounding (claim-to-source matching) + verifier; abstention when ungrounded | `security/grounding_guard.py`, `agents/verifier.py`, `agents/workflow.py` | `test_unsupported_answer_is_abstained`, `test_quality_agent_abstains_on_unsupported_claims` |
| Output | secret redaction, system-prompt leak blocking, PII output policy (`authorized` default), removal of unauthorised sources | `security/output_guard.py`, policy engine | `test_output_guard_secrets_prompt_leak_and_pii`, `test_unauthorized_sources_are_removed` |
| Identity | PBKDF2-SHA256 (600k iterations, salted), password policy, lockout after 5 failures (15 min), API tokens stored as SHA-256 hashes, revocation | `security/auth.py`, `database/repository.py` | `test_password_hashing`, `test_policy_and_lockout`, `test_tokens_are_hashed_and_revocable` |
| Authorisation | RBAC admin/editor/viewer, enforced in services and API (not only UI); admin pages hidden and guarded | `security/auth.py`, `api/main.py`, `app.py` | `test_rbac_admin_only_user_management`, `test_api_end_to_end_isolation`, `test_viewer_cannot_open_admin_pages` |
| Audit | blocked/flagged input, output filtering, denied retrieval, logins, uploads, deletions, permission and policy changes | `database.models.AuditEvent` | `test_blocked_requests_are_audited` |
| Logging | request IDs; secrets/e-mails/phones redacted; question text never logged; metrics labels never contain user text | `config/logging_config.py`, `monitoring/` | `test_spans_and_safe_logs`, `test_metrics_and_health_endpoints` |
| Transport | Caddy: automatic HTTPS, HSTS, nosniff, frame options; `/metrics` not public; services bound to 127.0.0.1 | `deployment/caddy/Caddyfile`, `docker-compose.yml` | config review only (no image registry in the build sandbox) |
| Supply chain | pinned lock file, pip-audit, bandit, Trivy image scan in CI | `requirements.lock`, `.github/workflows/ci.yml` | pip-audit: **no known vulnerabilities**; bandit: **0 high / 0 medium** |

## Findings fixed during this audit

| # | Finding | Severity | Fix |
|---|---|---|---|
| S-1 | Source file contained 9 literal invisible/bidirectional Unicode characters inside a regex (Trojan-Source pattern, bandit B613) | High | replaced by `\u` escapes; repo scanned clean |
| S-2 | Pillow 12.3.0 had 39 published advisories | High | Pillow ≥ 13.0 required and locked; pip-audit clean |
| S-3 | Caddy directive order would have exposed `/api/metrics` publicly | Medium | metrics block moved inside `handle_path` |
| S-4 | "kill a hung process" matched the violence rule (false positive) | Low | rule requires a person as the object; regression tests added |
| S-5 | Superseded document versions were still readable in single-user mode | Medium | ACL denies any known-but-not-current content hash |
| S-6 | Service created a second request id per API call (audit/log correlation) | Low | inbound `X-Request-ID` propagates end to end |

## Residual risks and limitations (read before production)

1. **Detectors are heuristics.** Injection, harmful-content, topic and PII detection are regex/checksum based.
   Paraphrased or novel attacks can pass (false negatives); unusual benign wording can be blocked (false
   positives; INJECTION_POLICY=warn trades safety for fewer false positives). They are one layer among several,
   not a guarantee.
2. **PII coverage**: names, free-form postal addresses, and many national ID formats are not detected.
3. **Grounding is lexical**: a sentence that reuses the source's words but changes meaning (e.g. negation) can
   pass the claim-to-source check; the verifier's cross-encoder reduces but does not remove this.
4. **Parsers run on untrusted files** (PyMuPDF, python-docx, openpyxl, tesseract). Validation reduces the
   attack surface; run the container with no extra privileges and keep images patched (Trivy in CI).
5. **Cloud providers** receive non-sensitive document excerpts when enabled. Data classification depends on the
   uploader's label plus PII auto-detection, which can miss sensitive content without PII.
6. **AUTH_MODE=none** (default for local use) trusts whoever can reach the port; the API binds to localhost in
   that mode. Use AUTH_MODE=users for any shared deployment.
7. **Timeouts** stop waiting for an agent but cannot kill its thread (Python limitation); the result is discarded.
8. The UI session holds the authenticated principal in Streamlit session state; sessions end when the browser
   tab closes. There is no SSO/OIDC integration yet.
9. Not tested here (sandbox limits): container image build/scan, Caddy TLS, live cloud/Ollama providers.
