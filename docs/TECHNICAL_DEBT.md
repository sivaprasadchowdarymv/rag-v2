# Technical Debt Register (baseline)

| # | Item | Severity | Phase that addresses it |
|---|---|---|---|
| TD-1 | Ingestion is PDF-only; MIME check is a `%PDF-` substring test | High | 2 |
| TD-2 | No local LLM; app cannot answer offline | High | 4 |
| TD-3 | Cloud providers receive any document text (no sensitivity policy) | High | 4, 6 |
| TD-4 | User questions are not screened (direct injection, PII, harmful, off-topic) | High | 6 |
| TD-5 | Single shared password; no users, roles, tenants, audit log | High | 6, 9 |
| TD-6 | Document cache shared by content hash; authorization relies on link secrecy | Medium | 6 |
| TD-7 | File-based persistence only; no migrations, no backup/restore procedure | Medium | 9 |
| TD-8 | Logs are plain text; no request IDs, traces, metrics endpoint | Medium | 10 |
| TD-9 | No CI, no dependency pinning (ranges only), no image scanning | Medium | 11 |
| TD-10 | `snapshot_download` without revision pin (bandit B615) | Low | 11 |
| TD-11 | `answer_pro` is a 190-line function mixing orchestration concerns | Low | 5 (wrapped, not rewritten) |
| TD-12 | Router retries only once on 502/503; no backoff, no Retry-After, no circuit breaker | Medium | 4 |
| TD-13 | No faithfulness / hallucination metrics; evaluation results are not stored | Medium | 7 |
| TD-14 | Single chunking strategy; chunk strategy not configurable | Low | 2 |
