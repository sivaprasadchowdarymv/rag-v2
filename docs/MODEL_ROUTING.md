# Free-model fallback routing (2.2)

Two **independent** pipelines, discovered from the providers' live model lists. No model id is hard-coded.

```text
TEXT   (roles master · fast · verifier)            VISION (role vision — figure / image questions only)
  1 Groq primary                                     1 AnyRouter free vision model
  2 Groq fallback 1                                  2 AnyRouter free vision model
  3 Groq fallback 2                                  3 AnyRouter free vision model
  4 Groq fallback 3                                  (+ local Ollama vision model if OLLAMA_VISION_MODEL is set)
  5 AnyRouter free text fallback 1
  6 AnyRouter free text fallback 2
  (+ local Ollama model if OLLAMA_BASE_URL is set)
```

Turned on automatically when `ANYROUTER_API_KEY` is set (`MODEL_DISCOVERY=auto`), or with `MODEL_DISCOVERY=on`.
With `MODEL_DISCOVERY=off`, or no AnyRouter key, the 2.1 routing is used unchanged.

## Combining Groq, AnyRouter and OpenRouter (placeholders)

In discovery mode, the chain variables accept placeholders. Each one expands to that provider's discovered,
verified-free models, in the order you write them:

| Placeholder | Expands to |
|---|---|
| `@groq` | Groq primary + 3 fallbacks |
| `@anyrouter_text` / `@anyrouter_vision` | 2 / 3 AnyRouter verified-free models |
| `@openrouter_text` / `@openrouter_vision` | 2 / 3 OpenRouter models whose id ends in `:free` **and** whose pricing is zero |

```ini
MASTER_MODELS=@groq,@anyrouter_text,@openrouter_text      # also FAST_MODELS / VERIFIER_MODELS
VISION_MODELS=@openrouter_vision,@anyrouter_vision
```

Placeholders can be mixed with explicit `provider:model` entries (for example `ollama:llava`). Unknown
placeholders are dropped with a warning. A ready-made file with every setting is `docs/env.complete.example`.

## Architecture

| Component | File | Role |
|---|---|---|
| Discovery + classification + catalogue + health | `llm/discovery.py` (new) | Groq/AnyRouter model lists → ranked candidates → slots; cache (memory + `DATA_DIR/model_catalog.json`); fail-closed revalidation; exclusion and replacement |
| AnyRouter provider | `llm/providers.py` → `AnyRouterProvider` (new class) | OpenAI-compatible wire format, error translation (401 / 402 `insufficient_balance` / 403 `free_tier_requires_subscription` / 404 / 429 / 5xx / unsupported image) |
| Chains | `llm/catalog.py` → `chain_for_settings()` | discovery chains per role; paid providers only with explicit approval |
| Cost guard | `llm/catalog.py` → `cost_guard()` | the single free-only gate, checked by the router before every candidate |
| Fallback engine | `llm/model_router.py` → `ModelRouter` (existing, extended) | `generate_text()`, `analyze_image()`, `generate_multimodal_response()` on top of the existing `chat()`; retries, attempt log, health, post-call cost check |
| RAG pipeline | `agents/master.py`, `agents/rag_pro.py` (unchanged) | images go only to role `vision`; if every vision model fails, the text chain answers from captions/text and says so |
| Admin UI | `ui/admin_pages.py` → `routing_page()` | 🔀 Model routing dashboard |

There is one router. The new methods are thin wrappers that pick the right chain; no second routing implementation exists.

## Complete routing flow

1. **Catalogue** (at most every `MODEL_CATALOG_REFRESH_MINUTES`, never per question).
   - `GET {GROQ_BASE_URL}/models` → active chat models. Excluded: speech (whisper/tts), safety classifiers
     (guard), embeddings, agentic `compound` systems, inactive/deprecated models. Ranked by the app's
     preferred families, then context window, then parameter size.
   - `GET {ANYROUTER_BASE_URL}/models` with `ANYROUTER_API_KEY` (the per-key list, including `access` and
     `can_route`). Each model is classified **fail-closed**:
     - `can_route` must be explicitly `true` (`false` → "cannot be routed"; missing → "can_route unknown");
     - `access` must mark a free route (`free` / `pool`). BYOK-only → rejected, even with a $0 gateway price.
       A credit price → paid. Missing → unknown, so rejected;
     - every pricing value must be explicitly `0` (or "free"), and both an input and an output price must be
       present. Otherwise the model is rejected as paid or unknown pricing;
     - virtual aliases (`anyrouter/free`, `anyrouter/auto`, any entry with `members`) are not used, because they
       hide which model answers. Deprecated or disabled models are excluded too;
     - **vision** additionally needs an explicit image input modality (`input_modalities` /
       `architecture.input_modalities` / `modality` "text+image->text") or a vision capability flag.
   - Every rejection is logged and shown on the dashboard with its reason.
2. **Slots**: the top 4 Groq models, the top 2 free AnyRouter text models (tool-capable and text-only preferred;
   distinct from the vision picks when possible), and the top 3 free AnyRouter vision models. Excluded models are
   skipped, so the next ranked candidate moves up automatically. If there are too few, the available ones are used
   and the shortage is logged and shown. No model id is ever invented.
3. **Request** (`ModelRouter.chat` for the role):
   - images with a text role → refused (`unsupported`); images for vision are validated **before any call**
     (PNG/JPEG/GIF/WebP magic bytes, ≤ `MODEL_MAX_IMAGE_MB`, base64-encodable) and sent as OpenAI `image_url`
     data URLs;
   - each candidate passes policy filters (mode, sensitivity, context window), then the **cost guard**, then
     the vision/text capability check;
   - errors:

     | Error | Action |
     |---|---|
     | 429, 500/502/503/504, timeout, connection | retry the same model up to `MODEL_MAX_RETRIES` with exponential backoff (0.5 s, 1 s, 2 s…, capped at 8 s; honours Retry-After), then the next model |
     | 404 model, empty answer | next model; 404 excludes the model until refresh |
     | 401 invalid key, 403 permission, 403 free tier needs a plan | **no retry**; skip every remaining model of that provider and continue with the other provider |
     | 402 credits requested | the model is marked "not free" and excluded; next model |
     | image format rejected | the vision model is excluded; next vision model |
     | 400 invalid input, content-policy refusal | stop at once (another model would not help) |

   - every attempt is logged: pipeline, provider, model, attempt number, error category, latency and result
     (`routing["attempt_log"]`, structured log line `model attempt …`, dashboard). Prompts and keys are never logged;
   - after a success, if an AnyRouter "free" model reports a cost above zero, it is excluded immediately and a
     cost alert is recorded.
4. **Health**: `MODEL_FAILURE_EXCLUDE` consecutive failures exclude a model until the next catalogue refresh.
   The existing per-model circuit breaker still applies.
5. **Revalidation**: a cached AnyRouter selection older than the refresh interval is re-checked before use. If
   AnyRouter cannot be reached, its models are paused (fail closed) and Groq keeps answering.

## Multimodal RAG protection

- Ingestion, extraction (text, tables, equations, figures), metadata and indexing are unchanged; figure files stay
  on disk whatever happens to vision calls.
- Figure questions retrieve text **and** figure evidence; only the vision chain receives images.
- If every vision model fails, the text chain answers from captions and nearby text, with the warning
  *"The figure images could not be read this time…"*. No visual interpretation is claimed.
- Sources keep document id, file name, page, section, figure/table id and extraction metadata. Citations are
  verified exactly as before.
- Text retrieval and answers never depend on the vision chain.

## Confirmations

- **Paid models cannot be called through automatic fallback.** The discovery chains contain only Groq, verified-free
  AnyRouter models and local Ollama. OpenAI, Anthropic and Bedrock are appended only when
  `MODEL_ALLOW_PAID_FALLBACK=true` **and** `MODEL_FREE_ONLY=false`. Even models an admin lists explicitly
  (`MASTER_MODELS=…`) pass through `cost_guard()` before every call. A 402 or a reported cost removes a model at
  once. (Tests: `test_paid_model_never_called_even_when_configured_explicitly`,
  `test_paid_providers_need_explicit_approval`, `test_reported_cost_excludes_model_immediately`,
  `test_402_credits_request_marks_model_paid`.)
- **The text and vision pipelines are independent.** They have separate chains, separate capability registries
  (`any_text`, `any_vision`) and separate public methods. Images are refused on text roles, and text failures never
  consume vision slots or the reverse. (Tests: `test_discovered_chains_have_required_order_and_are_independent`,
  `test_vision_chain_falls_back_in_order`, `test_vision_capability_is_verified`.)

## Known limitations

- AnyRouter's public docs confirm the `access`, `can_route`, pricing and capability fields, but not their exact JSON
  key names. The classifier therefore accepts the common shapes (string/object `access`; `pricing.prompt/completion`
  or `input/output`, numbers or strings; `input_modalities`, `architecture.*`, `modality`, `capabilities`) and
  **rejects anything it cannot read**. If AnyRouter uses other key names, models show up as rejected with
  "unknown pricing" or "unknown access" on the dashboard. Nothing paid is used in that case, but the free AnyRouter
  slots stay empty until the classifier is adapted.
- AnyRouter free models need an eligible plan (Go or higher, per AnyRouter's free-tier docs). With a key on an
  ineligible account, requests return 403 `free_tier_requires_subscription`. The router reports this as
  `entitlement` and stops using AnyRouter for that request.
- Groq is treated as the account's free tier. Whether a Groq key is billed is decided by your Groq account, not
  by this app.
- The catalogue and health state are per process. Each replica discovers and caches its own catalogue.
- The first question after start-up (or after the interval) waits for the catalogue refresh, which is bounded
  by 20 s per provider.
- Verified here with mocked APIs and a live run against the repository's fake LLM server. Real Groq/AnyRouter
  calls were not possible in the build sandbox.
