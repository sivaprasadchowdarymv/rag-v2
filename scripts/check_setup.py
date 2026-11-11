"""
Check your OpenRouter setup before running the app.

    python -m scripts.check_setup            # key, credits, every model id, one tiny call per role
    python -m scripts.check_setup --no-calls # only key + catalogue checks (costs nothing)

It verifies that:
  1. GROQ_API_KEY (text) and OPENROUTER_API_KEY (images) work;
  2. every model in your chains exists in that provider's live catalogue
     (model ids change when new versions ship);
  3. each role really answers: master with a tool call (Groq), fast and verifier (Groq),
     vision with an image (Gemma / Qwen); plus OpenRouter embeddings / reranker if you enabled them.
With the free profile the full check costs $0.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import load_settings  # noqa: E402
from llm.catalog import ROLES, split  # noqa: E402
from llm.model_router import ModelRouter  # noqa: E402
from llm.providers import GroqProvider, LLMError, OpenRouterProvider  # noqa: E402
from rag.tools import TOOL_SCHEMAS  # noqa: E402

OK, BAD, WARN = "\033[32m✓\033[0m", "\033[31m✗\033[0m", "\033[33m!\033[0m"


def _png() -> bytes:
    """A 64x32 PNG with a black bar (enough for a vision smoke test)."""
    w, h = 64, 32
    rows = b"".join(b"\x00" + b"".join(b"\x00\x00\x00" if 8 < y < 24 and 8 < x < 56 else b"\xff\xff\xff"
                                      for x in range(w)) for y in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-calls", action="store_true", help="skip the paid test calls")
    a = ap.parse_args()
    s = load_settings()
    failures = 0
    print(f"Profile: {s.model_profile}  (text -> Groq, images -> Qwen/Gemma on OpenRouter)")
    if not s.groq_api_key:
        failures += 1
        print(f"{BAD} GROQ_API_KEY is not set: text answers need it (free key at console.groq.com/keys).")
    if not s.openrouter_api_key:
        print(f"{WARN} OPENROUTER_API_KEY is not set: figure questions use captions only and scanned pages "
              "are not read (free key at openrouter.ai/settings/keys).")
    catalogues = {}
    for name, provider in (("groq", GroqProvider(s)), ("openrouter", OpenRouterProvider(s))):
        if not provider.configured():
            continue
        try:
            k = provider.key_info()
            extra = ""
            if name == "openrouter":
                limit = "no limit (set one at openrouter.ai/settings/keys)" if k.limit is None else f"${k.limit:.2f}"
                extra = f" · used ${k.usage:.4f} · limit {limit}"
            print(f"{OK} {provider.label} key works{extra}")
            catalogues[name] = set(provider.list_models(families=()))
        except LLMError as exc:
            failures += 1
            print(f"{BAD} {provider.label} key check failed: {exc.user_message}")
    wanted = {r: s.models_for(r) for r in ROLES}
    for role, entries in wanted.items():
        for entry in entries:
            name, model = split(entry)
            cat = catalogues.get(name)
            if cat is None:
                print(f"{WARN} {role:<9} {entry} (skipped: no {name} key)")
            elif model in cat or model.split(":")[0] in cat:
                print(f"{OK} {role:<9} {entry}")
            else:
                failures += 1
                print(f"{BAD} {role:<9} {entry} is NOT available - fix *_MODELS in .env or llm/catalog.py")
    if a.no_calls or failures:
        return 1 if failures else 0
    p = OpenRouterProvider(s)

    r = ModelRouter(s)
    tests = {
        "master": dict(messages=[{"role": "user", "content": "Use the calculate tool to compute 7*1.5."}],
                       tools=TOOL_SCHEMAS, max_tokens=300),
        "fast": dict(messages=[{"role": "user", "content": "Reply with the single word: ready"}], max_tokens=20),
        "verifier": dict(messages=[{"role": "user", "content": 'Reply with JSON only: {"grounded": true}'}],
                         max_tokens=50, json_mode=True),
        "vision": dict(messages=[{"role": "user", "content": "What shape and colour is in this image? One sentence."}],
                       images=[_png()], max_tokens=60),
    }
    total = 0.0
    for role, kw in tests.items():
        if not r.has_role(role):
            print(f"{WARN} {role:<9} skipped (no key for its provider)")
            continue
        try:
            out = r.chat(role, **kw)
            res = out.result
            total += res.cost
            what = (f"tool call {res.tool_calls[0].name}({res.tool_calls[0].arguments})" if res.tool_calls
                    else json.dumps(res.content[:60]))
            fb = "" if out.attempts[0][2] == "ok" else f"  (fallback after {out.attempts[0][1]}: {out.attempts[0][2]})"
            print(f"{OK} {role:<9} {res.provider}:{res.model} -> {what} · ${res.cost:.5f}{fb}")
            if role == "master" and not res.tool_calls:
                print(f"{WARN} master answered without calling the tool; agent mode still works but uses fewer tools")
        except LLMError as exc:
            failures += 1
            print(f"{BAD} {role:<9} {exc.user_message}")
    if s.remote_embeddings:
        try:
            e = p.embed(s.embed_model_remote, ["output voltage", "Instruct: test\nQuery: VOUT"], s.embed_dims)
            total += e.cost
            print(f"{OK} embeddings {s.embed_model_remote} -> {len(e.vectors[0])} dims · ${e.cost:.6f}")
        except LLMError as exc:
            failures += 1
            print(f"{BAD} embeddings {exc.user_message} (set EMBED_BACKEND=local to use the local model)")
    if s.remote_rerank:
        try:
            sc = p.rerank(s.rerank_model_remote, "maximum output current",
                          ["The device operates from -40 to 125 °C.", "Output current up to 1.5 A."])
            good = sc[1] > sc[0]
            print(f"{OK if good else WARN} reranker   {s.rerank_model_remote} -> {[round(x, 3) for x in sc]}")
        except LLMError as exc:
            print(f"{WARN} reranker   {exc.user_message} - the app will use the local cross-encoder instead "
                  "(or set RERANK_BACKEND=local)")
    print(f"\nTotal cost of this check: ${total:.5f}")
    print("All good." if not failures else f"{failures} problem(s) found.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
