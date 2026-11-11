"""TTL + LRU cache for verified answers (and the shared TTLCache primitive).

Only standalone questions (no conversation memory) with an answer that passed
verification are cached, so a cache hit can never return an unchecked answer.

v2 (intelligent question cache) adds, without changing the v1 API
(`TTLCache`, `normalize_question`, `answer_key`, `answers`):

* `canonical_question()`  deterministic normalisation (case, unicode, punctuation, contractions,
                          filler words, plurals, common abbreviations) so "What's the max. VOUT?" and
                          "what is the maximum vout" share one entry.
* `critical_terms()`      numbers, identifiers, negations and min/max words. A *similar* (fuzzy) match is only
                          accepted when these are identical, so "pin 3" never answers "pin 4".
* `AnswerCache`           entries live inside a *context* (security scope + authorised document versions +
                          filters + model/answer/policy settings). Lookups never cross contexts, so an answer is
                          never served to another tenant, user, document version or configuration.
                          Stats (hits, misses, expiries, evictions, savings), per-tenant clearing, invalidation
                          by document hash, and a reason counter for every answer that was NOT cached.

Everything is in-process memory (bounded, no new dependency); each process keeps its own cache.
"""
from __future__ import annotations

import copy
import hashlib
import re
import threading
import time
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple


class TTLCache:
    def __init__(self, maxsize: int, ttl: float) -> None:
        self.maxsize, self.ttl = maxsize, ttl
        self._data: "OrderedDict[str, tuple]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        if self.maxsize <= 0 or self.ttl <= 0:
            return None
        with self._lock:
            item = self._data.get(key)
            if item is None or time.time() - item[0] > self.ttl:
                self._data.pop(key, None)
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return copy.deepcopy(item[1])

    def set(self, key: str, value: Any) -> None:
        if self.maxsize <= 0 or self.ttl <= 0:
            return
        with self._lock:
            self._data[key] = (time.time(), copy.deepcopy(value))
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


def normalize_question(q: str) -> str:
    return " ".join((q or "").lower().split()).rstrip(" ?.!")


def answer_key(question: str, doc_ids: Sequence[str], *parts: Any) -> str:
    raw = "\x1f".join([normalize_question(question), ",".join(sorted(doc_ids))] + [repr(p) for p in parts])
    return hashlib.sha256(raw.encode()).hexdigest()


# =============================================================================
# v2: canonical questions
# =============================================================================
_CONTRACTIONS = {"what's": "what is", "whats": "what is", "where's": "where is", "who's": "who is",
                 "how's": "how is", "it's": "it is", "isn't": "is not", "aren't": "are not", "doesn't": "does not",
                 "don't": "do not", "can't": "cannot", "won't": "will not", "shouldn't": "should not",
                 "wasn't": "was not", "there's": "there is", "that's": "that is"}
# Words that never change what is being asked (politeness, filler, articles).
_FILLER = {"please", "kindly", "the", "a", "an", "of", "for", "me", "tell", "show", "give", "can", "could", "would",
           "you", "i", "want", "to", "know", "do", "does", "is", "are", "in", "this", "that", "document", "datasheet",
           "doc", "file", "according", "exactly", "just", "quick", "question", "hi", "hello", "hey", "thanks",
           "what", "value", "values", "s"}
_ABBREV = {"max": "maximum", "min": "minimum", "temp": "temperature", "typ": "typical", "freq": "frequency",
           "approx": "approximately", "avg": "average", "spec": "specification", "specs": "specification",
           "specifications": "specification", "volt": "voltage", "volts": "voltage", "amps": "current",
           "pinout": "pin configuration"}
_NEGATIONS = {"not", "no", "without", "except", "never", "excluding", "cannot"}
_EXTREMES = {"maximum", "minimum", "typical", "average", "absolute", "peak", "nominal", "rms", "highest", "lowest"}
_TOKEN = re.compile(r"[a-z0-9µΩ°][a-z0-9µΩ°.\-+/_]*", re.I)
_NUM = re.compile(r"^[+-]?\d")


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").replace("’", "'").replace("‘", "'")
    return text.lower()


def _tokens(question: str) -> List[str]:
    text = _fold(question)
    for k, v in _CONTRACTIONS.items():
        text = re.sub(rf"\b{re.escape(k)}\b", v, text)
    out: List[str] = []
    for raw in _TOKEN.findall(text):
        tok = raw.strip(".-+/_")
        if not tok:
            continue
        tok = _ABBREV.get(tok, tok)
        for t in tok.split():
            if not _NUM.match(t) and len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
                t = t[:-1]  # plural → singular ("ratings" == "rating")
            out.append(t)
    return out


def canonical_question(question: str) -> str:
    """Deterministic canonical form used for exact cache matches (order kept, filler removed)."""
    toks = [t for t in _tokens(question) if t not in _FILLER]
    return " ".join(toks) or normalize_question(question)


def critical_terms(question: str) -> FrozenSet[str]:
    """Terms that must be identical for two questions to be treated as the same question."""
    out: Set[str] = set()
    for t in _tokens(question):
        if any(ch.isdigit() for ch in t) or t in _NEGATIONS or t in _EXTREMES:
            out.add(t)
    for m in re.findall(r"\b[A-Z][A-Z0-9_\-]{1,}\b", unicodedata.normalize("NFKC", question or "")):
        out.add(m.lower())  # identifiers written in capitals (VOUT, GND, LM7805)
    return frozenset(out)


def similarity(a: str, b: str) -> float:
    from rapidfuzz import fuzz

    return fuzz.token_sort_ratio(a, b) / 100.0


def fingerprint(*parts: Any) -> str:
    return hashlib.sha256("\x1f".join(repr(p) for p in parts).encode()).hexdigest()


# =============================================================================
# v2: context-aware answer cache
# =============================================================================
@dataclass
class _Entry:
    ts: float
    value: Any
    context: str
    canonical: str
    critical: FrozenSet[str]
    doc_ids: Tuple[str, ...]
    tenant: str
    saved_tokens: int = 0
    saved_cost: float = 0.0
    saved_seconds: float = 0.0
    hits: int = 0


@dataclass
class CacheStats:
    hits: int = 0
    exact_hits: int = 0
    similar_hits: int = 0
    misses: int = 0
    stores: int = 0
    expired: int = 0
    evicted: int = 0
    invalidated: int = 0
    revalidation_failures: int = 0
    saved_tokens: int = 0
    saved_cost: float = 0.0
    saved_seconds: float = 0.0
    not_cached: Dict[str, int] = field(default_factory=dict)


class AnswerCache(TTLCache):
    """TTLCache-compatible (get/set by key still work) plus context-scoped lookups and statistics."""

    def __init__(self, maxsize: int, ttl: float, min_similarity: float = 0.9, enabled: bool = True) -> None:
        super().__init__(maxsize, ttl)
        self.min_similarity = min_similarity
        self.enabled = enabled
        self._entries: "OrderedDict[str, _Entry]" = OrderedDict()
        self._by_context: Dict[str, Set[str]] = {}
        self.stats_ = CacheStats()

    # ------------------------------------------------------------------ internals
    @property
    def active(self) -> bool:
        return self.enabled and self.maxsize > 0 and self.ttl > 0

    def _drop(self, key: str) -> Optional[_Entry]:
        e = self._entries.pop(key, None)
        if e is not None:
            keys = self._by_context.get(e.context)
            if keys is not None:
                keys.discard(key)
                if not keys:
                    self._by_context.pop(e.context, None)
        return e

    def _fresh(self, key: str, now: float) -> Optional[_Entry]:
        e = self._entries.get(key)
        if e is None:
            return None
        if now - e.ts > self.ttl:
            self._drop(key)
            self.stats_.expired += 1
            return None
        return e

    @staticmethod
    def entry_key(context: str, canonical: str) -> str:
        return fingerprint(context, canonical)

    # -------------------------------------------------------------------- public
    def lookup(self, context: str, question: str, allowed_doc_ids: Optional[Iterable[str]] = None
               ) -> Tuple[Optional[Any], str, Dict[str, Any]]:
        """(value | None, match: "exact" | "similar" | "", info). `allowed_doc_ids` are the documents the caller
        may read *now*; an entry citing any other document is discarded (permission revalidation)."""
        if not self.active:
            return None, "", {}
        canonical, critical = canonical_question(question), critical_terms(question)
        now = time.time()
        allowed = set(allowed_doc_ids) if allowed_doc_ids is not None else None
        with self._lock:
            key = self.entry_key(context, canonical)
            entry, match = self._fresh(key, now), "exact"
            if entry is None and self.min_similarity > 0:
                best, best_score = None, 0.0
                for k in list(self._by_context.get(context, ())):
                    cand = self._fresh(k, now)
                    if cand is None or cand.critical != critical:
                        continue
                    score = similarity(canonical, cand.canonical)
                    if score >= self.min_similarity and score > best_score:
                        best, best_score = k, score
                if best is not None:
                    key, entry, match = best, self._entries[best], "similar"
            if entry is None:
                self.stats_.misses += 1
                self.misses += 1
                return None, "", {}
            if allowed is not None and not set(entry.doc_ids) <= allowed:
                self._drop(key)
                self.stats_.revalidation_failures += 1
                self.stats_.misses += 1
                self.misses += 1
                return None, "", {"revalidation": "failed"}
            self._entries.move_to_end(key)
            entry.hits += 1
            st = self.stats_
            st.hits += 1
            self.hits += 1
            if match == "exact":
                st.exact_hits += 1
            else:
                st.similar_hits += 1
            st.saved_tokens += entry.saved_tokens
            st.saved_cost += entry.saved_cost
            st.saved_seconds += entry.saved_seconds
            info = {"match": match, "age_s": round(now - entry.ts, 1), "saved_tokens": entry.saved_tokens,
                    "saved_cost": round(entry.saved_cost, 6), "saved_seconds": round(entry.saved_seconds, 2),
                    "hits": entry.hits}
            return copy.deepcopy(entry.value), match, info

    def store(self, context: str, question: str, value: Any, doc_ids: Sequence[str], tenant: str = "",
              saved_tokens: int = 0, saved_cost: float = 0.0, saved_seconds: float = 0.0) -> Optional[str]:
        if not self.active:
            return None
        canonical = canonical_question(question)
        key = self.entry_key(context, canonical)
        with self._lock:
            self._drop(key)
            self._entries[key] = _Entry(time.time(), copy.deepcopy(value), context, canonical,
                                        critical_terms(question), tuple(sorted(set(doc_ids))), tenant,
                                        int(saved_tokens or 0), float(saved_cost or 0.0), float(saved_seconds or 0.0))
            self._by_context.setdefault(context, set()).add(key)
            self.stats_.stores += 1
            while len(self._entries) > self.maxsize:
                oldest = next(iter(self._entries))
                self._drop(oldest)
                self.stats_.evicted += 1
        return key

    def discard(self, key: Optional[str]) -> bool:
        if not key:
            return False
        with self._lock:
            return self._drop(key) is not None

    def note_not_cached(self, reason: str) -> None:
        with self._lock:
            self.stats_.not_cached[reason] = self.stats_.not_cached.get(reason, 0) + 1

    def invalidate_docs(self, doc_ids: Iterable[str]) -> int:
        """Remove every entry that used one of these document versions (content hashes)."""
        ids = set(doc_ids)
        with self._lock:
            keys = [k for k, e in self._entries.items() if ids & set(e.doc_ids)]
            for k in keys:
                self._drop(k)
            self.stats_.invalidated += len(keys)
        return len(keys)

    def clear(self, tenant: Optional[str] = None) -> int:  # type: ignore[override]
        """Remove all entries (tenant=None) or only one tenant's entries. Returns how many were removed."""
        with self._lock:
            legacy = 0
            if tenant is None:  # plain get/set (v1 API) entries have no tenant
                legacy = len(self._data)
                self._data.clear()
            keys = [k for k, e in self._entries.items() if tenant is None or e.tenant == tenant]
            for k in keys:
                self._drop(k)
            self.stats_.invalidated += len(keys)
        return len(keys) + legacy

    def reset_stats(self) -> None:
        with self._lock:
            self.stats_ = CacheStats()

    def stats(self, tenant: Optional[str] = None) -> Dict[str, Any]:
        """Counters are process-wide; `size`/`entries` are filtered by tenant when given. No keys or questions."""
        with self._lock:
            now = time.time()
            live = [e for e in self._entries.values() if now - e.ts <= self.ttl and (tenant is None
                                                                                       or e.tenant == tenant)]
            st = self.stats_
            lookups = st.hits + st.misses
            return {"enabled": self.active, "size": len(live), "maxsize": self.maxsize, "ttl_s": self.ttl,
                    "min_similarity": self.min_similarity, "hits": st.hits, "exact_hits": st.exact_hits,
                    "similar_hits": st.similar_hits, "misses": st.misses,
                    "hit_rate": round(st.hits / lookups, 3) if lookups else None, "stores": st.stores,
                    "expired": st.expired, "evicted": st.evicted, "invalidated": st.invalidated,
                    "revalidation_failures": st.revalidation_failures, "saved_tokens": st.saved_tokens,
                    "saved_cost_usd": round(st.saved_cost, 6), "saved_seconds": round(st.saved_seconds, 1),
                    "not_cached": dict(st.not_cached),
                    "entries_by_hits": sorted((e.hits for e in live), reverse=True)[:10]}


_ANSWERS: Optional[TTLCache] = None
_LOCK = threading.Lock()


def answers(maxsize: int, ttl: float, min_similarity: float = 0.9, enabled: bool = True) -> AnswerCache:
    """Process-wide answer cache. Changing size/TTL re-creates it; similarity/enable switch in place."""
    global _ANSWERS
    with _LOCK:
        if not isinstance(_ANSWERS, AnswerCache) or (_ANSWERS.maxsize, _ANSWERS.ttl) != (maxsize, ttl):
            _ANSWERS = AnswerCache(maxsize, ttl, min_similarity, enabled)
        _ANSWERS.min_similarity, _ANSWERS.enabled = min_similarity, enabled
        return _ANSWERS


def for_settings(settings) -> AnswerCache:
    return answers(settings.answer_cache_size, settings.answer_cache_ttl,
                   getattr(settings, "answer_cache_similarity", 0.9), getattr(settings, "answer_cache_enabled", True))


def current() -> Optional[AnswerCache]:
    """The live cache (or None if no question was asked yet in this process)."""
    return _ANSWERS if isinstance(_ANSWERS, AnswerCache) else None


def invalidate_docs(doc_ids: Iterable[str]) -> int:
    c = current()
    return c.invalidate_docs(doc_ids) if c is not None else 0


def retrieval_cache_stats() -> Dict[str, Any]:
    """The query-embedding cache in rag/embeddings (retrieval side, separate from answers)."""
    try:
        from rag import embeddings

        hits, misses = embeddings.QVEC_STATS["hits"], embeddings.QVEC_STATS["misses"]
        return {"size": len(embeddings._QVEC), "maxsize": embeddings._QVEC_MAX, "hits": hits, "misses": misses,
                "hit_rate": round(hits / (hits + misses), 3) if hits + misses else None}
    except Exception:  # pragma: no cover - embeddings module unavailable
        return {}
