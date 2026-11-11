"""
HybridSearcher: one clean retrieval API with every score exposed (debugger, evaluation, API).

    hs = HybridSearcher(settings)
    hits, debug = hs.search("What is the dropout voltage?", indexes, k=5, mode="hybrid",
                            filters=MetadataFilter(types={"text", "table"}, pages=(1, 3)))

Modes
    dense     cosine similarity of the query embedding               (rag/embeddings)
    bm25      Okapi BM25 over node content                            (retrieval/bm25)
    fuzzy     RapidFuzz token-set ratio + section/metadata bonuses   (rag/retrieval, the original scorer)
    hybrid    Reciprocal Rank Fusion of dense + bm25 + fuzzy ranks
    rerank    hybrid, then the cross-encoder re-orders the top pool

Query variants (retrieval/query_rewriter) each produce ranked lists that are
fused by RRF too. Filters are applied BEFORE ranking (true pre-filtering).
Parent-child: a hit can carry its parent (all chunks of the same page+section).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from rapidfuzz import fuzz

from agents.specialists import doc_tools
from config.settings import Settings
from rag.embeddings import EmbeddingError, embed_query
from rag.models import DocumentIndex, Node
from rag.retrieval import _bonuses, semantic_scores
from retrieval import query_rewriter
from retrieval.reranker import rerank

RRF_K = 60
MODES = ("dense", "bm25", "fuzzy", "hybrid", "rerank")


@dataclass
class MetadataFilter:
    doc_ids: Optional[Set[str]] = None
    types: Optional[Set[str]] = None
    pages: Optional[Tuple[int, int]] = None      # inclusive range
    section_contains: str = ""
    ocr_only: bool = False
    # 2.1 metadata-aware filters (node level; document-level tags/category are resolved to doc_ids beforehand)
    part_numbers: Optional[Set[str]] = None      # chunk mentions one of these identifiers
    parameter_kinds: Optional[Set[str]] = None   # chunk carries a parameter of this kind (voltage, current, ...)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> Optional["MetadataFilter"]:
        if not d:
            return None
        pages = d.get("pages")

        def _set(key: str, upper: bool = False):
            v = d.get(key)
            if not v:
                return None
            items = v if isinstance(v, (list, tuple, set)) else str(v).split(",")
            return {(str(x).strip().upper() if upper else str(x).strip().lower()) for x in items if str(x).strip()}

        return cls(set(d["doc_ids"]) if d.get("doc_ids") else None, set(d["types"]) if d.get("types") else None,
                   (int(pages[0]), int(pages[1])) if pages else None, str(d.get("section_contains") or ""),
                   bool(d.get("ocr_only")), part_numbers=_set("part_numbers", upper=True),
                   parameter_kinds=_set("parameter_kinds"))

    def allows(self, doc_id: str, node: Node) -> bool:
        if self.doc_ids is not None and doc_id not in self.doc_ids:
            return False
        if self.types is not None and node.type not in self.types:
            return False
        if self.pages is not None and not (self.pages[0] <= int(node.page or 0) <= self.pages[1]):
            return False
        if self.section_contains and self.section_contains.lower() not in (node.section or "").lower():
            return False
        if self.ocr_only and not (node.meta or {}).get("ocr"):
            return False
        if self.part_numbers:
            meta_parts = {str(x).upper() for x in (node.meta or {}).get("part_numbers") or []}
            text = (node.content or "").upper()
            if not (meta_parts & self.part_numbers or any(p in text for p in self.part_numbers)):
                return False
        if self.parameter_kinds and not (self.parameter_kinds & {k.lower() for k in (node.meta or {})}):
            return False
        return True


@dataclass
class Hit:
    doc_id: str
    doc_name: str
    node_index: int
    node: Node
    dense: float = 0.0
    bm25: float = 0.0
    fuzzy: float = 0.0
    ranks: Dict[str, int] = field(default_factory=dict)
    rrf: float = 0.0
    rerank: Optional[float] = None
    parent: str = ""

    @property
    def key(self) -> str:
        return f"{self.doc_id}#n{self.node_index}"

    def as_row(self) -> dict:
        return {"doc": self.doc_name, "page": self.node.page, "type": self.node.type, "section": self.node.section,
                "dense": round(self.dense, 4), "bm25": round(self.bm25, 4), "fuzzy": round(self.fuzzy, 1),
                "rrf": round(self.rrf, 5), "rerank": None if self.rerank is None else round(self.rerank, 4),
                "ranks": dict(self.ranks), "text": self.node.raw_content[:240]}


def parent_text(index: DocumentIndex, node: Node, max_chars: int) -> str:
    """Small-to-big: all text chunks of the same page and section, in order, bounded."""
    if node.type != "text":
        return node.raw_content
    parts, seen = [], set()
    for n in index.nodes:
        if n.type == "text" and n.page == node.page and n.section == node.section and n.raw_content not in seen:
            seen.add(n.raw_content)
            parts.append(n.raw_content)
    text = "\n".join(parts)
    return text if len(text) <= max_chars else node.raw_content


class HybridSearcher:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _candidates(self, index: DocumentIndex, flt: Optional[MetadataFilter]) -> List[int]:
        return [i for i, n in enumerate(index.nodes) if flt is None or flt.allows(index.doc_id, n)]

    def _score_index(self, query: str, index: DocumentIndex, cand: List[int]) -> Dict[str, np.ndarray]:
        qvec = None
        if index.has_embedding.any():
            try:
                qvec = embed_query(query, self.settings, index.embed_model)
            except EmbeddingError:
                qvec = None
        dense_all = semantic_scores(index, qvec) / 100.0
        bm = doc_tools(index).bm25.scores(query, cand)
        ql = query.lower()
        words, long_words = ql.split(), [w for w in ql.split() if len(w) > 3]
        fz = np.array([fuzz.token_set_ratio(ql, index.lower_content[i]) + _bonuses(ql, words, long_words,
                                                                                 index.nodes[i])
                       for i in cand], dtype=np.float32)
        return {"dense": dense_all[cand] if len(cand) else np.zeros(0), "bm25": bm, "fuzzy": fz}

    def search(self, query: str, indexes: Sequence[DocumentIndex], k: int = 5, mode: str = "rerank",
               filters: Optional[MetadataFilter] = None, n_variants: Optional[int] = None,
               pool: int = 30) -> Tuple[List[Hit], dict]:
        t0 = time.perf_counter()
        s = self.settings
        n_variants = n_variants if n_variants is not None else (s.multi_query if s.query_rewrite else 1)
        queries = query_rewriter.variants(query, n_variants) if n_variants > 1 else [query]
        signals = {"dense": ("dense",), "bm25": ("bm25",), "fuzzy": ("fuzzy",)}.get(mode, ("dense", "bm25", "fuzzy"))
        hits: Dict[str, Hit] = {}
        for qi, q in enumerate(queries):
            for index in indexes:
                cand = self._candidates(index, filters)
                if not cand:
                    continue
                scores = self._score_index(q, index, cand)
                for sig in signals:
                    arr = scores[sig]
                    order = np.argsort(-arr)[:pool]
                    for rank, j in enumerate(order):
                        if arr[j] <= 0:
                            continue
                        i = cand[j]
                        key = f"{index.doc_id}#n{i}"
                        h = hits.get(key) or Hit(index.doc_id, index.filename, i, index.nodes[i])
                        h.dense = max(h.dense, float(scores["dense"][j]))
                        h.bm25 = max(h.bm25, float(scores["bm25"][j]))
                        h.fuzzy = max(h.fuzzy, float(scores["fuzzy"][j]))
                        name = f"{sig}" if qi == 0 else f"{sig}@q{qi}"
                        h.ranks[name] = min(rank + 1, h.ranks.get(name, rank + 1))
                        hits[key] = h
        for h in hits.values():
            h.rrf = sum(1.0 / (RRF_K + r) for r in h.ranks.values())
        ranked = sorted(hits.values(), key=lambda h: h.rrf, reverse=True)
        t_rank = (time.perf_counter() - t0) * 1000
        rerank_ms, rerank_note = 0.0, ""
        if mode == "rerank" and ranked:
            top = ranked[:pool]
            scores, rerank_ms, rerank_note = rerank(query, [h.node.content for h in top], s)
            if scores:
                for h, sc in zip(top, scores):
                    h.rerank = float(sc)
                ranked = sorted(top, key=lambda h: (h.rerank, h.rrf), reverse=True) + ranked[pool:]
        out = ranked[:k]
        if s.parent_child:
            by_doc = {i.doc_id: i for i in indexes}
            for h in out:
                h.parent = parent_text(by_doc[h.doc_id], h.node, s.max_chars_per_chunk * 2)
        debug = {"query": query, "variants": queries, "mode": mode, "candidates": len(hits),
                 "filters": None if filters is None else {k_: (sorted(v) if isinstance(v, set) else v)
                                                         for k_, v in vars(filters).items() if v},
                 "latency_ms": {"rank": round(t_rank, 1), "rerank": round(rerank_ms, 1)},
                 "note": rerank_note, "hits": [h.as_row() for h in out]}
        return out, debug


def compress(text: str, query: str, max_chars: int) -> str:
    """Context compression: keep the sentences sharing the most query terms (plus their order)."""
    if len(text) <= max_chars:
        return text
    import re

    terms = {t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2} - query_rewriter._STOP
    sents = [x for x in re.split(r"(?<=[.!?])\s+|\n", text) if x.strip()]
    scored = sorted(range(len(sents)), key=lambda i: -len(terms & set(re.findall(r"[a-z0-9]+", sents[i].lower()))))
    keep, size = set(), 0
    for i in scored:
        if size + len(sents[i]) > max_chars and keep:
            continue
        keep.add(i)
        size += len(sents[i]) + 1
    return "\n".join(sents[i] for i in sorted(keep))
