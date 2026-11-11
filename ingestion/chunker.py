"""
Configurable chunking strategies (CHUNK_STRATEGY, CHUNK_SIZE, CHUNK_OVERLAP).

    recursive   (default) section-aware recursive splitting with breadcrumbs — the original RAG.py lane
    fixed       fixed-size windows with overlap, cut at whitespace
    semantic    TextTiling-style: sentence windows, cut where lexical cohesion between neighbours drops
                (deterministic, no model calls, so indexing stays offline and reproducible)
    structure   Markdown/heading/slide structure first; a section stays whole if it fits, else paragraphs
                are packed up to the size limit (tables and lists are never split mid-line)

Every strategy returns rag.chunking.PageChunks with the same Node format
(content = "[breadcrumb]\\ntext", raw_content = text), so retrieval, citations
and caching work identically whichever strategy built the index.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable, Dict, List

from config.settings import AVG_CHARS_PER_TOK, Settings
from rag.chunking import PageChunks, chunk_page_text, detect_sections, tok
from rag.metadata import extract_metadata
from rag.models import Node

STRATEGIES = ("recursive", "fixed", "semantic", "structure")
_SENT = re.compile(r"(?<=[.!?;:])\s+|\n{2,}|\n(?=[-*•\d]+[.)]?\s)")
_WORD = re.compile(r"[a-z0-9]{3,}")
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def _node(text: str, section: str, breadcrumb: str, page: int, idx: int) -> Node:
    prefix = f"[{breadcrumb}]\n" if breadcrumb else ""
    return Node(type="text", section=section, content=f"{prefix}{text.strip()}", raw_content=text.strip(),
                page=page, chunk_idx=idx, parent_ctx=breadcrumb, meta=extract_metadata(text))


def _windows(text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    size = max(80, max_tokens * AVG_CHARS_PER_TOK)
    step = max(40, size - overlap_tokens * AVG_CHARS_PER_TOK)
    text = text.strip()
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):  # cut at the last whitespace inside the window
            cut = text.rfind(" ", start + size // 2, end)
            end = cut if cut > start else end
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= len(text):
            break
        start = max(start + 1, end - (size - step))
    return out


def fixed_chunks(text: str, page: int, max_tokens: int, overlap_tokens: int) -> PageChunks:
    res = PageChunks()
    heading = next((s["title"] for s in detect_sections(text) if s["title"] != "General"), "General")
    res.headings.append(heading)
    for i, w in enumerate(_windows(text, max_tokens, overlap_tokens)):
        res.nodes.append(_node(w, heading, heading if heading != "General" else "", page, i))
    return res


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT.split(text) if s and s.strip()]


def _cos(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    return dot / (math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values())))


def semantic_chunks(text: str, page: int, max_tokens: int, overlap_tokens: int, window: int = 2) -> PageChunks:
    res = PageChunks()
    for sec in detect_sections(text):
        sents = _sentences(sec["content"])
        if not sents:
            continue
        res.headings.append(sec["title"])
        crumb = " > ".join(res.headings[-2:]) if sec["title"] != "General" else ""
        bags = [Counter(_WORD.findall(s.lower())) for s in sents]
        gaps = []
        for i in range(1, len(sents)):
            left = sum(bags[max(0, i - window):i], Counter())
            right = sum(bags[i:i + window], Counter())
            gaps.append(_cos(left, right))
        cut_level = sorted(gaps)[len(gaps) // 4] if gaps else 0.0  # lowest quartile = topic shift
        chunks, cur = [], [sents[0]]
        for i in range(1, len(sents)):
            too_big = tok(" ".join(cur + [sents[i]])) > max_tokens
            if too_big or (gaps[i - 1] <= cut_level and tok(" ".join(cur)) > max_tokens // 4):
                chunks.append(" ".join(cur))
                cur = []
            cur.append(sents[i])
        chunks.append(" ".join(cur))
        for i, c in enumerate(chunks):
            for piece in (_windows(c, max_tokens, overlap_tokens) if tok(c) > max_tokens else [c]):
                res.nodes.append(_node(piece, sec["title"], crumb, page, i))
    return res


def _structure_sections(text: str) -> List[Dict[str, str]]:
    if not any(_MD_HEADING.match(line) for line in text.split("\n")):
        return detect_sections(text)
    sections, cur = [], {"title": "General", "content": ""}
    for line in text.split("\n"):
        m = _MD_HEADING.match(line.strip())
        if m:
            if cur["content"].strip():
                sections.append(cur)
            cur = {"title": m.group(2).strip()[:120], "content": ""}
        else:
            cur["content"] += line + "\n"
    if cur["content"].strip():
        sections.append(cur)
    return sections


def structure_chunks(text: str, page: int, max_tokens: int, overlap_tokens: int) -> PageChunks:
    res = PageChunks()
    for sec in _structure_sections(text):
        body = sec["content"].strip()
        if not body:
            continue
        res.sections.append(sec)
        res.headings.append(sec["title"])
        crumb = " > ".join(res.headings[-2:]) if sec["title"] != "General" else ""
        if tok(body) <= max_tokens:
            res.nodes.append(_node(body, sec["title"], crumb, page, 0))
            continue
        packed, cur = [], ""
        for para in re.split(r"\n{2,}", body):
            if cur and tok(cur + "\n\n" + para) > max_tokens:
                packed.append(cur)
                cur = ""
            cur = f"{cur}\n\n{para}" if cur else para
        if cur:
            packed.append(cur)
        idx = 0
        for p in packed:
            for piece in (_windows(p, max_tokens, overlap_tokens) if tok(p) > max_tokens else [p]):
                res.nodes.append(_node(piece, sec["title"], crumb, page, idx))
                idx += 1
    return res


_FUNCS: Dict[str, Callable[[str, int, int, int], PageChunks]] = {
    "recursive": chunk_page_text, "fixed": fixed_chunks, "semantic": semantic_chunks, "structure": structure_chunks,
}


def chunk_page(text: str, page: int, settings: Settings, strategy: str = "") -> PageChunks:
    strategy = strategy or settings.chunk_strategy
    if strategy == "recursive" and any(_MD_HEADING.match(ln) for ln in (text or "").split("\n")):
        strategy = "structure"  # Markdown headings (docx/html/md/pptx loaders): keep the document's own outline
    fn = _FUNCS.get(strategy, chunk_page_text)
    res = fn(text or "", page, settings.max_chunk_tokens, settings.overlap_tokens)
    if not res.nodes and len((text or "").strip()) >= 12:  # heading-only text (short OCR, title slides)
        res.nodes.append(_node(text.strip()[: settings.max_chunk_tokens * AVG_CHARS_PER_TOK], "General", "", page, 0))
    return res
