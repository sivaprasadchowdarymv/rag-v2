"""
Metadata intelligence (2.1): a document-level metadata *profile* built from an indexed document.

    profile = build_profile(index, file_info={"size_bytes": ..., "sensitivity": ...})

Everything here is deterministic, local and offline (regular expressions over the parsed nodes; no LLM).
It ADDS to `rag/metadata.py` (whose per-chunk `extract_metadata()` output, used by retrieval scoring, is left
unchanged) and never modifies the index.

Every extracted value carries
    source       "file" (embedded in the file, e.g. PDF title) | "auto" (extracted by this module)
    confidence   0..1  how sure the extractor is (lower for OCR text and unit-less table values)
    pages        where it was found (provenance)
and is presented as *suggested*, never as an authoritative fact. Users can confirm or override values; their
entries are stored separately (storage/metadata_store.py) and always win in display and filtering.

Units are normalised to SI (`si_value`, `si_unit`) while the original text is kept (`raw`).
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

from rag.models import DocumentIndex, Node

PROFILE_VERSION = 2

# unit -> (SI base unit, kind)
_BASE = {"V": ("V", "voltage"), "A": ("A", "current"), "W": ("W", "power"), "Ω": ("Ω", "resistance"),
         "ohm": ("Ω", "resistance"), "F": ("F", "capacitance"), "Hz": ("Hz", "frequency"), "H": ("H", "inductance"),
         "°C": ("°C", "temperature"), "C": ("°C", "temperature"), "s": ("s", "time"), "bar": ("bar", "pressure"),
         "Pa": ("Pa", "pressure"), "rpm": ("rpm", "speed"), "%": ("%", "ratio"), "dB": ("dB", "ratio"),
         "m": ("m", "length"), "g": ("g", "mass"), "l": ("l", "volume"), "L": ("l", "volume")}
_PREFIX = {"p": 1e-12, "n": 1e-9, "µ": 1e-6, "u": 1e-6, "m": 1e-3, "k": 1e3, "K": 1e3, "M": 1e6, "G": 1e9, "": 1.0}
_UNIT_RE = (r"(?:[pnµumkKMG]?(?:V|A|W|Ω|ohm|F|Hz|H|s)|°C|°F|bar|kPa|MPa|Pa|rpm|%|dB|mm|cm|km|m|kg|g|ml|l|L)")
_ASSIGN = re.compile(rf"\b(?P<name>[A-Za-z][A-Za-z0-9_()]{{0,10}})\s*(?:=|:|≤|≥|<|>)\s*"
                     rf"(?P<val>[-+]?\d+(?:[.,]\d+)?)\s*(?P<unit>{_UNIT_RE})(?![A-Za-z])")
_VALUE_UNIT = re.compile(rf"^\s*(?P<val>[-+]?\d+(?:[.,]\d+)?)\s*(?P<unit>{_UNIT_RE})?\s*$")
_UNIT_ONLY = re.compile(rf"^\s*\(?(?P<unit>{_UNIT_RE})\)?\s*$")
_PART = re.compile(r"\b(?=[A-Z0-9\-]*\d)(?=[A-Z0-9\-]*[A-Z])[A-Z]{1,6}[\-]?\d{2,8}[A-Z0-9\-]{0,8}\b")
_STANDARD = re.compile(r"\b(?:ISO|IEC|EN|IEEE|ANSI|UL|DIN|ASTM|JEDEC|MIL-STD)[\s\-]?\d{2,6}(?:[\-:.]\d+)*\b")
_DOC_VERSION = re.compile(r"\b(?:Rev(?:ision)?\.?|Version|Ver\.?|v)\s?(?P<v>\d+(?:\.\d+){0,2}[A-Za-z]?)\b", re.I)
_FIG_REF = re.compile(r"\b(?:Figure|Fig\.)\s?(\d+[A-Za-z]?)", re.I)
_TABLE_REF = re.compile(r"\bTable\s?(\d+[A-Za-z]?)", re.I)
_COLS = ("min", "typ", "max", "nom", "value")
_NOT_PARTS = {"ISO", "IEC", "PAGE", "REV", "FIG", "TABLE"}

CATEGORY_RULES: List[Tuple[str, Tuple[str, ...]]] = [
    ("datasheet", ("datasheet", "electrical characteristics", "absolute maximum", "pin configuration", "pinout")),
    ("manual", ("manual", "installation", "maintenance", "operating instructions", "troubleshooting", "warranty")),
    ("specification", ("specification", "requirements", "shall ", "compliance")),
    ("inventory", ("stock", "inventory", "sku", "quantity", "warehouse", "price list")),
    ("guide", ("guide", "how to", "getting started", "tutorial", "step 1")),
    ("report", ("report", "summary", "findings", "analysis", "results")),
    ("policy", ("policy", "procedure", "employee", "compliance", "hr ")),
]


def _field(value: Any, source: str, confidence: float, pages: Iterable[int] = ()) -> Dict[str, Any]:
    return {"value": value, "source": source, "confidence": round(float(confidence), 2),
            "pages": sorted({int(p) for p in pages if p})[:20]}


def normalize_unit(value: float, unit: str) -> Tuple[Optional[float], str, str]:
    """(si_value, si_unit, kind) — e.g. (500, "mA") -> (0.5, "A", "current"). Unknown units: (None, unit, "")."""
    unit = (unit or "").strip()
    if not unit:
        return None, "", ""
    if unit == "°F":
        return round((value - 32) * 5 / 9, 6), "°C", "temperature"
    fixed = {"mm": ("m", 1e-3), "cm": ("m", 1e-2), "km": ("m", 1e3), "kg": ("g", 1e3), "ml": ("l", 1e-3),
             "kPa": ("Pa", 1e3), "MPa": ("Pa", 1e6)}
    if unit in fixed:
        base, f = fixed[unit]
        return round(value * f, 12), base, _BASE[base][1]
    if unit in _BASE:
        base, kind = _BASE[unit]
        return value, base, kind
    if len(unit) > 1 and unit[0] in _PREFIX and unit[1:] in _BASE:
        base, kind = _BASE[unit[1:]]
        return round(value * _PREFIX[unit[0]], 15), base, kind
    return None, unit, ""


def _num(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", "."))
    except (TypeError, ValueError):
        return None


def _conf(node: Node, base: float) -> float:
    return base * (0.8 if (node.meta or {}).get("ocr") else 1.0)


def _assignments(nodes: Iterable[Node]) -> List[Dict[str, Any]]:
    out = []
    for n in nodes:
        if n.type not in ("text", "equation"):
            continue
        for m in _ASSIGN.finditer(n.raw_content or ""):
            v = _num(m.group("val"))
            if v is None:
                continue
            si, si_unit, kind = normalize_unit(v, m.group("unit"))
            out.append({"name": m.group("name"), "kind": kind, "raw": m.group(0).strip(), "value": v,
                        "unit": m.group("unit"), "si_value": si, "si_unit": si_unit, "qualifier": "",
                        "page": n.page, "confidence": _conf(n, 0.9), "origin": "text"})
    return out


def _table_params(nodes: Iterable[Node]) -> List[Dict[str, Any]]:
    """Rows of Min/Typ/Max style tables -> parameters (unit from the row, else unit-less, lower confidence)."""
    out = []
    for n in nodes:
        if n.type != "row":
            continue
        lines = (n.raw_content or "").split("\n")
        if len(lines) < 2:
            continue
        header = [h.strip().lower() for h in lines[0].split("|")]
        cells = [c.strip() for c in lines[-1].split("|")]
        if len(cells) < 2 or not cells[0] or not any(h in _COLS or h.startswith(_COLS) for h in header):
            continue
        unit = ""
        for c in cells[1:]:
            mu = _UNIT_ONLY.match(c)
            if mu:
                unit = mu.group("unit")
        for i, c in enumerate(cells[1:], start=1):
            mv = _VALUE_UNIT.match(c)
            if not mv:
                continue
            v = _num(mv.group("val"))
            if v is None:
                continue
            u = mv.group("unit") or unit
            qual = header[i] if i < len(header) and header[i] in _COLS else ""
            si, si_unit, kind = normalize_unit(v, u)
            out.append({"name": cells[0][:60], "kind": kind, "raw": f"{cells[0]} {qual} {c}".strip(), "value": v,
                        "unit": u, "si_value": si, "si_unit": si_unit, "qualifier": qual, "page": n.page,
                        "confidence": _conf(n, 0.75 if u else 0.5), "origin": "table"})
    return out


def _dedupe_params(params: List[Dict[str, Any]], limit: int = 200) -> List[Dict[str, Any]]:
    merged: Dict[Tuple, Dict[str, Any]] = {}
    for p in params:
        key = (p["name"].lower(), p["qualifier"], p["si_value"] if p["si_value"] is not None else p["value"],
               p["si_unit"] or p["unit"])
        if key in merged:
            m = merged[key]
            m["pages"] = sorted(set(m["pages"]) | {p["page"]})
            m["occurrences"] += 1
            m["confidence"] = round(min(1.0, max(m["confidence"], p["confidence"]) + 0.05), 2)
        else:
            q = {k: v for k, v in p.items() if k != "page"}
            q.update(pages=[p["page"]], occurrences=1, source="auto", confidence=round(p["confidence"], 2))
            merged[key] = q
    return sorted(merged.values(), key=lambda p: (-p["confidence"], p["name"].lower()))[:limit]


def _part_numbers(index: DocumentIndex) -> List[Dict[str, Any]]:
    counts: Counter = Counter()
    pages: Dict[str, set] = {}
    texts = [(n.raw_content or "", n.page) for n in index.nodes] + [((index.doc_meta or {}).get("title") or "", 1)]
    for text, page in texts:
        for m in _PART.findall(text):
            if m in _NOT_PARTS or m.isdigit() or len(m) < 4:
                continue
            counts[m] += 1
            pages.setdefault(m, set()).add(page)
    out = []
    for part, c in counts.most_common(30):
        conf = 0.55 + min(0.4, 0.1 * c) + (0.05 if part in ((index.doc_meta or {}).get("title") or "") else 0)
        out.append({"value": part, "count": c, "pages": sorted(pages[part])[:20], "confidence": round(min(conf, 0.99), 2),
                    "source": "auto"})
    return out


def _refs(index: DocumentIndex) -> Dict[str, List[Dict[str, Any]]]:
    tables = [{"label": (n.section or "Table")[:80], "page": n.page} for n in index.nodes if n.type == "table"]
    equations = [{"label": (n.raw_content or "").split("\n")[0][:80], "page": n.page}
                 for n in index.nodes if n.type == "equation"][:100]
    figures = [{"label": (f.caption or "Figure")[:100], "page": f.page} for f in index.figures]
    text = "\n".join(n.raw_content or "" for n in index.nodes if n.type == "text")
    return {"tables": tables[:100], "equations": equations, "figures": figures[:100],
            "figure_refs": sorted(set(_FIG_REF.findall(text)), key=lambda x: (len(x), x))[:50],
            "table_refs": sorted(set(_TABLE_REF.findall(text)), key=lambda x: (len(x), x))[:50]}


def suggest_category(index: DocumentIndex) -> Dict[str, Any]:
    meta = index.doc_meta or {}
    blob = " ".join([index.filename, meta.get("title") or "", meta.get("subject") or ""]
                    + [s[0] for s in meta.get("sections") or [] if s]).lower()
    body = " ".join((n.raw_content or "")[:300] for n in index.nodes[:80]).lower()
    name = index.filename.lower()
    best, best_score = "other", 0.0
    for cat, words in CATEGORY_RULES:
        score = sum(2.0 for w in words if w in blob) + sum(0.5 for w in words if w in body)
        score += 3.0 if cat in name else 0.0  # "install_guide.md" is a guide, "pump_manual.docx" a manual
        if score > best_score:
            best, best_score = cat, score
    conf = 0.3 if best == "other" else min(0.95, 0.45 + 0.1 * best_score)
    return _field(best, "auto", conf)


def suggest_tags(index: DocumentIndex, category: str, parts: List[Dict[str, Any]],
                 params: List[Dict[str, Any]]) -> List[str]:
    tags = [category] if category and category != "other" else []
    fmt = (index.doc_meta or {}).get("format")
    if fmt:
        tags.append(str(fmt))
    tags += [p["value"] for p in parts[:3] if p["confidence"] >= 0.7]
    kinds = Counter(p["kind"] for p in params if p["kind"])
    tags += [k for k, _ in kinds.most_common(4)]
    if (index.doc_meta or {}).get("ocr_pages"):
        tags.append("ocr")
    return list(dict.fromkeys(t.lower() for t in tags if t))[:12]


def build_profile(index: DocumentIndex, file_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Automatically extracted metadata for one document version (index.doc_id = content hash)."""
    meta = index.doc_meta or {}
    info = file_info or {}
    stats = index.stats()
    params = _dedupe_params(_assignments(index.nodes) + _table_params(index.nodes))
    parts = _part_numbers(index)
    category = suggest_category(index)
    text_all = "\n".join((n.raw_content or "") for n in index.nodes[:400])
    version = _DOC_VERSION.search(text_all[:20000])
    standards = sorted(set(_STANDARD.findall(text_all)))[:30]
    first_line = next(((n.raw_content or "").split("\n")[0].lstrip("# ").strip() for n in index.nodes
                       if n.type == "text"), "")
    # loaders fall back to the first line of text when the file has no title property: that is a guess
    title_src = "file" if meta.get("title") and meta.get("title") != first_line[:120] else "auto"
    sec_pages = [p for _, p in meta.get("sections") or []]
    doc = {
        "title": _field(meta.get("title") or index.filename, title_src,
                        0.95 if title_src == "file" else 0.5 if meta.get("title") else 0.3, [1]),
        "author": _field(meta.get("author") or "", "file" if meta.get("author") else "auto",
                         0.95 if meta.get("author") else 0.0),
        "subject": _field(meta.get("subject") or "", "file", 0.9 if meta.get("subject") else 0.0),
        "file_type": _field(meta.get("format") or index.filename.rsplit(".", 1)[-1].lower(), "file", 1.0),
        "file_size_bytes": _field(int(info.get("size_bytes") or 0), "file", 1.0 if info.get("size_bytes") else 0.0),
        "pages": _field(index.page_count, "file", 1.0),
        "category": category,
        "version": _field(version.group("v") if version else "", "auto", 0.6 if version else 0.0,
                          [1] if version else []),
        "uploaded_at": _field(info.get("uploaded_at") or 0, "file", 1.0 if info.get("uploaded_at") else 0.0),
        "sensitivity": _field(info.get("sensitivity") or "", "file", 1.0 if info.get("sensitivity") else 0.0),
        "owner": _field(info.get("owner") or "", "file", 1.0 if info.get("owner") else 0.0),
    }
    return {
        "profile_version": PROFILE_VERSION,
        "doc_id": index.doc_id,
        "filename": index.filename,
        "document": doc,
        "sections": [{"title": t, "page": p} for t, p in (meta.get("sections") or [])][:300],
        "section_pages": sorted(set(sec_pages))[:300],
        "counts": {"text": stats["text"], "tables": stats["table"], "rows": stats["row_pin"],
                   "equations": stats["equation"], "figures": stats["figure"], "chunks": stats["total"]},
        "parameters": params,
        "part_numbers": parts,
        "standards": standards,
        "references": _refs(index),
        "ocr_pages": list(meta.get("ocr_pages") or []),
        "scanned_pages": list(meta.get("scanned_pages") or []),
        "suggested_tags": suggest_tags(index, category["value"], parts, params),
    }


# ------------------------------------------------------------------ effective view (auto + user)
USER_FIELDS = ("title", "author", "category", "tags", "description", "version_label", "part_numbers", "custom")


def effective(profile: Dict[str, Any], user: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Flat, display/filter-ready view. User-entered values win and are marked source="user"."""
    user = user or {}
    doc = profile.get("document", {})

    def pick(name: str, auto_value: Any, auto_src: str, conf: float) -> Dict[str, Any]:
        if user.get(name) not in (None, "", []):
            return {"value": user[name], "source": "user", "confidence": 1.0}
        return {"value": auto_value, "source": auto_src, "confidence": conf}

    t = doc.get("title", {})
    a = doc.get("author", {})
    c = doc.get("category", {})
    v = doc.get("version", {})
    auto_parts = [p["value"] for p in profile.get("part_numbers", []) if p.get("confidence", 0) >= 0.7]
    return {
        "title": pick("title", t.get("value", ""), t.get("source", "auto"), t.get("confidence", 0.0)),
        "author": pick("author", a.get("value", ""), a.get("source", "auto"), a.get("confidence", 0.0)),
        "category": pick("category", c.get("value", "other"), "auto", c.get("confidence", 0.0)),
        "tags": pick("tags", profile.get("suggested_tags", []), "auto", 0.6),
        "version_label": pick("version_label", v.get("value", ""), "auto", v.get("confidence", 0.0)),
        "part_numbers": pick("part_numbers", auto_parts, "auto", 0.7),
        "description": pick("description", "", "auto", 0.0),
        "custom": {"value": dict(user.get("custom") or {}), "source": "user", "confidence": 1.0},
        "file_type": doc.get("file_type", {}).get("value", ""),
        "pages": doc.get("pages", {}).get("value", 0),
    }


def search_blob(profile: Dict[str, Any], user: Optional[Dict[str, Any]]) -> str:
    """Lower-cased text used by the library's metadata search."""
    e = effective(profile, user)
    parts: List[str] = [profile.get("filename", "")]
    for k in ("title", "author", "category", "version_label", "description"):
        parts.append(str(e[k]["value"] or ""))
    parts += [str(x) for x in e["tags"]["value"] or []] + [str(x) for x in e["part_numbers"]["value"] or []]
    parts += [f"{k} {v}" for k, v in (e["custom"]["value"] or {}).items()]
    parts += [p["name"] for p in profile.get("parameters", [])[:100]] + profile.get("standards", [])
    return " ".join(parts).lower()


def matches(profile: Dict[str, Any], user: Optional[Dict[str, Any]], *, query: str = "", tags: Iterable[str] = (),
            category: str = "", part_number: str = "", file_type: str = "") -> bool:
    e = effective(profile, user)
    if query and not all(w in search_blob(profile, user) for w in query.lower().split()):
        return False
    want = {t.lower() for t in tags if t}
    if want and not want <= {str(t).lower() for t in e["tags"]["value"] or []}:
        return False
    if category and str(e["category"]["value"]).lower() != category.lower():
        return False
    if part_number:
        pn = part_number.upper()
        all_parts = {str(p).upper() for p in e["part_numbers"]["value"] or []} | \
                    {p["value"].upper() for p in profile.get("part_numbers", [])}
        if not any(pn in p for p in all_parts):
            return False
    if file_type and str(e["file_type"]).lower() != file_type.lower():
        return False
    return True
