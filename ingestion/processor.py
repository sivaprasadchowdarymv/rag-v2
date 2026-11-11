"""
Document processor: validated bytes of ANY supported format -> ParsedPdf (nodes, figures, pages, metadata).

    parse_document(data, filename, settings, figures_dir, progress)

* PDF       -> rag/pdf_parser.parse_pdf (text, tables, equations, figures), then local OCR of image-only pages
* others    -> ingestion/loaders -> the same Node types:
                 text      chunked with CHUNK_STRATEGY (ingestion/chunker.py)
                 table/row table_to_nodes() — identical to PDF tables, so table search and citations work
                 equation  extract_equations() on the page text
                 figure    images (PNG/JPG uploads, slide pictures) saved as figures; OCR text kept as context
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, List, Optional

from config.settings import Settings, get_logger
from ingestion import ocr
from ingestion.chunker import chunk_page
from ingestion.loaders import LOADERS, TEXT_LOADERS, LoadedDocument, Page
from ingestion.validation import FileInfo, IngestionError, validate_upload
from rag.equation_extractor import extract_equations
from rag.models import Node
from rag.pdf_parser import ParsedPdf, parse_pdf
from rag.table_extractor import table_to_nodes

log = get_logger("processor")
ProgressFn = Optional[Callable[[float, str], None]]


def _table_dict(headers: List[str], rows: List[List[str]], caption: str) -> dict:
    header_line = " | ".join(headers)
    lines = [" | ".join(r) for r in rows]
    return {"headers": headers, "rows": lines, "header_line": header_line, "caption": caption,
            "content": header_line + "\n" + "\n".join(lines)}


def _save_image(data: bytes, figures_dir: Path) -> str:
    figures_dir.mkdir(parents=True, exist_ok=True)
    ext = "jpg" if data[:3] == b"\xff\xd8\xff" else "png"
    name = f"img_{hashlib.sha256(data).hexdigest()[:12]}.{ext}"
    path = figures_dir / name
    if not path.exists():
        path.write_bytes(data)
    return name


def _page_nodes(page: Page, settings: Settings, figures_dir: Path, figures: List[Node],
                ocr_pages: List[int], filename: str) -> List[Node]:
    nodes: List[Node] = []
    text = page.text or ""
    for k, img in enumerate(page.images):
        found = ocr.ocr_image(img) if settings.ocr_local else ""
        if found:
            text = f"{text}\n{found}" if text else found
            ocr_pages.append(page.number)
        try:
            fname = _save_image(img, figures_dir)
            caption = page.title or (filename if len(page.images) == 1 else f"{filename} image {k + 1}")
            figures.append(Node(type="figure", section="Figure", content=caption, raw_content=caption,
                                page=page.number, chunk_idx=len(figures), figure_file=fname, caption=caption,
                                meta={"context": found[:600], "ocr": bool(found)}))
        except OSError:
            log.warning("could not store an image of %s", filename)
    chunks = chunk_page(text, page.number, settings)
    for n in chunks.nodes:
        if page.number in ocr_pages:
            n.meta = {**(n.meta or {}), "ocr": True}
    nodes.extend(chunks.nodes)
    section = page.title or chunks.last_heading or "Table"
    for t in page.tables:
        if t.headers or t.rows:
            nodes.extend(table_to_nodes(_table_dict(t.headers, t.rows, t.caption or section), page.number, section))
    nodes.extend(extract_equations(text, page.number))
    return nodes


def _sections(nodes: List[Node]) -> List[list]:
    seen, out = set(), []
    for n in nodes:
        if n.type == "text" and n.section and n.section != "General" and n.section not in seen and len(out) < 300:
            seen.add(n.section)
            out.append([n.section, n.page])
    return out


def _parse_loaded(doc: LoadedDocument, info: FileInfo, settings: Settings, figures_dir: Path, filename: str,
                  progress: ProgressFn) -> ParsedPdf:
    if len(doc.pages) > settings.max_pages:
        raise IngestionError(f"This file has {len(doc.pages)} pages/sheets/slides. The limit is {settings.max_pages}.")
    nodes: List[Node] = []
    figures: List[Node] = []
    ocr_pages: List[int] = []
    for i, page in enumerate(doc.pages):
        if progress:
            progress(i / max(len(doc.pages), 1), f"Reading part {i + 1} of {len(doc.pages)}")
        nodes.extend(_page_nodes(page, settings, figures_dir, figures, ocr_pages, filename))
    if not nodes and not figures:
        raise IngestionError("No readable text was found in this file." +
                             (" (OCR is not available on this server.)" if info.kind in ("png", "jpg")
                              and not ocr.available() else ""))
    first_text = next((n.raw_content.split("\n")[0].lstrip("# ").strip() for n in nodes if n.type == "text"), "")
    meta = {"title": (doc.meta.get("title") or "").strip() or first_text[:120] or filename,
            "author": doc.meta.get("author", ""), "subject": doc.meta.get("subject", ""),
            "sections": _sections(nodes), "scanned_pages": [], "ocr_pages": sorted(set(ocr_pages)),
            "format": info.kind, "mime": info.mime}
    meta.update({k: v for k, v in doc.meta.items() if k in ("sheets", "columns", "rows")})
    return ParsedPdf(nodes=nodes, figures=figures, page_count=len(doc.pages), doc_meta=meta)


def _local_ocr_pdf(parsed: ParsedPdf, data: bytes, settings: Settings) -> None:
    pending = list(parsed.doc_meta.get("scanned_pages") or [])
    if not (settings.ocr_local and pending and ocr.available()):
        return
    done = []
    for page_num in pending[: settings.max_ocr_pages]:
        text = ocr.ocr_pdf_page(data, page_num)
        if not text:
            continue
        for n in chunk_page(text, page_num, settings).nodes + extract_equations(text, page_num):
            n.meta = {**(n.meta or {}), "ocr": True}
            parsed.nodes.append(n)
        done.append(page_num)
    if done:
        meta = parsed.doc_meta
        meta["ocr_pages"] = sorted(set(meta.get("ocr_pages") or []) | set(done))
        meta["scanned_pages"] = [p for p in pending if p not in done]
        log.info("Local OCR read %d scanned page(s)", len(done))


def parse_document(data: bytes, filename: str, settings: Settings, figures_dir: Path,
                   progress: ProgressFn = None, info: Optional[FileInfo] = None) -> ParsedPdf:
    info = info or validate_upload(data, filename, settings)
    if info.kind == "pdf":
        parsed = parse_pdf(data, settings, figures_dir, progress=progress)
        parsed.doc_meta.setdefault("format", "pdf")
        _local_ocr_pdf(parsed, data, settings)
        return parsed
    try:
        if info.kind in TEXT_LOADERS:
            loaded = TEXT_LOADERS[info.kind](info.text)
        else:
            loaded = LOADERS[info.kind](data)
    except IngestionError:
        raise
    except Exception as exc:  # parser crash on a malformed file: never leak details to the user
        log.warning("%s loader failed: %s", info.kind, type(exc).__name__)
        raise IngestionError(f"Unable to read this .{info.kind} file. It may be damaged.") from exc
    return _parse_loaded(loaded, info, settings, figures_dir, filename, progress)
