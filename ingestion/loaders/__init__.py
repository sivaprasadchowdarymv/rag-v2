"""
Format loaders: bytes -> LoadedDocument (pages of text + tables + optional images to OCR).

PDF keeps its own specialised parser (rag/pdf_parser.py: tables, equations,
figures). Every other format goes through a loader here; the processor turns
LoadedDocument into the same Node types the PDF lane produces, so retrieval,
citations ("p3" = page / slide / sheet 3) and verification are format-agnostic.

    docx  python-docx   paragraphs (headings kept as Markdown "#"), tables; page = ~3,000 chars block
    pptx  python-pptx   one page per slide: title, text frames, tables, notes
    xlsx  openpyxl      one page per sheet; rows become table/row nodes (values, not formulas)
    csv   csv module    one table (first row = header), split into pages of 200 rows
    txt / md            plain text; Markdown headings drive structure-aware chunking
    html  html.parser   visible text only (script/style/head dropped), <h1-6> -> Markdown headings, tables
    json  json module   flattened "path.to.key: value" lines
    png / jpg           local OCR (tesseract)
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Callable, Dict, List

from config.settings import get_logger

log = get_logger("loaders")

PAGE_CHARS = 3000
CSV_ROWS_PER_PAGE = 200
MAX_TABLE_ROWS = 2000


@dataclass
class Table:
    headers: List[str]
    rows: List[List[str]]
    caption: str = ""


@dataclass
class Page:
    number: int
    text: str = ""
    tables: List[Table] = field(default_factory=list)
    title: str = ""
    images: List[bytes] = field(default_factory=list)  # to OCR


@dataclass
class LoadedDocument:
    pages: List[Page]
    meta: Dict[str, Any] = field(default_factory=dict)


def _clean(v: Any) -> str:
    return " ".join(str("" if v is None else v).split())


def _paginate(blocks: List[str], tables_at: Dict[int, List[Table]]) -> List[Page]:
    """Group text blocks into ~PAGE_CHARS pages; tables stay on the page where they appear."""
    pages, cur, size = [Page(1)], [], 0
    for i, block in enumerate(blocks):
        if size + len(block) > PAGE_CHARS and cur:
            pages[-1].text = "\n".join(cur)
            pages.append(Page(len(pages) + 1))
            cur, size = [], 0
        cur.append(block)
        size += len(block)
        pages[-1].tables.extend(tables_at.get(i, []))
    pages[-1].text = "\n".join(cur)
    return pages


# ------------------------------------------------------------------- Office
def load_docx(data: bytes) -> LoadedDocument:
    import docx
    from docx.table import Table as DocxTable
    from docx.text.paragraph import Paragraph

    d = docx.Document(io.BytesIO(data))
    blocks: List[str] = []
    tables_at: Dict[int, List[Table]] = {}
    for child in d.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(child, d)
            text = p.text.strip()
            if not text:
                continue
            style = (p.style.name or "").lower() if p.style is not None else ""
            if style.startswith("heading") or style == "title":
                level = int(style.split()[-1]) if style.split()[-1].isdigit() else 1
                text = "#" * min(level, 6) + " " + text
            blocks.append(text)
        elif tag == "tbl":
            t = DocxTable(child, d)
            grid = [[_clean(c.text) for c in r.cells] for r in t.rows]
            grid = [r for r in grid if any(r)]
            if grid:
                tables_at.setdefault(max(len(blocks) - 1, 0), []).append(Table(grid[0], grid[1:MAX_TABLE_ROWS]))
                blocks.append("\n".join(" | ".join(r) for r in grid))
    cp = d.core_properties
    return LoadedDocument(_paginate(blocks or [""], tables_at),
                          {"title": cp.title or "", "author": cp.author or "", "subject": cp.subject or ""})


def load_pptx(data: bytes) -> LoadedDocument:
    from pptx import Presentation

    prs = Presentation(io.BytesIO(data))
    pages: List[Page] = []
    for n, slide in enumerate(prs.slides, start=1):
        page = Page(n)
        lines: List[str] = []
        title_shape = slide.shapes.title
        if title_shape is not None and title_shape.has_text_frame and title_shape.text_frame.text.strip():
            page.title = _clean(title_shape.text_frame.text)
            lines.append(f"# {page.title}")
        for shape in slide.shapes:
            if shape is title_shape:
                continue
            if getattr(shape, "has_table", False) and shape.has_table:
                grid = [[_clean(c.text) for c in r.cells] for r in shape.table.rows]
                grid = [r for r in grid if any(r)]
                if grid:
                    page.tables.append(Table(grid[0], grid[1:MAX_TABLE_ROWS], page.title))
            elif getattr(shape, "has_text_frame", False) and shape.has_text_frame:
                text = shape.text_frame.text.strip()
                if text:
                    lines.append(text)
            elif shape.shape_type == 13 and hasattr(shape, "image"):  # picture: OCR later
                page.images.append(shape.image.blob)
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                lines.append(f"Speaker notes: {notes}")
        page.text = "\n".join(lines)
        pages.append(page)
    cp = prs.core_properties
    return LoadedDocument(pages or [Page(1)], {"title": cp.title or "", "author": cp.author or ""})


def load_xlsx(data: bytes) -> LoadedDocument:
    import openpyxl

    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    pages: List[Page] = []
    try:
        for n, ws in enumerate(wb.worksheets, start=1):
            rows = []
            for row in ws.iter_rows(values_only=True):
                cells = [_clean(v) for v in row]
                if any(cells):
                    rows.append(cells)
                if len(rows) > MAX_TABLE_ROWS:
                    break
            width = max((len(r) for r in rows), default=0)
            rows = [r + [""] * (width - len(r)) for r in rows]
            page = Page(n, title=ws.title, text=f"# Sheet: {ws.title}")
            if rows:
                page.tables.append(Table(rows[0], rows[1:], f"Sheet {ws.title}"))
            pages.append(page)
    finally:
        wb.close()
    return LoadedDocument(pages or [Page(1)], {"title": "", "sheets": [p.title for p in pages]})


# --------------------------------------------------------------------- text
def load_text(text: str) -> LoadedDocument:
    blocks = [b for b in text.replace("\r\n", "\n").split("\n\n")]
    return LoadedDocument(_paginate([b for b in blocks if b.strip()] or [""], {}), {})


def load_csv(text: str) -> LoadedDocument:
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    rows = [[_clean(c) for c in r] for r in csv.reader(io.StringIO(text), dialect) if any(c.strip() for c in r)]
    rows = rows[:MAX_TABLE_ROWS + 1]
    if not rows:
        return LoadedDocument([Page(1)], {})
    header, body = rows[0], rows[1:]
    pages = []
    for start in range(0, max(len(body), 1), CSV_ROWS_PER_PAGE):
        chunk = body[start:start + CSV_ROWS_PER_PAGE]
        pages.append(Page(len(pages) + 1, text=f"Columns: {', '.join(header)}", tables=[Table(header, chunk)]))
    return LoadedDocument(pages, {"columns": header, "rows": len(body)})


def _flatten(obj: Any, path: str, out: List[str], limit: int = 20000) -> None:
    if len(out) >= limit:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(v, f"{path}.{k}" if path else str(k), out, limit)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _flatten(v, f"{path}[{i}]", out, limit)
    else:
        out.append(f"{path or 'value'}: {obj}")


def load_json(text: str) -> LoadedDocument:
    lines: List[str] = []
    _flatten(json.loads(text), "", lines)
    return LoadedDocument(_paginate(lines or [""], {}), {})


class _HTMLText(HTMLParser):
    _SKIP = {"script", "style", "head", "noscript", "template", "svg", "iframe"}
    _BLOCK = {"p", "div", "br", "li", "tr", "section", "article", "header", "footer", "pre", "blockquote"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: List[str] = []
        self.skip = 0
        self.heading = 0
        self.tables: List[List[List[str]]] = []
        self.row: List[str] = []
        self.cell: List[str] = []
        self.in_cell = False
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self.skip += 1
            self._in_title = tag == "head" and False
        if tag == "title":
            self._in_title = True
        if self.skip:
            return
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.heading = int(tag[1])
            self.out.append("\n" + "#" * self.heading + " ")
        elif tag == "table":
            self.tables.append([])
        elif tag == "tr":
            self.row = []
        elif tag in ("td", "th"):
            self.cell, self.in_cell = [], True
        elif tag in self._BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in self._SKIP:
            self.skip = max(0, self.skip - 1)
            return
        if self.skip:
            return
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.heading = 0
            self.out.append("\n")
        elif tag in ("td", "th"):
            self.row.append(_clean("".join(self.cell)))
            self.in_cell = False
        elif tag == "tr" and self.tables and any(self.row):
            self.tables[-1].append(self.row)

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        if self.skip:
            return
        if self.in_cell:
            self.cell.append(data)
        self.out.append(data)


def load_html(text: str) -> LoadedDocument:
    parser = _HTMLText()
    parser.feed(text)
    parser.close()
    body = "\n".join(line.strip() for line in "".join(parser.out).split("\n"))
    while "\n\n\n" in body:
        body = body.replace("\n\n\n", "\n\n")
    doc = load_text(body)
    tables = [Table(t[0], t[1:MAX_TABLE_ROWS]) for t in parser.tables if t]
    if tables:
        doc.pages[0].tables.extend(tables)
    doc.meta["title"] = _clean(parser.title)
    return doc


def load_image(data: bytes) -> LoadedDocument:
    return LoadedDocument([Page(1, images=[data])], {})


LOADERS: Dict[str, Callable[..., LoadedDocument]] = {
    "docx": load_docx, "pptx": load_pptx, "xlsx": load_xlsx, "png": load_image, "jpg": load_image,
}
TEXT_LOADERS: Dict[str, Callable[[str], LoadedDocument]] = {
    "txt": load_text, "md": load_text, "csv": load_csv, "json": load_json, "html": load_html,
}
