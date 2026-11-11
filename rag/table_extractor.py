"""
Table lane: PyMuPDF find_tables() -> one full-table node + one row/pin node
per data row (header always injected so every row is self-contained).

Cell text is read by clipping page text to each cell (robust across PyMuPDF
versions); failures are logged instead of silently swallowed.
"""
from __future__ import annotations

from typing import Any, Dict, List

from config.settings import get_logger
from rag.models import Node

log = get_logger("tables")

_PIN_KEYWORDS = ("pin", "port", "signal", "gpio")


def _table_caption(page, table) -> str:
    """Nearest text block above the table (e.g. 'Table 3. Electrical characteristics')."""
    try:
        top = table.bbox[1]
        best, best_gap = "", 80.0
        for b in page.get_text("blocks") or []:
            text = str(b[4] or "").strip()
            gap = top - b[3]
            if text and 0 <= gap < best_gap:
                best, best_gap = text, gap
        return best.split("\n")[0][:150]
    except Exception:
        return ""


def _cell_text(page, cell) -> str:
    if cell is None:
        return ""
    import pymupdf

    return " ".join((page.get_text("text", clip=pymupdf.Rect(cell)) or "").split())


def _table_grid(page, table) -> List[List[str]]:
    """Cell text read by clipping the page text to each cell box.

    PyMuPDF >= 1.26 / 2.x `Table.extract()` / `to_pandas()` can drop spaces and
    split decimals ("Output voltage" -> "Outputvoltage", "4.8" -> "48\n."),
    which silently corrupts datasheet values. Clipped text keeps them intact."""
    grid = [[_cell_text(page, c) for c in row.cells] for row in table.rows]
    return [r for r in grid if any(x.strip() for x in r)]


def extract_tables(page) -> List[Dict[str, Any]]:
    tables: List[Dict[str, Any]] = []
    try:
        found = page.find_tables().tables
    except Exception as exc:  # find_tables can fail on malformed pages
        log.debug("find_tables failed on page %s: %s", page.number + 1, exc)
        return tables

    for table in found:
        try:
            grid = _table_grid(page, table)
            external = bool(getattr(table.header, "external", False))
            if not grid:
                continue
            if external:  # header row sits above the table's box
                headers = [str(h or "") for h in table.header.names]
                body = grid
            else:
                headers, body = grid[0], grid[1:]
        except Exception as exc:
            log.debug("Could not read a table on page %s: %s", page.number + 1, exc)
            continue
        caption = _table_caption(page, table)
        header_line = " | ".join(headers)
        rows = [" | ".join(r) for r in body]
        tables.append(
            {
                "headers": headers,
                "content": header_line + "\n" + "\n".join(rows),
                "rows": rows,
                "header_line": header_line,
                "caption": caption,
            }
        )
    return tables


def table_to_nodes(tbl: Dict[str, Any], page_num: int, section_title: str) -> List[Node]:
    nodes: List[Node] = []
    header = tbl["header_line"]
    content = tbl["content"]

    nodes.append(  # full-table node (semantic table search)
        Node(
            type="table",
            section=section_title,
            content=content,
            raw_content=content,
            page=page_num,
            caption=tbl.get("caption") or None,
        )
    )

    row_type = "pin" if any(k in content.lower() for k in _PIN_KEYWORDS) else "row"
    for idx, row in enumerate(tbl["rows"]):
        if not row.strip():
            continue
        raw = f"{header}\n{row}"
        nodes.append(
            Node(
                type=row_type,
                section=section_title,
                content=raw,
                raw_content=raw,
                page=page_num,
                chunk_idx=idx,
            )
        )
    return nodes
