"""Builders for non-PDF test documents (generated in memory, nothing downloaded)."""
from __future__ import annotations

import io
import json
import zipfile


def make_docx() -> bytes:
    import docx

    d = docx.Document()
    d.core_properties.title = "Pump Controller Manual"
    d.add_heading("Electrical Ratings", 1)
    d.add_paragraph("The pump controller accepts a supply voltage of 24 V DC. Maximum motor current is 8 A.")
    t = d.add_table(rows=3, cols=3)
    for r, row in enumerate([["Parameter", "Value", "Unit"], ["Supply voltage", "24", "V"],
                             ["Standby power", "1.5", "W"]]):
        for c, val in enumerate(row):
            t.cell(r, c).text = val
    d.add_heading("Maintenance", 1)
    d.add_paragraph("Replace the inlet filter every 500 operating hours.")
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def make_pptx() -> bytes:
    from pptx import Presentation
    from pptx.util import Inches

    prs = Presentation()
    s = prs.slides.add_slide(prs.slide_layouts[1])
    s.shapes.title.text = "Quarterly Results"
    s.placeholders[1].text = "Revenue grew 12 percent to 4.2 million."
    s2 = prs.slides.add_slide(prs.slide_layouts[5])
    s2.shapes.title.text = "Regional Breakdown"
    tbl = s2.shapes.add_table(3, 2, Inches(1), Inches(2), Inches(6), Inches(1.5)).table
    for r, row in enumerate([["Region", "Revenue"], ["North", "2.5"], ["South", "1.7"]]):
        for c, v in enumerate(row):
            tbl.cell(r, c).text = v
    s2.notes_slide.notes_text_frame.text = "South region missed target."
    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


def make_xlsx() -> bytes:
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Inventory"
    for row in [["Item", "Stock", "Location"], ["Bearing 6204", 140, "Aisle 3"], ["Seal kit", 22, "Aisle 7"]]:
        ws.append(row)
    ws2 = wb.create_sheet("Prices")
    ws2.append(["Item", "Price"])
    ws2.append(["Bearing 6204", 4.75])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


CSV = "sensor,threshold,unit\ntemperature,85,C\npressure,6.5,bar\n"
MARKDOWN = "# Install Guide\n\nRun the installer as administrator.\n\n## Network\n\nOpen port 8443 for the API.\n"
HTML = ("<html><head><title>Warranty</title><script>alert('x')</script></head><body><h1>Warranty terms</h1>"
        "<p>The warranty period is 36 months from delivery.</p><table><tr><th>Part</th><th>Months</th></tr>"
        "<tr><td>Motor</td><td>24</td></tr></table></body></html>")
JSON = json.dumps({"device": {"model": "TX-900", "firmware": "4.1.2", "limits": {"max_temp_c": 70}}})


def make_png_text(text: str = "INVOICE TOTAL 1250 EUR") -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (1400, 220), "white")
    ImageDraw.Draw(img).text((40, 60), text, fill="black", font=ImageFont.load_default(size=64))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_macro_docx() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(make_docx())) as src, zipfile.ZipFile(buf, "w") as dst:
        for item in src.infolist():
            dst.writestr(item, src.read(item.filename))
        dst.writestr("word/vbaProject.bin", b"\x00" * 64)
    return buf.getvalue()


def make_zip_bomb_docx() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/document.xml", b"0" * (200 * 1024 * 1024))
    return buf.getvalue()
