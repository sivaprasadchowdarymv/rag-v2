"""
Controlled benchmark corpus, generated deterministically (no downloads): a 4-page regulator datasheet
(text, tables, equations, figures), a Word manual, an Excel inventory and a Markdown guide.

    python -m evaluation.corpus data/eval_corpus     # writes the files
"""
from __future__ import annotations

import io
import struct
import sys
import zlib
from pathlib import Path
from typing import Dict


def _png(w: int, h: int) -> bytes:
    rows = b"".join(b"\x00" + bytes([255, 255, 255]) * w for _ in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def datasheet() -> bytes:
    import pymupdf

    doc = pymupdf.open()

    def lines(page, y, items, size=10):
        for item in items:
            page.insert_text((50, y), item, fontsize=size)
            y += size + 5
        return y

    def table(page, x, y, rows, colw=120, rowh=18):
        for r, row in enumerate(rows):
            for c, cell in enumerate(row):
                rect = pymupdf.Rect(x + c * colw, y + r * rowh, x + (c + 1) * colw, y + (r + 1) * rowh)
                page.draw_rect(rect, color=(0, 0, 0), width=0.8)
                page.insert_text((rect.x0 + 3, rect.y1 - 5), cell, fontsize=9)
        return y + len(rows) * rowh + 10

    p = doc.new_page()
    lines(p, 60, ["LM7805X VOLTAGE REGULATOR", "", "FEATURES", "Output current up to 1.5 A",
                  "Thermal overload protection", "Output voltage VOUT = 5.0V", "", "Absolute Maximum Ratings",
                  "Input voltage VIN = 35V", "Operating temperature TA = 125°C"])
    p = doc.new_page()
    y = lines(p, 60, ["ELECTRICAL CHARACTERISTICS", "Conditions: VIN = 10V, IO = 500mA, TJ = 25°C"])
    y = table(p, 50, y + 10, [["Parameter", "Min", "Typ", "Max"], ["Output voltage", "4.8", "5.0", "5.2"],
                               ["Quiescent current", "", "5", "8 mA"], ["Peak output current", "", "2.2", "A"]])
    lines(p, y + 20, ["Dropout voltage VDO = 2.0V"])
    p = doc.new_page()
    y = lines(p, 60, ["POWER DISSIPATION", "The power dissipated in the regulator is", "PD = (VIN - VOUT) x IOUT",
                      "where PD is the power dissipation in watts, VIN the input voltage",
                      "and IOUT the output current.", "Junction temperature TJ = TA + PD x RthJA"])
    p.insert_image(pymupdf.Rect(50, y + 10, 300, y + 160), stream=_png(200, 120))
    lines(p, y + 175, ["Figure 1. Output voltage vs input voltage"])
    p = doc.new_page()
    y = lines(p, 60, ["PIN CONFIGURATION"])
    y = table(p, 50, y + 10, [["Pin", "Name", "Function"], ["1", "INPUT", "Unregulated input"],
                               ["2", "GND", "Ground"], ["3", "OUTPUT", "Regulated output"]])
    p.insert_image(pymupdf.Rect(50, y + 10, 300, y + 160), stream=_png(180, 110))
    lines(p, y + 175, ["Figure 2. Functional block diagram"])
    data = doc.tobytes()
    doc.close()
    return data


def manual_docx() -> bytes:
    import docx

    d = docx.Document()
    d.core_properties.title = "Pump Controller Manual"
    d.add_heading("Electrical Ratings", 1)
    d.add_paragraph("The pump controller accepts a supply voltage of 24 V DC. Maximum motor current is 8 A.")
    t = d.add_table(rows=3, cols=3)
    for r, row in enumerate([["Parameter", "Value", "Unit"], ["Supply voltage", "24", "V"], ["Standby power", "1.5", "W"]]):
        for c, val in enumerate(row):
            t.cell(r, c).text = val
    d.add_heading("Maintenance", 1)
    d.add_paragraph("Replace the inlet filter every 500 operating hours. Lubricate the bearings once a year.")
    d.add_heading("Error Codes", 1)
    d.add_paragraph("Error E04 indicates dry running: the pump stops after 30 seconds without water flow.")
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def inventory_xlsx() -> bytes:
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Inventory"
    for row in [["Item", "Stock", "Location"], ["Bearing 6204", 140, "Aisle 3"], ["Seal kit", 22, "Aisle 7"],
                ["Impeller 90 mm", 9, "Aisle 2"]]:
        ws.append(row)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


GUIDE_MD = ("# Installation Guide\n\nRun the installer as administrator.\n\n## Network\n\n"
            "Open TCP port 8443 for the controller API. The default gateway timeout is 90 seconds.\n\n"
            "## Firmware\n\nFirmware updates are signed; version 4.1.2 is the minimum supported release.\n")


def files() -> Dict[str, bytes]:
    return {"LM7805X.pdf": datasheet(), "pump_manual.docx": manual_docx(), "inventory.xlsx": inventory_xlsx(),
            "install_guide.md": GUIDE_MD.encode()}


def write(target: Path) -> Dict[str, Path]:
    target.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, data in files().items():
        (target / name).write_bytes(data)
        out[name] = target / name
    return out


if __name__ == "__main__":  # pragma: no cover
    print(write(Path(sys.argv[1] if len(sys.argv) > 1 else "data/eval_corpus")))
