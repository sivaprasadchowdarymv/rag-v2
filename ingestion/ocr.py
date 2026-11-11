"""
Local OCR with Tesseract (offline). Used for PNG/JPG uploads, pictures in slides
and image-only (scanned) PDF pages. The cloud vision OCR (rag/vision.py) remains
available as a second pass for pages tesseract cannot read.
"""
from __future__ import annotations

import io
import shutil
from functools import lru_cache
from typing import Optional

from config.settings import get_logger

log = get_logger("ocr")

MIN_CHARS = 15
MAX_PIXELS = 40_000_000  # refuse decompression-bomb images


@lru_cache(maxsize=1)
def available() -> bool:
    if shutil.which("tesseract") is None:
        return False
    try:
        import pytesseract  # noqa: F401
        from PIL import Image  # noqa: F401
        return True
    except ImportError:
        return False


def ocr_image(data: bytes, lang: str = "eng", timeout: int = 60) -> str:
    """Text found in an image ('' if OCR is unavailable or nothing readable was found)."""
    if not available():
        return ""
    import pytesseract
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(data))
        if img.width * img.height > MAX_PIXELS:
            log.warning("Image too large for OCR (%sx%s)", img.width, img.height)
            return ""
        img = img.convert("L")
        if max(img.size) < 1200:  # small images: upscale for better recognition
            scale = 1200 / max(img.size)
            img = img.resize((int(img.width * scale), int(img.height * scale)))
        text = pytesseract.image_to_string(img, lang=lang, timeout=timeout) or ""
    except Exception as exc:  # corrupt image, tesseract timeout
        log.warning("OCR failed: %s", type(exc).__name__)
        return ""
    text = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    return text if len(text) >= MIN_CHARS else ""


def ocr_pdf_page(pdf_bytes: bytes, page_number: int, dpi: int = 200) -> str:
    import pymupdf

    doc: Optional["pymupdf.Document"] = None
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        pix = doc[page_number - 1].get_pixmap(dpi=dpi)
        return ocr_image(pix.tobytes("png"))
    except Exception as exc:
        log.warning("Could not render page %s for OCR: %s", page_number, type(exc).__name__)
        return ""
    finally:
        if doc is not None:
            doc.close()
