"""
Upload validation: size, extension allow-list, content sniffing (magic bytes), and
container checks for Office files. Nothing is parsed before this passes.

Rejected:
  * empty or oversized files (MAX_UPLOAD_MB);
  * extensions not in ALLOWED_EXTENSIONS;
  * content that does not match its extension (an .exe renamed to .pdf, HTML in a .png, ...);
  * executables and scripts (MZ, ELF, Mach-O, shebang) whatever their name;
  * Office files with macros (vbaProject.bin), encrypted Office files, zip bombs
    (total uncompressed size > MAX_UNZIPPED_MB or ratio > MAX_RATIO), path-traversal entries;
  * text files with binary content (NUL bytes) or that are not valid UTF-8 / Latin-1.

The checks are deterministic and fast. They reduce, not eliminate, the risk of
malicious files: parsers (PyMuPDF, python-docx, openpyxl, tesseract) still run on
untrusted input, so ingestion runs in the application container with no secrets
beyond what the app needs (see docs/SECURITY_AUDIT.md).
"""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath

from config.settings import Settings
from rag.pdf_parser import PdfProcessingError

MAX_UNZIPPED_MB = 300
MAX_RATIO = 120
MAX_ZIP_ENTRIES = 5000

MIME = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "csv": "text/csv", "txt": "text/plain", "md": "text/markdown", "markdown": "text/markdown",
    "html": "text/html", "htm": "text/html", "json": "application/json",
    "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
}
TEXT_KINDS = {"csv", "txt", "md", "markdown", "html", "htm", "json"}
_OOXML_DIR = {"docx": "word/", "pptx": "ppt/", "xlsx": "xl/"}
_EXEC_MAGIC = (b"MZ", b"\x7fELF", b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xce", b"\xca\xfe\xba\xbe", b"#!")


class IngestionError(PdfProcessingError):
    """Validation or parsing failure; `user_message` is safe to show. `permanent` = retrying cannot help."""

    def __init__(self, user_message: str, permanent: bool = True):
        super().__init__(user_message)
        self.permanent = permanent


@dataclass(frozen=True)
class FileInfo:
    kind: str       # normalised extension: pdf, docx, ..., png
    mime: str
    size_bytes: int
    text: str = ""  # decoded content for text kinds


def extension(filename: str) -> str:
    ext = PurePosixPath((filename or "").replace("\\", "/")).suffix.lower().lstrip(".")
    return "jpg" if ext == "jpeg" else ("md" if ext == "markdown" else ("html" if ext == "htm" else ext))


def _decode_text(data: bytes) -> str:
    if b"\x00" in data[:8192]:
        raise IngestionError("This text file contains binary data.")
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    text = data.decode("latin-1")
    control = sum(1 for ch in text[:5000] if ord(ch) < 32 and ch not in "\r\n\t\f")
    if control > 20:
        raise IngestionError("This file is not readable text.")
    return text


def _check_ooxml(data: bytes, kind: str) -> None:
    if data[:4] != b"PK\x03\x04":
        if data[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":  # OLE container: legacy .doc or encrypted OOXML
            raise IngestionError("Encrypted or legacy (.doc/.xls/.ppt) Office files are not supported. "
                                 "Save the file as an unprotected .docx/.xlsx/.pptx.")
        raise IngestionError(f"This is not a valid .{kind} file.")
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
        infos = zf.infolist()
    except zipfile.BadZipFile as exc:
        raise IngestionError(f"This .{kind} file is damaged.") from exc
    names = [i.filename for i in infos]
    if len(infos) > MAX_ZIP_ENTRIES:
        raise IngestionError("This file contains too many parts.")
    total = sum(i.file_size for i in infos)
    packed = max(1, sum(i.compress_size for i in infos))
    if total > MAX_UNZIPPED_MB * 1024 * 1024 or total / packed > MAX_RATIO:
        raise IngestionError("This file expands to an unsafe size (possible zip bomb).")
    if any(n.startswith("/") or ".." in PurePosixPath(n).parts for n in names):
        raise IngestionError("This file contains unsafe paths.")
    if "[Content_Types].xml" not in names or not any(n.startswith(_OOXML_DIR[kind]) for n in names):
        raise IngestionError(f"The content of this file does not match the .{kind} extension.")
    if any(n.lower().endswith("vbaproject.bin") for n in names):
        raise IngestionError("Macro-enabled Office files are not accepted.")


def validate_upload(data: bytes, filename: str, settings: Settings) -> FileInfo:
    """Raise IngestionError if the upload must be rejected; return what it really is otherwise."""
    if not data:
        raise IngestionError("The uploaded file is empty.")
    size_mb = len(data) / (1024 * 1024)
    if size_mb > settings.max_upload_mb:
        raise IngestionError(f"This file is {size_mb:.1f} MB. The limit is {settings.max_upload_mb} MB.")
    kind = extension(filename)
    allowed = {extension("x." + e) for e in settings.allowed_extensions}
    if kind not in allowed or kind not in MIME:
        raise IngestionError(f"Files of type '.{kind or '?'}' are not accepted. Allowed: "
                             + ", ".join(sorted(allowed)))
    if data.startswith(_EXEC_MAGIC):
        raise IngestionError("Executable files are not accepted.")
    text = ""
    if kind == "pdf":
        if b"%PDF-" not in data[:1024]:
            raise IngestionError("Unable to process this PDF.\nPlease verify that the file is a valid PDF.")
    elif kind in _OOXML_DIR:
        _check_ooxml(data, kind)
    elif kind == "png":
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise IngestionError("This is not a valid PNG image.")
    elif kind == "jpg":
        if not data.startswith(b"\xff\xd8\xff"):
            raise IngestionError("This is not a valid JPEG image.")
    else:  # text kinds
        if data[:5] == b"%PDF-" or data[:4] == b"PK\x03\x04" or data[:4] == b"\x89PNG":
            raise IngestionError(f"The content of this file does not match the .{kind} extension.")
        text = _decode_text(data)
        if kind == "json":
            try:
                json.loads(text)
            except ValueError as exc:
                raise IngestionError("This JSON file is not valid JSON.") from exc
    return FileInfo(kind=kind, mime=MIME[kind], size_bytes=len(data), text=text)
