"""
Indexing: PDF -> DocumentIndex (parsed once, then cached).

    get_index(pdf_bytes, filename, settings, progress, vision)   upload -> index
    open_index(doc_id, filename, settings, vision)               reopen from cache (no PDF needed)

Stages of a new document (progress 0 -> 1):
    0.00-0.60  parse: text chunks, tables (+ rows/pins), equations, figures, metadata
    0.60-0.75  vision (optional): OCR of scanned pages, figure descriptions
    0.75-1.00  embeddings (Qwen3-Embedding via OpenRouter, or local fastembed)

Caching layers, fastest first:
  1. in-process LRU (`_registry`) shared by all browser sessions;
  2. disk cache under data/cache/<doc_id>/ (survives restarts, synced to HF);
  3. full parse + embed: only for a new PDF or new chunking/embedding settings.
A lock makes sure only one PDF is indexed at a time, which keeps memory use
predictable on a small VM and stops two users indexing the same file twice.
"""
from __future__ import annotations

import re
import threading
from collections import OrderedDict
from typing import Callable, List, Optional, Tuple

import numpy as np

from config.settings import Settings, get_logger
from ingestion.chunker import chunk_page
from rag.embeddings import EmbeddingError, active_model, embedding_signature, fill_missing_embeddings
from rag.equation_extractor import extract_equations
from rag.models import DocumentIndex, Node
from ingestion.processor import parse_document
from ingestion.validation import validate_upload
from storage import feedback as feedback_store
from storage import vector_store
from storage.cache import (
    DocumentPaths,
    VisionCache,
    clear_document_cache,
    compute_doc_id,
    read_json,
    sanitize_filename,
    save_document_meta,
)

log = get_logger("pipeline")

ProgressFn = Optional[Callable[[float, str], None]]

_REGISTRY_SIZE = 8
_registry: "OrderedDict[Tuple[str, str, str], DocumentIndex]" = OrderedDict()
_registry_lock = threading.Lock()
_build_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------
def _registry_get(key) -> Optional[DocumentIndex]:
    with _registry_lock:
        index = _registry.get(key)
        if index is not None:
            _registry.move_to_end(key)
        return index


def _registry_put(key, index: DocumentIndex) -> None:
    with _registry_lock:
        _registry[key] = index
        _registry.move_to_end(key)
        while len(_registry) > _REGISTRY_SIZE:
            _registry.popitem(last=False)


def _scaled(progress: ProgressFn, lo: float, hi: float) -> ProgressFn:
    return (lambda f, m: progress(lo + f * (hi - lo), m)) if progress else None


def _ocr_nodes(pages: dict, settings: Settings) -> List[Node]:
    nodes: List[Node] = []
    for page_num, text in sorted(pages.items()):
        chunks = chunk_page(text, page_num, settings)
        for n in chunks.nodes + extract_equations(text, page_num):
            n.meta = {**(n.meta or {}), "ocr": True}
            nodes.append(n)
    return nodes


def _apply_vision(index: DocumentIndex, pdf_bytes: Optional[bytes], settings: Settings, vision,
                  paths: DocumentPaths, p_key: str, e_key: str, progress: ProgressFn) -> None:
    """OCR pending scanned pages (needs the PDF) and describe undescribed figures (needs only the images)."""
    if vision is None:
        return
    meta = index.doc_meta
    changed = False
    done_pages = set(meta.get("ocr_pages") or [])
    pending = [p for p in (meta.get("scanned_pages") or []) if p not in done_pages]
    if pdf_bytes and pending and vision.ocr_enabled:
        texts = vision.ocr_pages(pdf_bytes, pending, _scaled(progress, 0.0, 0.6))
        new = _ocr_nodes(texts, settings)
        if new:
            n_old = len(index.nodes)
            index.nodes.extend(new)
            if index.embeddings.ndim == 2 and index.embeddings.shape[0] == n_old and index.embeddings.size:
                index.embeddings = np.vstack([index.embeddings,
                                              np.zeros((len(new), index.embeddings.shape[1]), np.float32)])
            index.has_embedding = np.concatenate([index.has_embedding[:n_old], np.zeros(len(new), bool)])
            for attr in ("lower_content", "indices_by_type"):  # derived lookups must see the new nodes
                index.__dict__.pop(attr, None)
            if index.embeddings.ndim == 2 and index.embeddings.shape[0] == len(index.nodes):
                vector_store.save_embeddings(paths, e_key, index.embeddings, index.has_embedding)
        meta["ocr_pages"] = sorted(done_pages | set(texts))
        meta["scanned_pages"] = [p for p in meta.get("scanned_pages", []) if p not in texts]
        changed = True
    if index.figures and vision.figures_enabled:
        added = vision.describe_figures(index.figures, index.figures_dir, VisionCache(paths.vision_file),
                                        _scaled(progress, 0.6, 1.0))
        changed = changed or added > 0
    if changed:
        vector_store.save_nodes(paths, p_key, index.nodes, index.figures, index.page_count, meta)


def _load_or_build(pdf_bytes: Optional[bytes], filename: str, settings: Settings, paths: DocumentPaths,
                   p_key: str, e_key: str, progress: ProgressFn, vision=None) -> Optional[DocumentIndex]:
    cached = vector_store.load_nodes(paths, p_key)
    if cached is not None:
        nodes, figures, page_count, doc_meta = cached
        log.info("Loaded index for %s from disk cache", paths.doc_id)
    elif pdf_bytes is None:
        return None  # nothing cached and no PDF to rebuild from
    else:
        paths.ensure()
        parsed = parse_document(pdf_bytes, filename, settings, paths.figures_dir,
                                progress=_scaled(progress, 0.0, 0.6))  # any supported format
        nodes, figures, page_count, doc_meta = parsed.nodes, parsed.figures, parsed.page_count, parsed.doc_meta
        vector_store.save_nodes(paths, p_key, nodes, figures, page_count, doc_meta)
        save_document_meta(paths, filename, page_count)

    meta = read_json(paths.meta_file, {}) or {}
    index = DocumentIndex(
        doc_id=paths.doc_id,
        filename=meta.get("filename", filename),
        page_count=page_count,
        nodes=nodes,
        figures=figures,
        figures_dir=paths.figures_dir,
        embed_model=active_model(settings),
        doc_meta=doc_meta,
        has_embedding=np.zeros(len(nodes), dtype=bool),
    )
    loaded = vector_store.load_embeddings(paths, e_key, len(nodes))
    if loaded is not None:
        index.embeddings, index.has_embedding = loaded
    try:
        _apply_vision(index, pdf_bytes, settings, vision, paths, p_key, e_key, _scaled(progress, 0.6, 0.75))
    except Exception:  # vision is an enrichment: never fail indexing because of it
        log.exception("Vision enrichment failed")
    return index


def _keys(settings: Settings) -> Tuple[str, str]:
    p_key = vector_store.parse_key(settings)
    return p_key, vector_store.embed_key(p_key, embedding_signature(settings))


def _embed(index: DocumentIndex, settings: Settings, paths: DocumentPaths, e_key: str,
           progress: ProgressFn) -> str:
    """Fill missing vectors; returns a user-facing reason if some are still missing."""
    try:
        with _build_lock:
            if fill_missing_embeddings(index, settings, progress):
                vector_store.save_embeddings(paths, e_key, index.embeddings, index.has_embedding)
    except EmbeddingError as exc:
        return exc.user_message
    return ""


def get_index(
    pdf_bytes: bytes,
    filename: str,
    settings: Settings,
    progress: ProgressFn = None,
    vision=None,
) -> Tuple[DocumentIndex, List[str]]:
    """
    Return the index for this PDF plus user-facing warnings.
    Accepts every format of ingestion/validation.py (PDF, Office, text, HTML, JSON, images).
    Raises PdfProcessingError / IngestionError (with a friendly message) for bad files.
    `vision` is an optional rag.vision.VisionHelper (OCR + figure descriptions).
    """
    validate_upload(pdf_bytes, filename, settings)
    filename = sanitize_filename(filename)
    doc_id = compute_doc_id(pdf_bytes)
    paths = DocumentPaths(settings.data_dir, doc_id)
    p_key, e_key = _keys(settings)
    key = (doc_id, p_key, e_key)

    index = _registry_get(key)
    if index is None:
        with _build_lock:
            index = _registry_get(key)
            if index is None:
                log.info("Document hash: %s (%s)", doc_id, filename)
                index = _load_or_build(pdf_bytes, filename, settings, paths, p_key, e_key, progress, vision)
                _registry_put(key, index)

    warnings: List[str] = []
    if index.missing_embeddings:
        reason = _embed(index, settings, paths, e_key, _scaled(progress, 0.75, 1.0))
        if index.missing_embeddings:
            warnings.append("Semantic search is off for part of this document, so answers use "
                            f"keyword matching only.\n\n{reason}".strip())
    scanned = index.doc_meta.get("scanned_pages") or []
    if scanned:
        warnings.append(f"{len(scanned)} image-only page(s) could not be read ({scanned[:8]}). "
                        "Turn on 'OCR scanned pages' with a vision model to include them.")
    if progress:
        progress(1.0, "Ready")
    return index, warnings


def open_index(doc_id: str, filename: str, settings: Settings, vision=None) -> Optional[DocumentIndex]:
    """Re-open an already indexed document from its cache (no PDF needed).
    Returns None if the cache is missing or was built with different chunk settings."""
    try:
        paths = DocumentPaths(settings.data_dir, doc_id)
    except ValueError:
        return None
    p_key, e_key = _keys(settings)
    key = (doc_id, p_key, e_key)
    index = _registry_get(key)
    if index is None:
        with _build_lock:
            index = _registry_get(key) or _load_or_build(None, sanitize_filename(filename), settings, paths,
                                                         p_key, e_key, None, vision)
            if index is not None:
                _registry_put(key, index)
    if index is not None and index.missing_embeddings:
        _embed(index, settings, paths, e_key, None)  # keyword search still works if this fails
    return index


def clear_cache(doc_id: str, settings: Settings) -> None:
    with _registry_lock:
        for key in [k for k in _registry if k[0] == doc_id]:
            del _registry[key]
    clear_document_cache(DocumentPaths(settings.data_dir, doc_id))


def document_paths(index: DocumentIndex, settings: Settings) -> DocumentPaths:
    return DocumentPaths(settings.data_dir, index.doc_id)


# ---------------------------------------------------------------------------
# Answer text helpers
# ---------------------------------------------------------------------------
_IMAGE_MD = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


def normalize_answer(text: str) -> str:
    """LaTeX delimiters Streamlit understands; drop remote-image markdown and stray <think> blocks."""
    text = _THINK.sub("", text or "").strip()
    text = text.replace("\\[", "$$").replace("\\]", "$$")
    text = text.replace("\\(", "$").replace("\\)", "$")
    return _IMAGE_MD.sub("", text)


def save_feedback(index: DocumentIndex, settings: Settings, question: str, answer: str) -> None:
    feedback_store.store_feedback(document_paths(index, settings), question, answer)
