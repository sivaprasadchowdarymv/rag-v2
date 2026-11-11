"""
Vision at indexing time (Qwen 3.8 / Gemma 4 are natively multimodal).

    VisionHelper.ocr_pages(pdf_bytes, pages)    scanned / image-only pages -> Markdown text nodes
    VisionHelper.describe_figures(index)        every figure gets a searchable visual description

v1 could only flag scanned pages and read figure captions. Here:
  * OCR text is chunked exactly like normal page text (sections, overlap,
    equations), so scanned datasheets become fully searchable and citable.
  * Figure descriptions are stored on the figure (meta["vision"]) and in the
    document's vision cache, added to the figure search index, and given to
    the master agent as evidence ("FIG_3 ... Visual description: ...").
Both run in parallel (VISION_WORKERS), are capped per document
(MAX_OCR_PAGES / MAX_VISION_FIGURES), are cached, and never block indexing:
a failed page or figure is simply skipped.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from config.settings import Settings, get_logger
from rag.prompts import FIGURE_PROMPT, OCR_PROMPT

log = get_logger("vision")
ProgressFn = Optional[Callable[[float, str], None]]
OCR_MAX_PX = 2000


class VisionHelper:
    def __init__(self, router, settings: Settings):
        self.router, self.settings = router, settings
        self.cost = 0.0
        self.calls = 0
        self._lock = threading.Lock()

    @property
    def ocr_enabled(self) -> bool:
        return self.settings.ocr_scanned and self.settings.max_ocr_pages > 0

    @property
    def figures_enabled(self) -> bool:
        return self.settings.vision_index and self.settings.max_vision_figures > 0

    def _ask(self, prompt: str, image: bytes, max_tokens: int) -> str:
        from llm.providers import LLMError

        try:
            res = self.router.chat("vision", [{"role": "user", "content": prompt}], images=[image],
                                   max_tokens=max_tokens, temperature=0.0)
        except LLMError as exc:
            log.info("Vision call skipped (%s)", exc.kind)
            return ""
        with self._lock:
            self.cost += res.cost
            self.calls += 1
        return (res.result.content or "").strip()

    # ------------------------------------------------------------------ OCR
    def ocr_pages(self, pdf_bytes: bytes, pages: Sequence[int], progress: ProgressFn = None) -> Dict[int, str]:
        """page number -> transcribed Markdown (pages that failed are left out)."""
        import pymupdf

        pages = list(pages)[: self.settings.max_ocr_pages]
        if not pages:
            return {}
        images: Dict[int, bytes] = {}
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            for p in pages:
                page = doc[p - 1]
                zoom = min(150 / 72, OCR_MAX_PX / max(page.rect.width, page.rect.height, 1))
                images[p] = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False).tobytes("png")
        out: Dict[int, str] = {}
        done = 0

        def work(p: int):
            return p, self._ask(OCR_PROMPT, images[p], max_tokens=4000)

        with ThreadPoolExecutor(max_workers=self.settings.vision_workers) as pool:
            for p, text in pool.map(work, pages):
                done += 1
                if progress:
                    progress(done / len(pages), f"Reading scanned page {p} with the vision model ({done}/{len(pages)})")
                if len(text.strip()) >= 10:
                    out[p] = text
        log.info("OCR: %d/%d scanned pages transcribed", len(out), len(pages))
        return out

    # -------------------------------------------------------------- figures
    def describe_figures(self, figures: List, figures_dir: Path, cache, progress: ProgressFn = None) -> int:
        """Describe figures that have no description yet (largest first). Returns how many were added."""
        todo = [f for f in figures if f.figure_file and not (f.meta or {}).get("vision")]
        known: Dict[str, str] = {}
        for f in todo:  # reuse the cache (same image can appear on several pages)
            cached = cache.get(f.figure_file)
            if cached:
                known[f.figure_file] = cached
        files = []
        for f in todo:
            path = figures_dir / f.figure_file
            if f.figure_file not in known and f.figure_file not in files and path.exists():
                files.append(f.figure_file)
        files.sort(key=lambda name: -(figures_dir / name).stat().st_size)
        files = files[: self.settings.max_vision_figures]
        captions = {f.figure_file: (f.caption or "") for f in todo}
        done = 0

        def work(name: str):
            cap = captions.get(name, "")
            hint = f' The caption printed next to it is: "{cap}".' if cap else ""
            return name, self._ask(FIGURE_PROMPT.format(hint=hint), (figures_dir / name).read_bytes(), 400)

        if files:
            with ThreadPoolExecutor(max_workers=self.settings.vision_workers) as pool:
                for name, text in pool.map(work, files):
                    done += 1
                    if progress:
                        progress(done / len(files), f"Describing figures with the vision model ({done}/{len(files)})")
                    if text:
                        known[name] = text[:1200]
                        cache.set(name, known[name])
        added = 0
        for f in todo:
            if f.figure_file in known:
                f.meta = {**(f.meta or {}), "vision": known[f.figure_file]}
                added += 1
        log.info("Figure descriptions: %d added (%d vision calls)", added, len(files))
        return added


def make_vision(router, settings: Settings) -> Optional[VisionHelper]:
    """A helper when a vision model is configured and indexing-time vision is on."""
    if router is None or not router.has_role("vision"):
        return None
    if not (settings.vision_index or settings.ocr_scanned):
        return None
    return VisionHelper(router, settings)
