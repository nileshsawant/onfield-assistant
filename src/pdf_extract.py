"""PDF text extraction for RAG ingestion.

Small wrapper around ``pdfplumber`` (MIT-licensed, pure Python) that
returns per-page text plus a bit of structure. Kept as its own module so
``rebuild_indices.py`` stays framework-agnostic and so a different
backend (pdftotext, pymupdf, etc.) could be swapped in later.

Design notes:
  * We keep one chunk per PDF page as the unit passed on to the
    downstream chunker. Multi-page documents that need finer splitting
    are handled by the fixed-size character chunker in
    ``rebuild_indices.py``, which will further split any page whose
    extracted text exceeds the target chunk size.
  * Empty / near-empty pages (< 20 chars after strip) are skipped —
    typically figure-only or fully-image pages.
  * Extraction failures on individual pages are logged and skipped, not
    propagated, so one bad page doesn't kill the whole ingestion.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterator


_MIN_PAGE_CHARS = 20   # skip pages with less usable text than this

# pdfplumber emits a literal "(cid:N)" token whenever a glyph has no
# Unicode mapping in the PDF's font (common for math symbols in older
# scientific PDFs). The token carries no recoverable information and only
# pollutes chunks / embeddings, so drop it.
_CID_RE = re.compile(r"\(cid:\d+\)")

# Vision-OCR tuning. Rendered pages go to a local vision model via
# ofa_main.chat_stream, so nothing leaves the node.
_OCR_RESOLUTION = 150          # DPI for page render; 150 is legible without huge PNGs
_OCR_TIMEOUT_S = 180           # per-page vision call ceiling
# A page "needs" OCR when the text layer is clearly degraded: many unmapped
# glyphs relative to its length, or almost no extractable text at all.
_OCR_CID_RATIO = 0.005         # >0.5% of chars were (cid:N) tokens
_OCR_MIN_TEXT = 200            # fewer than this many chars on a non-trivial page

_OCR_PROMPT = (
    "Transcribe ALL text from this page image to Markdown, exactly as it "
    "appears and in natural reading order (respect columns). Render every "
    "mathematical expression in LaTeX: inline as $...$ and displayed "
    "equations as $$...$$. Do not summarise, explain, or add commentary — "
    "output only the transcription."
)


def _count_cid(raw_before_clean: str) -> int:
    return len(_CID_RE.findall(raw_before_clean))


def _needs_ocr(raw_text: str) -> bool:
    """Heuristic: does this page's text layer look too degraded to trust?

    Runs against the RAW extract (before _clean_page_text strips cid tokens),
    so the cid ratio is measurable.
    """
    n = len(raw_text)
    if n < _OCR_MIN_TEXT:
        return True
    return (_count_cid(raw_text) / max(n, 1)) > _OCR_CID_RATIO


def _render_page_png_b64(page, resolution: int = _OCR_RESOLUTION) -> str | None:
    """Render a pdfplumber page to a base64 PNG using pdfplumber's own
    to_image() (no poppler / pdf2image needed). Returns None on failure."""
    import base64
    import io
    try:
        img = page.to_image(resolution=resolution).original  # PIL image
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as e:
        print(f"[pdf_extract] page render failed: {e}", file=sys.stderr)
        return None


def _ocr_page(page) -> str:
    """Transcribe one page via the local ofa vision model. Returns "" on
    any failure so the caller can fall back to the text-layer extract.

    Imported lazily to avoid a hard dependency cycle (ofa_main imports this
    module's extract_pages) and so non-OCR ingestion never pays the cost.
    """
    b64 = _render_page_png_b64(page)
    if not b64:
        return ""
    try:
        import ofa_main
    except ImportError:
        print("[pdf_extract] OCR requested but ofa_main not importable; "
              "skipping OCR for this page.", file=sys.stderr)
        return ""
    if not ofa_main.model_supports_vision():
        print(f"[pdf_extract] OCR requested but model '{ofa_main.MODEL}' has no "
              "vision support; skipping OCR. Set OFA_MODEL to a vision-capable "
              "model (e.g. gemma4:31b-it-q8_0).", file=sys.stderr)
        return ""
    messages = [{"role": "user", "content": _OCR_PROMPT, "images": [b64]}]
    try:
        out = "".join(ofa_main.chat_stream(
            messages, num_predict=4096, temperature=0.0,
        ))
        return out.strip()
    except Exception as e:
        print(f"[pdf_extract] OCR vision call failed: {e}", file=sys.stderr)
        return ""


def _clean_page_text(text: str) -> str:
    text = _CID_RE.sub("", text)
    # Collapse the runs of spaces that removing cid tokens can leave behind,
    # without touching newlines (page layout still matters for chunking).
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


def extract_pages(
    pdf_path: Path,
    *,
    start_page: int = 1,
    end_page: int | None = None,
    ocr: str = "off",
) -> Iterator[tuple[int, str]]:
    """Yield ``(page_number, text)`` tuples for each non-empty page of *pdf_path*.

    ``page_number`` is 1-indexed to match how humans (and citations)
    refer to PDF pages. ``text`` is UTF-8 with layout roughly preserved
    (``pdfplumber``'s default extract_text output).

    ``start_page`` (1-based, inclusive) skips pages before it. ``end_page``
    (1-based, inclusive; ``None`` = read to end) stops after it. Use these
    to ingest only a chapter or appendix of a large textbook without
    embedding its front-matter, table of contents, or unrelated chapters.

    ``ocr`` controls vision-OCR via the local ofa model (nothing leaves the
    node):
      * ``"off"`` (default) — text-layer extraction only.
      * ``"auto"`` — OCR only pages whose text layer looks degraded
        (``_needs_ocr``: many unmapped glyphs, or almost no text). Best
        for scientific PDFs where most pages are fine but equation-dense
        ones aren't.
      * ``"force"`` — OCR every page. Slow; use for scanned documents with
        no usable text layer at all.

    Raises ``ImportError`` if pdfplumber isn't installed. Raises
    ``FileNotFoundError`` if the PDF doesn't exist. Per-page extraction
    errors are printed to stderr and the page is skipped.
    """
    try:
        import pdfplumber  # noqa: PLC0415 — deliberate lazy import
    except ImportError as e:
        raise ImportError(
            "pdfplumber is required for PDF ingestion. Install it into "
            "the assistant env: `$OFA_ROOT/env/bin/pip install pdfplumber`"
        ) from e

    if not pdf_path.is_file():
        raise FileNotFoundError(f"pdf not found: {pdf_path}")

    try:
        pdf = pdfplumber.open(str(pdf_path))
    except Exception as e:
        print(f"[pdf_extract] failed to open {pdf_path.name}: {e}",
              file=sys.stderr)
        return

    try:
        for i, page in enumerate(pdf.pages, start=1):
            if i < start_page:
                continue
            if end_page is not None and i > end_page:
                break
            try:
                # x_tolerance=1.5 (default 3.0) is the single biggest lever
                # for scientific PDFs: at the default, tightly-set body text
                # loses inter-word spaces ("mergesquantummechanics"), which
                # wrecks both retrieval and the model's comprehension. A
                # smaller tolerance keeps genuine word gaps.
                raw = page.extract_text(x_tolerance=1.5) or ""
            except Exception as e:
                print(f"[pdf_extract] {pdf_path.name} page {i}: {e}",
                      file=sys.stderr)
                continue

            use_ocr = ocr == "force" or (ocr == "auto" and _needs_ocr(raw))
            if use_ocr:
                print(f"[pdf_extract] OCR page {i} of {pdf_path.name}...",
                      file=sys.stderr)
                ocr_text = _ocr_page(page)
                # Only accept OCR if it produced more than the text layer;
                # a failed/empty vision call must not blank out a page that
                # had usable (if imperfect) text.
                text = ocr_text if len(ocr_text) > len(_clean_page_text(raw)) else _clean_page_text(raw)
            else:
                text = _clean_page_text(raw)

            if len(text.strip()) < _MIN_PAGE_CHARS:
                continue
            yield i, text
    finally:
        pdf.close()


def extract_all(pdf_path: Path, *, ocr: str = "off") -> str:
    """Convenience: return the whole PDF as one text blob with page
    markers so downstream chunking can still surface page numbers via
    regex if needed."""
    parts = []
    for page_num, text in extract_pages(pdf_path, ocr=ocr):
        parts.append(f"[page {page_num}]\n{text}")
    return "\n\n".join(parts)


if __name__ == "__main__":
    # Small CLI so users can sanity-check what a PDF extracts to:
    #   python3 src/pdf_extract.py path/to/thesis.pdf
    if len(sys.argv) != 2:
        print("usage: python3 pdf_extract.py <path.pdf>", file=sys.stderr)
        sys.exit(2)
    p = Path(sys.argv[1])
    for page_num, text in extract_pages(p):
        print(f"---- page {page_num} ({len(text)} chars) ----")
        print(text[:500] + ("..." if len(text) > 500 else ""))
