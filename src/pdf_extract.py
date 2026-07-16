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

import sys
from pathlib import Path
from typing import Iterator


_MIN_PAGE_CHARS = 20   # skip pages with less usable text than this


def extract_pages(pdf_path: Path) -> Iterator[tuple[int, str]]:
    """Yield ``(page_number, text)`` tuples for each non-empty page of *pdf_path*.

    ``page_number`` is 1-indexed to match how humans (and citations)
    refer to PDF pages. ``text`` is UTF-8 with layout roughly preserved
    (``pdfplumber``'s default extract_text output).

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
            try:
                text = page.extract_text() or ""
            except Exception as e:
                print(f"[pdf_extract] {pdf_path.name} page {i}: {e}",
                      file=sys.stderr)
                continue
            if len(text.strip()) < _MIN_PAGE_CHARS:
                continue
            yield i, text
    finally:
        pdf.close()


def extract_all(pdf_path: Path) -> str:
    """Convenience: return the whole PDF as one text blob with page
    markers so downstream chunking can still surface page numbers via
    regex if needed."""
    parts = []
    for page_num, text in extract_pages(pdf_path):
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
