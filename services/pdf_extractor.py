"""Extract screenplay text from a PDF document."""

from __future__ import annotations

from pathlib import Path

import fitz


class PDFExtractionError(RuntimeError):
    """The PDF cannot be read or contains no extractable text."""


def extract_script_text(path: str | Path) -> str:
    """Extract all pages in document order; scanned-only PDFs require OCR."""
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise PDFExtractionError(f"Script PDF not found: {pdf_path}")
    try:
        with fitz.open(pdf_path) as document:
            if document.needs_pass:
                raise PDFExtractionError("The script PDF is password-protected.")
            pages = [page.get_text("text") for page in document]
    except (RuntimeError, ValueError, OSError) as exc:
        if isinstance(exc, PDFExtractionError):
            raise
        raise PDFExtractionError(f"Could not extract PDF text: {pdf_path}") from exc
    text = "\n\n".join(pages).strip()
    if not text:
        raise PDFExtractionError("The PDF has no extractable text; OCR may be needed.")
    return text
