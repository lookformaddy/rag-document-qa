"""Document loading and text extraction (ingestion step 1–2).

Supported inputs: plain text / Markdown files, PDFs (via PyMuPDF or pypdf
when installed), and raw bytes. The OCR hook is an explicit extension
point: it raises a clear error unless an OCR backend is configured,
instead of silently dropping scanned pages.
"""

import os
import re
from typing import Optional

from ragqa.models import Document

TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst"}


class UnsupportedDocumentError(ValueError):
    pass


class OcrNotConfiguredError(RuntimeError):
    """Raised when a scanned/image-only page needs OCR and none is set up."""


def load_text_file(path: str, doc_id: Optional[str] = None) -> Document:
    """Load a plain-text document, preserving its source as metadata."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    return Document(
        doc_id=doc_id or os.path.basename(path),
        title=os.path.basename(path),
        source=path,
        text=text,
    )


def extract_pdf_text(path: str) -> list:
    """Return one text string per PDF page.

    Tries PyMuPDF, then pypdf. A page that yields no text is reported as
    an empty string with a flag the caller can route to the OCR hook.
    """
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(path)
        return [page.get_text() for page in doc]
    except ImportError:
        pass
    try:
        from pypdf import PdfReader

        return [(page.extract_text() or "") for page in PdfReader(path).pages]
    except ImportError:
        pass
    raise UnsupportedDocumentError(
        "PDF extraction needs PyMuPDF or pypdf: pip install ragqa[pdf]"
    )


def ocr_page(image_bytes: bytes) -> str:
    """OCR hook. Wire a real backend (e.g. tesseract) here.

    Deliberately fail-closed: scanned content must never silently vanish
    from the index, so callers surface this error as a failed document.
    """
    raise OcrNotConfiguredError(
        "OCR backend not configured; install tesseract and set RAGQA_OCR=enabled, "
        "or supply extracted text for scanned pages."
    )


def load_document(path: str, doc_id: Optional[str] = None) -> Document:
    """Load any supported file into a Document with per-page text kept.

    For PDFs, page boundaries are preserved in ``metadata['pages']`` so
    the chunker can attach a page number to every chunk.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in TEXT_EXTENSIONS:
        return load_text_file(path, doc_id)
    if ext == ".pdf":
        pages = extract_pdf_text(path)
        doc = Document(
            doc_id=doc_id or os.path.basename(path),
            title=os.path.basename(path),
            source=path,
            text="\n\n".join(pages),
        )
        doc.metadata["pages"] = pages
        return doc
    raise UnsupportedDocumentError(f"Unsupported document type: {ext or '(none)'}")


def split_pages(text: str) -> list:
    """Best-effort page split for plain text (form-feed markers)."""
    return [p for p in re.split(r"\f", text) if p.strip()]
