"""Chunking with overlap and metadata (ingestion step 3, second half).

Chunks are character windows over the cleaned text, cut on word
boundaries. Overlap exists so an answer span straddling a boundary is
still retrievable whole from at least one chunk. Every chunk carries its
source document, page (when known), and character offsets — the metadata
that later becomes a citation.
"""

from ragqa.ingestion.cleaning import clean_text
from ragqa.models import Chunk, Document


class Chunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _windows(self, text: str):
        """Yield (start, end) character windows, stepping by size-overlap."""
        step = self.chunk_size - self.chunk_overlap
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            # Prefer a word boundary so chunks never split mid-word.
            if end < len(text):
                boundary = text.rfind(" ", start, end)
                if boundary > start + self.chunk_size // 2:
                    end = boundary
            yield start, end
            if end >= len(text):
                break
            start = end - self.chunk_overlap

    def chunk_document(self, doc: Document) -> list:
        """Chunk a document, attaching source + page metadata per chunk."""
        pages = doc.metadata.get("pages")
        chunks: list = []
        if pages:
            # Page-aware chunking: chunk each page separately so the page
            # number on every chunk is exact, not estimated.
            offset = 0
            for page_no, page_text in enumerate(pages, start=1):
                cleaned = clean_text(page_text)
                for i, (s, e) in enumerate(self._windows(cleaned)):
                    chunks.append(
                        Chunk(
                            chunk_id=f"{doc.doc_id}:p{page_no}c{i}",
                            doc_id=doc.doc_id,
                            text=cleaned[s:e],
                            chunk_index=len(chunks),
                            source=doc.source,
                            page=page_no,
                            start_char=offset + s,
                            end_char=offset + e,
                        )
                    )
                offset += len(cleaned) + 2
            return chunks
        cleaned = clean_text(doc.text)
        for i, (s, e) in enumerate(self._windows(cleaned)):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}:c{i}",
                    doc_id=doc.doc_id,
                    text=cleaned[s:e],
                    chunk_index=i,
                    source=doc.source,
                    page=None,
                    start_char=s,
                    end_char=e,
                )
            )
        return chunks
