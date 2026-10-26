"""Chunking: overlap exists, metadata survives, text fully covered."""

from ragqa.ingestion.chunking import Chunker
from ragqa.models import Document


def test_chunks_overlap_and_cover_text():
    text = " ".join(f"word{i}" for i in range(400))  # ~2900 chars
    doc = Document(doc_id="d1", title="t", source="sample.txt", text=text)
    chunks = Chunker(chunk_size=800, chunk_overlap=150).chunk_document(doc)
    assert len(chunks) > 1
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.start_char < prev.end_char  # overlap, not a hard cut
    assert chunks[0].start_char == 0
    assert chunks[-1].end_char == len(text.strip())


def test_chunk_metadata_preserved():
    doc = Document(doc_id="d2", title="t", source="handbook.txt", text="hello world " * 200)
    chunks = Chunker(chunk_size=200, chunk_overlap=50).chunk_document(doc)
    assert chunks
    for chunk in chunks:
        assert chunk.doc_id == "d2"
        assert chunk.source == "handbook.txt"
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_page_metadata_preserved_for_pdfs():
    doc = Document(doc_id="d3", title="t", source="manual.pdf", text="")
    doc.metadata["pages"] = ["first page content " * 30, "second page content " * 30]
    chunks = Chunker(chunk_size=200, chunk_overlap=50).chunk_document(doc)
    pages = {c.page for c in chunks}
    assert pages == {1, 2}
    assert all(c.chunk_id.startswith("d3:p") for c in chunks)


def test_overlap_must_be_smaller_than_size():
    import pytest

    with pytest.raises(ValueError):
        Chunker(chunk_size=100, chunk_overlap=100)
