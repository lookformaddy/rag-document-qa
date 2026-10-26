"""Ingestion pipeline: load -> extract -> clean -> chunk -> embed -> store.

The pipeline also owns the document metadata store: every document is
recorded with its processing status, version, and access roles, so the
query side can filter results and report what is (and is not) indexed.
Re-ingesting the same doc_id bumps the version and replaces its chunks.
"""

from ragqa.embeddings import EmbeddingProvider, get_embedding_provider
from ragqa.ingestion.chunking import Chunker
from ragqa.ingestion.loaders import load_document
from ragqa.models import Document
from ragqa.store import VectorStore, get_vector_store


class IngestionPipeline:
    def __init__(self, embedder=None, store=None, chunker=None, settings=None):
        from ragqa.config import Settings

        self.settings = settings or Settings()
        self.embedder: EmbeddingProvider = embedder or get_embedding_provider(self.settings)
        self.store: VectorStore = store or get_vector_store(self.settings)
        self.chunker = chunker or Chunker(
            self.settings.chunk_size, self.settings.chunk_overlap
        )
        self.documents: dict = {}  # doc_id -> Document (the metadata store)

    def ingest_document(self, doc: Document) -> Document:
        """Run one document through the full pipeline and index its chunks."""
        doc.status = "pending"
        try:
            chunks = self.chunker.chunk_document(doc)
            texts = [c.text for c in chunks]
            vectors = self.embedder.embed_texts(texts) if texts else []
            for chunk, vector in zip(chunks, vectors):
                chunk.embedding = vector
            self.store.delete_document(doc.doc_id)
            if chunks:
                self.store.add(chunks)
            doc.status = "indexed"
            doc.version = self.documents.get(doc.doc_id, doc).version + (
                1 if doc.doc_id in self.documents else 0
            )
            self.documents[doc.doc_id] = doc
            return doc
        except Exception:
            doc.status = "failed"
            self.documents[doc.doc_id] = doc
            raise

    def ingest_file(self, path: str, doc_id=None, allowed_roles=()) -> Document:
        """Load a file from disk and ingest it."""
        doc = load_document(path, doc_id)
        doc.allowed_roles = tuple(allowed_roles)
        return self.ingest_document(doc)

    def ingest_text(self, doc_id: str, title: str, source: str, text: str,
                    allowed_roles=()) -> Document:
        """Ingest already-extracted text (web pages, KB articles, tests)."""
        doc = Document(doc_id=doc_id, title=title, source=source, text=text)
        doc.allowed_roles = tuple(allowed_roles)
        return self.ingest_document(doc)

    def list_documents(self) -> list:
        return list(self.documents.values())
