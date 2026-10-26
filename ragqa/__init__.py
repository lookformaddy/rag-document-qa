"""ragqa — a document question-answering system using Retrieval-Augmented
Generation (RAG), built around the classic two-pipeline architecture:

Ingestion pipeline: documents -> text extraction -> cleaning & chunking
(with source/page metadata) -> embeddings -> vector store, plus a document
metadata store (info, processing status, versions, access control).

Query pipeline: question -> query embedding (same model as ingestion) ->
retrieve relevant chunks (vector similarity, optional keyword hybrid and
reranking) -> build context -> generate an answer with source citations ->
response (answer + citations + document links).

The default stack runs fully offline with zero API keys: a deterministic
hashing embedder and an extractive, citation-attaching answerer. Real
embedding models and LLM APIs plug in behind the same interfaces.
"""

from ragqa.config import Settings
from ragqa.models import Answer, Chunk, Citation, Document, RetrievalResult
from ragqa.service import QAService

__all__ = [
    "Answer",
    "Chunk",
    "Citation",
    "Document",
    "QAService",
    "RetrievalResult",
    "Settings",
]
__version__ = "0.1.0"
