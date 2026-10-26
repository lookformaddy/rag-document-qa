"""Vector store (ingestion step 5).

Default: an exact in-memory cosine store — correct, dependency-free, and
fast enough for thousands of chunks. Optional: a FAISS adapter (lazy
import) for larger corpora. Both implement the same interface, so the
swap is one settings change. Access filtering happens at retrieval time
via the metadata store, not inside the store, keeping the store generic.
"""

import math
from typing import Protocol

from ragqa.models import Chunk


class VectorStore(Protocol):
    def add(self, chunks: list) -> None:
        ...

    def delete_document(self, doc_id: str) -> None:
        ...

    def search(self, query_vector: list, top_k: int) -> list:
        """Return [(chunk, cosine_score)] sorted by score descending."""
        ...


def cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class InMemoryVectorStore:
    """Exact brute-force cosine store."""

    def __init__(self):
        self._chunks: list = []

    def add(self, chunks: list) -> None:
        self._chunks.extend(chunks)

    def delete_document(self, doc_id: str) -> None:
        self._chunks = [c for c in self._chunks if c.doc_id != doc_id]

    def search(self, query_vector: list, top_k: int) -> list:
        scored = [
            (c, cosine(query_vector, c.embedding))
            for c in self._chunks
            if c.embedding is not None
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self._chunks)


class FaissVectorStore:
    """FAISS-backed store (``pip install ragqa[faiss]``), same interface."""

    def __init__(self, dimension: int):
        try:
            import faiss
            import numpy as np
        except ImportError as exc:
            raise ImportError("pip install ragqa[faiss] for FAISS support") from exc
        self._faiss = faiss
        self._np = np
        self._index = faiss.IndexFlatIP(dimension)
        self._chunks: list = []

    def add(self, chunks: list) -> None:
        import numpy as np

        vectors = np.array([c.embedding for c in chunks], dtype="float32")
        self._index.add(vectors)
        self._chunks.extend(chunks)

    def delete_document(self, doc_id: str) -> None:
        # IndexFlatIP has no deletion; rebuild from the surviving chunks.
        survivors = [c for c in self._chunks if c.doc_id != doc_id]
        self._index.reset()
        self._chunks = []
        if survivors:
            self.add(survivors)

    def search(self, query_vector: list, top_k: int) -> list:
        import numpy as np

        query = np.array([query_vector], dtype="float32")
        scores, ids = self._index.search(query, min(top_k, len(self._chunks)))
        return [(self._chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]


def get_vector_store(settings) -> VectorStore:
    name = getattr(settings, "vector_store", "memory")
    if name == "faiss":
        return FaissVectorStore(settings.embedding_dim)
    return InMemoryVectorStore()
