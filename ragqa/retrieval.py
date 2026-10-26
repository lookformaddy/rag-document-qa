"""Retrieval (query pipeline steps 2–3).

Retriever: embed the question with the SAME provider as ingestion, run a
vector search, optionally blend in a keyword score (hybrid search), apply
the similarity threshold, and rerank by term coverage so the chunk that
actually contains the question's terms wins over a merely similar one.

Abstention is a retrieval decision, not a generation trick: if nothing
passes the threshold, the caller gets an empty list and the answerer
says so instead of hallucinating.
"""

import math
import re

from ragqa.models import RetrievalResult

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what", "how", "when",
    "where", "who", "does", "do", "did", "can", "could", "should", "of",
    "to", "in", "on", "for", "and", "or", "with", "by", "from", "it", "its",
}


def content_tokens(text: str) -> set:
    return {t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS}


def keyword_score(query: str, chunk_text: str) -> float:
    """Fraction of the question's content terms present in the chunk."""
    terms = content_tokens(query)
    if not terms:
        return 0.0
    found = terms & content_tokens(chunk_text)
    return len(found) / len(terms)


def rerank(query: str, results: list) -> list:
    """Reorder by blended score, then term coverage, then vector score.

    The blend is computed in ``retrieve``; reranking only re-sorts by the
    same signals so a chunk that literally contains the answer terms
    beats a chunk that is merely near the question in vector space.
    """
    return sorted(
        results,
        key=lambda r: (r.score, r.keyword_score, r.vector_score),
        reverse=True,
    )


class Retriever:
    def __init__(self, embedder, store, settings, documents=None):
        self.embedder = embedder
        self.store = store
        self.settings = settings
        # documents: doc_id -> Document, used for access-control filtering.
        self.documents = documents if documents is not None else {}

    def _visible(self, chunk, roles) -> bool:
        doc = self.documents.get(chunk.doc_id)
        if doc is None or not doc.allowed_roles:
            return True
        return bool(set(roles or ()) & set(doc.allowed_roles))

    def retrieve(self, question: str, top_k=None, roles=()) -> list:
        """Return ranked RetrievalResults, empty when nothing is relevant."""
        k = top_k or self.settings.top_k
        query_vector = self.embedder.embed_texts([question])[0]
        # Over-fetch so threshold + access filtering still leave top_k.
        candidates = self.store.search(query_vector, top_k=k * 3)
        results = []
        for chunk, vector_score in candidates:
            if not self._visible(chunk, roles):
                continue
            kw = keyword_score(question, chunk.text) if self.settings.hybrid else 0.0
            blended = (
                (1 - self.settings.hybrid_keyword_weight) * vector_score
                + self.settings.hybrid_keyword_weight * kw
                if self.settings.hybrid
                else vector_score
            )
            if blended < self.settings.similarity_threshold:
                continue
            results.append(
                RetrievalResult(
                    chunk=chunk, score=blended,
                    keyword_score=kw, vector_score=vector_score,
                )
            )
        return rerank(question, results)[:k]
