"""Embedding providers (ingestion step 4, shared with the query side).

The query pipeline MUST use the same provider as ingestion — vectors from
different models are not comparable. ``get_embedding_provider`` is the
single place both sides resolve the provider from settings.

Default: a deterministic hashing embedder (zero dependencies, zero API
keys, reproducible). Optional: sentence-transformers and OpenAI, both
behind env flags and lazy imports so the core never depends on them.
"""

import hashlib
import math
import re
from typing import Protocol

_TOKEN = re.compile(r"[a-z0-9]+")


class EmbeddingProvider(Protocol):
    dimension: int

    def embed_texts(self, texts: list) -> list:
        ...


class HashingEmbedder:
    """Deterministic offline embedder: tokens hashed into a fixed vector.

    Each token contributes to a small set of hashed buckets; the vector is
    L2-normalized so cosine similarity is a dot product. Quality is far
    below a trained model — this is a working default and a test double,
    not a quality claim. The interface is what matters: swap in a real
    provider without touching ingestion, retrieval, or generation.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _embed_one(self, text: str) -> list:
        vec = [0.0] * self.dimension
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.sha256(token.encode()).digest()
            for k in range(4):  # 4 buckets per token reduces collisions
                bucket = int.from_bytes(digest[k * 2 : k * 2 + 2], "big") % self.dimension
                vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_texts(self, texts: list) -> list:
        return [self._embed_one(t) for t in texts]


class SentenceTransformersEmbedder:
    """Adapter over sentence-transformers (``pip install ragqa[st]``)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("pip install ragqa[st] for sentence-transformers") from exc
        self._model = SentenceTransformer(model_name)
        self.dimension = self._model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list) -> list:
        return [list(map(float, v)) for v in self._model.encode(texts, normalize_embeddings=True)]


class OpenAIEmbedder:
    """Adapter over the OpenAI embeddings API (needs OPENAI_API_KEY)."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: str = ""):
        import os

        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI embedder")
        self.dimension = 1536  # text-embedding-3-small default

    def embed_texts(self, texts: list) -> list:
        import json
        import urllib.request

        body = json.dumps({"model": self.model, "input": texts}).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.load(resp)
        return [item["embedding"] for item in payload["data"]]


def get_embedding_provider(settings) -> EmbeddingProvider:
    name = settings.embedding_provider.lower()
    if name == "hashing":
        return HashingEmbedder(settings.embedding_dim)
    if name in ("sentence-transformers", "st"):
        return SentenceTransformersEmbedder()
    if name == "openai":
        return OpenAIEmbedder(api_key=settings.openai_api_key)
    raise ValueError(f"Unknown embedding provider: {name}")
