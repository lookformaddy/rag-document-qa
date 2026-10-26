"""Configuration, loaded from the environment (see .env.example)."""

import os
from dataclasses import dataclass, field


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    """All tunables for the two pipelines, with sane offline defaults."""

    # Chunking
    chunk_size: int = field(default_factory=lambda: _int("RAGQA_CHUNK_SIZE", 800))
    chunk_overlap: int = field(default_factory=lambda: _int("RAGQA_CHUNK_OVERLAP", 150))
    # Embeddings
    embedding_provider: str = field(
        default_factory=lambda: os.environ.get("RAGQA_EMBEDDING_PROVIDER", "hashing")
    )
    embedding_dim: int = field(default_factory=lambda: _int("RAGQA_EMBEDDING_DIM", 384))
    # Retrieval
    top_k: int = field(default_factory=lambda: _int("RAGQA_TOP_K", 5))
    similarity_threshold: float = field(
        default_factory=lambda: _float("RAGQA_SIMILARITY_THRESHOLD", 0.15)
    )
    hybrid: bool = field(
        default_factory=lambda: os.environ.get("RAGQA_HYBRID", "true").lower() == "true"
    )
    hybrid_keyword_weight: float = field(
        default_factory=lambda: _float("RAGQA_HYBRID_KEYWORD_WEIGHT", 0.3)
    )
    # Generation
    llm_provider: str = field(
        default_factory=lambda: os.environ.get("RAGQA_LLM_PROVIDER", "extractive")
    )
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    openai_model: str = field(
        default_factory=lambda: os.environ.get("RAGQA_OPENAI_MODEL", "gpt-4o-mini")
    )
    # API
    host: str = field(default_factory=lambda: os.environ.get("RAGQA_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _int("RAGQA_PORT", 8000))

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in [0, 1]")
