"""Core data types shared by both pipelines."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Document:
    """One ingested source document."""

    doc_id: str
    title: str
    source: str  # e.g. file path, URL, or knowledge-base key
    text: str = ""
    version: int = 1
    status: str = "pending"  # pending | indexed | failed
    allowed_roles: tuple = ()  # empty = visible to everyone
    metadata: dict = field(default_factory=dict)

    @property
    def link(self) -> str:
        """Stable document link surfaced in answers."""
        return f"doc://{self.doc_id}"


@dataclass
class Chunk:
    """A retrievable slice of a document. Metadata is preserved end to end."""

    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    source: str
    page: Optional[int] = None
    start_char: int = 0
    end_char: int = 0
    embedding: Optional[list] = None


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    keyword_score: float = 0.0
    vector_score: float = 0.0


@dataclass
class Citation:
    source: str
    doc_id: str
    page: Optional[int]
    chunk_id: str
    excerpt: str

    def render(self) -> str:
        page = f", page {self.page}" if self.page is not None else ""
        return f"[{self.source}{page}]"


@dataclass
class Answer:
    question: str
    text: str
    citations: list
    retrieved: list
    abstained: bool = False

    def render(self) -> str:
        lines = [self.text, ""]
        if self.citations:
            lines.append("Sources:")
            for c in self.citations:
                lines.append(f"  {c.render()} {c.excerpt}")
        return "\n".join(lines)
