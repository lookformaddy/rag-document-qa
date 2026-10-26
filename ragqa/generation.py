"""Answer generation (query pipeline steps 4–6).

Default: an offline extractive answerer. It selects the sentences from
the retrieved chunks that best cover the question's terms, attaches a
citation (source + page + excerpt) to each, and ABSTAINS — explicitly —
when retrieval returned nothing relevant. Extractive answering cannot
hallucinate new facts; it can only quote what was retrieved.

Optional: an OpenAI-compatible adapter that receives the same built
context (chunks + sources) with instructions to cite sources, and parses
its citations back into the same Answer shape.
"""

import re
from typing import Protocol

from ragqa.models import Answer, Citation
from ragqa.retrieval import content_tokens

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


class LLM(Protocol):
    def generate(self, question: str, context: str, citations: list) -> str:
        ...


def build_context(results: list) -> tuple:
    """Format top-k chunks for the LLM, with sources attached.

    Returns (context_text, citations). Each chunk is labeled with its
    source so the model — or the extractive answerer — can cite it.
    """
    parts, citations = [], []
    for result in results:
        chunk = result.chunk
        label = chunk.source + (f", page {chunk.page}" if chunk.page else "")
        parts.append(f"[Source: {label}]\n{chunk.text}")
        excerpt = chunk.text.strip().replace("\n", " ")[:160]
        citations.append(
            Citation(
                source=chunk.source, doc_id=chunk.doc_id, page=chunk.page,
                chunk_id=chunk.chunk_id, excerpt=excerpt,
            )
        )
    return "\n\n".join(parts), citations


class ExtractiveAnswerer:
    """Quote-based answerer: best sentences + citations, or abstention."""

    def generate(self, question: str, context: str, citations: list) -> str:
        terms = content_tokens(question)
        # Strip the [Source: ...] label lines first: otherwise a label
        # merges with the following sentence and that sentence is skipped.
        body = "\n".join(
            line for line in context.splitlines() if not line.startswith("[Source:")
        )
        sentences = [s.strip() for s in _SENTENCE.split(body) if s.strip()]
        scored = []
        for sentence in sentences:
            overlap = len(terms & content_tokens(sentence))
            if overlap:
                scored.append((overlap, sentence))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        best = [s for _, s in scored[:3]]
        if not best:
            return ""
        return " ".join(best)


class OpenAIAnswerer:
    """OpenAI-compatible chat completions adapter."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str = ""):
        import os

        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI answerer")

    def generate(self, question: str, context: str, citations: list) -> str:
        import json
        import urllib.request

        system = (
            "Answer using ONLY the retrieved context below. "
            "Cite every factual claim with its [Source: ...] label. "
            "If the context does not contain the answer, say so plainly."
        )
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                ],
                "temperature": 0.0,
            }
        ).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.load(resp)
        return payload["choices"][0]["message"]["content"]


def get_answerer(settings):
    name = settings.llm_provider.lower()
    if name == "extractive":
        return ExtractiveAnswerer()
    if name == "openai":
        return OpenAIAnswerer(model=settings.openai_model, api_key=settings.openai_api_key)
    raise ValueError(f"Unknown LLM provider: {name}")
