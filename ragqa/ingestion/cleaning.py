"""Text cleaning (ingestion step 3, first half).

Cleaning is deliberately conservative: it normalizes whitespace and
repairs hyphenated line breaks, but never drops sentences, so citations
always point at text that exists in the source document.
"""

import re

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text: str) -> str:
    """Normalize whitespace, de-hyphenate wrapped lines, strip controls."""
    text = _CONTROL.sub("", text)
    # Rejoin words split by a hyphen at a line break: "deploy-\nment" -> "deployment"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3+ blank lines, and runs of spaces/tabs, but keep paragraphs.
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_paragraphs(text: str) -> list:
    """Split cleaned text into non-empty paragraphs."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]
