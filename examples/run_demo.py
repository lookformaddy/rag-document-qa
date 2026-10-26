"""End-to-end demo: ingest the synthetic sample docs, ask two questions,
print answers with citations. Runs fully offline, zero API keys.

    python examples/run_demo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragqa.cli import _ingest_samples  # noqa: E402
from ragqa.config import Settings  # noqa: E402
from ragqa.service import QAService  # noqa: E402


def main() -> None:
    service = QAService(Settings())
    _ingest_samples(service)
    print(f"Indexed {len(service.list_documents())} documents, "
          f"{len(service.pipeline.store)} chunks.\n")
    for question in (
        "What is the refund window for standard orders?",
        "How many remote days per week are allowed?",
        "What is the capital of France?",  # not in the docs -> abstention
    ):
        print(f"Q: {question}")
        print(service.ask(question).render())
        print("-" * 60)


if __name__ == "__main__":
    main()
