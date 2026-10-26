"""CLI: ingest documents and ask questions without the API.

Usage:
    python -m ragqa.cli ingest examples/sample_docs/*.txt
    python -m ragqa.cli ask "What is the refund window?"
    python -m ragqa.cli serve          # FastAPI service (needs ragqa[api])
    python -m ragqa.cli demo           # sample docs + two sample questions
"""

import argparse
import os
import sys

from ragqa.config import Settings
from ragqa.service import QAService

SAMPLE_DOCS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "examples", "sample_docs",
)


def _ingest_samples(service: QAService) -> None:
    for name in sorted(os.listdir(SAMPLE_DOCS_DIR)):
        if name.endswith(".txt"):
            service.ingest_file(os.path.join(SAMPLE_DOCS_DIR, name))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="ragqa")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Ingest files into the index")
    ingest.add_argument("paths", nargs="+")

    ask = sub.add_parser("ask", help="Ask a question over indexed documents")
    ask.add_argument("question")
    ask.add_argument("--roles", nargs="*", default=[])

    sub.add_parser("serve", help="Run the FastAPI service")
    sub.add_parser("demo", help="Ingest sample docs and answer two questions")

    args = parser.parse_args(argv)
    service = QAService(Settings())

    if args.command == "ingest":
        for path in args.paths:
            doc = service.ingest_file(path)
            print(f"indexed {doc.doc_id}: {doc.status}")
        return 0
    if args.command == "ask":
        print(service.ask(args.question, roles=args.roles).render())
        return 0
    if args.command == "serve":
        import uvicorn

        from ragqa.api import create_app

        uvicorn.run(
            create_app(service), host=service.settings.host, port=service.settings.port
        )
        return 0
    if args.command == "demo":
        _ingest_samples(service)
        for question in (
            "What is the refund window for standard orders?",
            "How many remote days per week are allowed?",
        ):
            print(f"\nQ: {question}")
            print(service.ask(question).render())
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
