"""QAService: the query pipeline end to end, plus document metadata.

ask(): embed question -> retrieve -> build context -> generate -> Answer
with citations and document links. When nothing relevant is retrieved, the
answer abstains explicitly instead of inventing one — the single most
important reliability behavior of a RAG system.
"""

from ragqa.embeddings import get_embedding_provider
from ragqa.generation import build_context, get_answerer
from ragqa.ingestion.pipeline import IngestionPipeline
from ragqa.models import Answer
from ragqa.retrieval import Retriever

ABSTENTION_TEXT = (
    "I couldn't find a relevant answer in the indexed documents. "
    "Try rephrasing the question, or add the document that covers it."
)


class QAService:
    def __init__(self, settings=None, pipeline=None, answerer=None):
        from ragqa.config import Settings

        self.settings = settings or Settings()
        self.pipeline = pipeline or IngestionPipeline(settings=self.settings)
        self.answerer = answerer or get_answerer(self.settings)
        self.retriever = Retriever(
            self.pipeline.embedder, self.pipeline.store,
            self.settings, documents=self.pipeline.documents,
        )

    # -- ingestion passthroughs (the metadata store lives on the pipeline)
    def ingest_text(self, *args, **kwargs):
        return self.pipeline.ingest_text(*args, **kwargs)

    def ingest_file(self, *args, **kwargs):
        return self.pipeline.ingest_file(*args, **kwargs)

    def list_documents(self):
        return self.pipeline.list_documents()

    # -- query pipeline
    def ask(self, question: str, roles=(), top_k=None) -> Answer:
        results = self.retriever.retrieve(question, top_k=top_k, roles=roles)
        if not results:
            return Answer(
                question=question, text=ABSTENTION_TEXT,
                citations=[], retrieved=[], abstained=True,
            )
        context, citations = build_context(results)
        text = self.answerer.generate(question, context, citations)
        if not text.strip():
            return Answer(
                question=question, text=ABSTENTION_TEXT,
                citations=[], retrieved=results, abstained=True,
            )
        # Only cite the sources whose chunks actually fed the answer.
        return Answer(
            question=question, text=text,
            citations=citations, retrieved=results, abstained=False,
        )
