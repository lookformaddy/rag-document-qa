"""FastAPI service: ingest document, ask question, list documents, health.

Run: ``python -m ragqa.cli serve`` or ``uvicorn ragqa.api:app``.
Requires ``pip install ragqa[api]`` (fastapi + uvicorn).
"""

from ragqa.config import Settings
from ragqa.service import QAService


def create_app(service=None):
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    svc = service or QAService()

    class IngestTextRequest(BaseModel):
        doc_id: str
        title: str
        source: str
        text: str
        allowed_roles: list = []

    class AskRequest(BaseModel):
        question: str
        roles: list = []
        top_k: int | None = None

    app = FastAPI(title="rag-document-qa", version="0.1.0")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "documents": len(svc.list_documents()),
            "chunks": len(svc.pipeline.store),
        }

    @app.post("/documents")
    def ingest_document(req: IngestTextRequest):
        doc = svc.ingest_text(
            doc_id=req.doc_id, title=req.title, source=req.source,
            text=req.text, allowed_roles=req.allowed_roles,
        )
        return {"doc_id": doc.doc_id, "status": doc.status, "version": doc.version}

    @app.get("/documents")
    def list_documents():
        return [
            {
                "doc_id": d.doc_id, "title": d.title, "source": d.source,
                "status": d.status, "version": d.version,
                "link": d.link, "allowed_roles": list(d.allowed_roles),
            }
            for d in svc.list_documents()
        ]

    @app.post("/ask")
    def ask(req: AskRequest):
        if not req.question.strip():
            raise HTTPException(status_code=400, detail="question must not be empty")
        answer = svc.ask(req.question, roles=req.roles, top_k=req.top_k)
        return {
            "question": answer.question,
            "answer": answer.text,
            "abstained": answer.abstained,
            "citations": [
                {
                    "source": c.source, "doc_id": c.doc_id, "page": c.page,
                    "excerpt": c.excerpt, "rendered": c.render(),
                }
                for c in answer.citations
            ],
            "retrieved": [
                {"chunk_id": r.chunk.chunk_id, "doc_id": r.chunk.doc_id,
                 "score": round(r.score, 4)}
                for r in answer.retrieved
            ],
        }

    return app


app = create_app()
