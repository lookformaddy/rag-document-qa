"""API: health, ingest, list, ask — skipped when FastAPI is absent."""

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from ragqa.api import create_app  # noqa: E402
from ragqa.config import Settings  # noqa: E402
from ragqa.service import QAService  # noqa: E402
from tests.conftest import REFUND_TEXT  # noqa: E402


@pytest.fixture
def client():
    service = QAService(Settings())
    service.ingest_text("refund-policy", "Refund Policy", "refund-policy.txt", REFUND_TEXT)
    return TestClient(create_app(service))


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_ingest_list_ask_flow(client):
    resp = client.post("/documents", json={
        "doc_id": "runbook", "title": "Runbook", "source": "runbook.txt",
        "text": "Deployments run every Tuesday at 2:00 PM.",
    })
    assert resp.json()["status"] == "indexed"
    docs = client.get("/documents").json()
    assert {d["doc_id"] for d in docs} == {"refund-policy", "runbook"}
    answer = client.post("/ask", json={"question": "When do deployments run?"}).json()
    assert not answer["abstained"]
    assert answer["citations"]


def test_ask_empty_question_rejected(client):
    assert client.post("/ask", json={"question": "  "}).status_code == 400
