"""Ingestion end to end: docs land in the store, metadata store tracks them."""

from tests.conftest import REFUND_TEXT, REMOTE_TEXT


def test_ingest_text_indexes_chunks_and_metadata(service):
    doc = service.ingest_text("refund-policy", "Refund Policy", "refund-policy.txt", REFUND_TEXT)
    assert doc.status == "indexed"
    assert len(service.pipeline.store) > 0
    listed = {d.doc_id: d for d in service.list_documents()}
    assert listed["refund-policy"].status == "indexed"
    assert listed["refund-policy"].link == "doc://refund-policy"


def test_reingest_bumps_version_and_replaces_chunks(service):
    service.ingest_text("doc", "Doc", "doc.txt", REFUND_TEXT)
    first_count = len(service.pipeline.store)
    doc = service.ingest_text("doc", "Doc", "doc.txt", REMOTE_TEXT)
    assert doc.version == 2
    assert len(service.pipeline.store) == first_count  # replaced, not duplicated
    answer = service.ask("How many remote days per week are allowed?")
    assert not answer.abstained


def test_ingest_file_from_disk(service, tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("The office Wi-Fi password rotates on the first Monday of each month.")
    doc = service.ingest_file(str(path))
    assert doc.status == "indexed"
    assert doc.source == str(path)
