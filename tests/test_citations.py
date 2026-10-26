"""Citations: answers quote retrieved text and point at its source."""

from tests.conftest import REFUND_TEXT


def test_answer_carries_citation_to_source_doc(seeded_service):
    answer = seeded_service.ask("What is the refund window for standard orders?")
    assert not answer.abstained
    assert "thirty" in answer.text
    assert answer.citations
    assert answer.citations[0].doc_id == "refund-policy"
    assert answer.citations[0].source == "refund-policy.txt"
    assert answer.citations[0].excerpt
    rendered = answer.render()
    assert "Sources:" in rendered
    assert "refund-policy.txt" in rendered


def test_citation_excerpt_comes_from_chunk_text(seeded_service):
    answer = seeded_service.ask("What is the refund window for standard orders?")
    for citation in answer.citations:
        assert citation.excerpt in REFUND_TEXT or citation.excerpt[:40] in REFUND_TEXT
