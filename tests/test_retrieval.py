"""Retrieval: the chunk that contains the answer must win."""

from ragqa.retrieval import keyword_score


def test_right_chunk_wins(seeded_service):
    results = seeded_service.retriever.retrieve(
        "What is the refund window for standard orders?"
    )
    assert results, "expected at least one relevant chunk"
    assert results[0].chunk.doc_id == "refund-policy"


def test_remote_question_retrieves_remote_doc(seeded_service):
    results = seeded_service.retriever.retrieve(
        "How many remote days per week are allowed?"
    )
    assert results and results[0].chunk.doc_id == "remote-policy"


def test_keyword_score_rewards_term_coverage():
    assert keyword_score("refund window standard orders", "refund within thirty days") > 0
    assert keyword_score("refund window", "unrelated deployment schedule") == 0.0


def test_retrieval_is_ranked(seeded_service):
    results = seeded_service.retriever.retrieve("refund standard orders")
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
