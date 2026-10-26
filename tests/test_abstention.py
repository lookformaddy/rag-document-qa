"""Abstention: no relevant retrieval means an explicit non-answer."""

from ragqa.service import ABSTENTION_TEXT


def test_abstains_when_nothing_relevant(seeded_service):
    answer = seeded_service.ask("What is the capital of France?")
    assert answer.abstained
    assert answer.text == ABSTENTION_TEXT
    assert answer.citations == []


def test_abstains_on_empty_index(service):
    answer = service.ask("What is the refund window?")
    assert answer.abstained
