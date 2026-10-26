"""Shared fixtures: a fresh offline service seeded with synthetic docs."""

import pytest

from ragqa.config import Settings
from ragqa.service import QAService

REFUND_TEXT = (
    "Standard orders may be refunded within thirty days of delivery, "
    "provided the items are unused and in their original packaging. "
    "Refunds are issued to the original payment method within five to "
    "seven business days of approval."
)
REMOTE_TEXT = (
    "Employees may work remotely up to three days per week, provided "
    "their manager approves the schedule in advance. Core collaboration "
    "hours are 10:00 AM to 3:00 PM."
)


@pytest.fixture
def service():
    return QAService(Settings())


@pytest.fixture
def seeded_service(service):
    service.ingest_text("refund-policy", "Refund Policy", "refund-policy.txt", REFUND_TEXT)
    service.ingest_text("remote-policy", "Remote Policy", "remote-policy.txt", REMOTE_TEXT)
    return service
