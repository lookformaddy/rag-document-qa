"""Access control: restricted documents are invisible without the role."""

ADMIN_TEXT = (
    "The budget approval threshold is fifty thousand dollars. "
    "Requests above that amount require finance sign-off."
)


def test_restricted_doc_hidden_without_role(service):
    service.ingest_text(
        "admin-doc", "Admin Doc", "admin.txt", ADMIN_TEXT, allowed_roles=("admin",)
    )
    answer = service.ask("What is the budget approval threshold?", roles=[])
    assert answer.abstained


def test_restricted_doc_visible_with_role(service):
    service.ingest_text(
        "admin-doc", "Admin Doc", "admin.txt", ADMIN_TEXT, allowed_roles=("admin",)
    )
    answer = service.ask("What is the budget approval threshold?", roles=["admin"])
    assert not answer.abstained
    assert "fifty thousand" in answer.text


def test_unrestricted_doc_visible_to_everyone(seeded_service):
    answer = seeded_service.ask("What is the refund window for standard orders?", roles=[])
    assert not answer.abstained
