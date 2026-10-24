"""End-to-end - spec cases: known issue, no similar tickets, LLM down."""

import pytest


def test_known_issue_retrieves_incidents(advisor):
    result = advisor.advise(
        "user cannot open link in internet explorer getting error message",
        use_llm=False,
    )
    assert result["similar"]
    assert result["similar"][0]["similarity"] >= 0.45


def test_unrelated_ticket_returns_only_weak_leads(advisor):
    """
    No VPN tickets exist here. Retrieval still returns ranked leads -
    on-call, a suppressed relevant hit costs more than a dismissed
    irrelevant one - but every hit must be marked weak and none may
    become recommendation material.
    """
    result = advisor.advise(
        "VPN disconnects every few minutes after windows update",
        use_llm=False,
    )
    assert result["similar"]
    assert all(hit["weak"] for hit in result["similar"])
    assert any("weak" in w for w in result["warnings"])


def test_steps_never_come_from_weak_matches(advisor):
    """
    A 30% match once produced bank-account-closure advice for a VPN
    question, printed under a RECOMMENDED RESOLUTION heading. Weak
    hits are leads for a human to judge, never recommendation input.
    """
    result = advisor.advise(
        "VPN disconnects every few minutes after windows update",
        use_llm=False,
    )
    assert result["steps"] == []


def test_weak_flag_matches_floor(advisor):
    """The flag is the contract Phase 12's API will expose."""
    from retriever import SIMILARITY_FLOOR

    result = advisor.advise(
        "cannot open the application link in the browser",
        use_llm=False,
    )
    assert result["similar"]

    for hit in result["similar"]:
        assert hit["weak"] == (hit["similarity"] < SIMILARITY_FLOOR)


def test_llm_skipped_when_no_evidence(advisor, monkeypatch):
    """Structural grounding guard: no evidence means no generation."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-never-used")

    result = advisor.advise(
        "VPN disconnects every few minutes after windows update",
        use_llm=True,
    )
    assert result["source"] == "retrieval"
    assert any("strong enough" in w for w in result["warnings"])


def test_missing_api_key_falls_back(advisor):
    """The no_network fixture has already removed the key."""
    result = advisor.advise(
        "user cannot open link in internet explorer getting error",
        use_llm=True,
    )
    assert result["source"] == "retrieval"
    assert any("GEMINI_API_KEY not set" in w for w in result["warnings"])


def test_short_input_is_rejected_before_any_work(advisor):
    result = advisor.advise("IE", use_llm=False)
    assert result.get("error")
    assert result["similar"] == []


def test_result_shape_is_stable(advisor):
    """Phase 12's API contract depends on these keys existing."""
    result = advisor.advise(
        "cannot login to the portal, getting an error",
        use_llm=False,
    )
    for key in ("category", "priority", "similar", "steps",
                "explanation", "source", "warnings"):
        assert key in result