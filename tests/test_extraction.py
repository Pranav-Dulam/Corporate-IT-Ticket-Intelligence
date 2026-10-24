"""Step extraction - the heuristic fallback used when the LLM is down."""

import pytest


@pytest.fixture(scope="module")
def extraction():
    import recommender
    return recommender


def test_actionable_step_is_kept(extraction):
    assert extraction.is_actionable(
        "please try to clear the browser cache and try to login"
    )


def test_instruction_phrased_as_question_is_kept(extraction):
    """
    Support engineers write instructions as polite questions. An
    earlier version rejected anything ending in '?' and threw away
    the single most useful line in the corpus.
    """
    assert extraction.is_actionable(
        "can you check if the url is added to the trusted sites?"
    )


@pytest.mark.parametrize("sentence", [
    "we will check and get back to you",
    "kindly be informed that your ticket has been assigned",
    "please consider this ticket as closed",
    "we are waiting for your kind feedback",
    "ok",
])
def test_non_actionable_is_rejected(extraction, sentence):
    assert not extraction.is_actionable(sentence)


def test_steps_are_deduplicated(extraction):
    hits = [
        {
            "ticket_id": "A",
            "similarity": 0.9,
            "resolution": "clear the browser cache. clear the browser cache.",
        },
        {
            "ticket_id": "B",
            "similarity": 0.8,
            "resolution": "Clear the browser cache!",
        },
    ]
    steps = extraction.extract_steps(hits)
    assert len(steps) == 1


def test_empty_resolutions_yield_no_steps(extraction):
    hits = [{"ticket_id": "A", "similarity": 0.9, "resolution": None}]
    assert extraction.extract_steps(hits) == []