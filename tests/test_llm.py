"""LLM layer - response parsing and result merging, without network."""

import json

import pytest


@pytest.fixture
def canned_response():
    return json.dumps({
        "steps": ["Clear the browser cache.", "Add the URL to trusted sites."],
        "explanation": "Past incidents were resolved by browser config.",
        "evidence_quality": "moderate",
        "caveats": ["The new ticket omits the error message."],
    })


def test_llm_success_path(advisor, monkeypatch, canned_response):
    import advisor as advisor_module

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(
        advisor_module, "call_gemini", lambda prompt, key: canned_response
    )

    result = advisor.advise(
        "user cannot open link in internet explorer getting error",
        use_llm=True,
    )

    assert result["source"] == "llm"
    assert len(result["steps"]) == 2
    assert result["evidence_quality"] == "moderate"
    assert any("omits the error message" in w for w in result["warnings"])
    assert any("Not a verified fix" in w for w in result["warnings"])


def test_llm_failure_falls_back_to_retrieval(advisor, monkeypatch):
    import advisor as advisor_module

    def boom(prompt, key):
        raise RuntimeError("429 rate limit exceeded")

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(advisor_module, "call_gemini", boom)

    result = advisor.advise(
        "user cannot open link in internet explorer getting error",
        use_llm=True,
    )

    assert result["source"] == "retrieval"
    assert any("rate limit" in w for w in result["warnings"])


def test_unparseable_response_falls_back(advisor, monkeypatch):
    import advisor as advisor_module

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(
        advisor_module, "call_gemini",
        lambda prompt, key: "I'm sorry, I can't help with that."
    )

    result = advisor.advise(
        "user cannot open link in internet explorer getting error",
        use_llm=True,
    )

    assert result["source"] == "retrieval"


def test_json_wrapped_in_prose_is_recovered():
    from advisor import parse_response

    text = 'Here you go:\n```json\n{"steps": ["a"], "explanation": "b"}\n```'
    parsed = parse_response(text)

    assert parsed["steps"] == ["a"]