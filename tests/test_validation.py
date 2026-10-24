"""Input validation - spec cases: blank, very short, malformed."""

import pytest


@pytest.fixture(scope="module")
def validate():
    from main import validate
    return validate


@pytest.mark.parametrize("text", ["", "   ", "\n\t  ", None])
def test_blank_is_rejected(validate, text):
    cleaned, error = validate(text)
    assert cleaned is None
    assert error


@pytest.mark.parametrize("text", ["IE", "vpn", "help me"])
def test_too_short_is_rejected(validate, text):
    cleaned, error = validate(text)
    assert cleaned is None
    assert "short" in error.lower()


def test_minimum_length_is_accepted(validate):
    cleaned, error = validate("vpn keeps dropping")
    assert error is None
    assert cleaned == "vpn keeps dropping"


def test_overlong_input_is_truncated(validate):
    cleaned, error = validate("word " * 5000)
    assert error is None
    assert len(cleaned.split()) == 2000


@pytest.mark.parametrize("text", [
    "ticket with emoji 🔥🔥 and unicode ünïcödé",
    "<script>alert('xss')</script> cannot login",
    "'; DROP TABLE tickets; -- cannot open link",
    "\x00\x01 null bytes in the description",
])
def test_malformed_input_does_not_crash(validate, text):
    """Hostile input must be handled, not raise."""
    cleaned, error = validate(text)
    assert cleaned is not None or error is not None