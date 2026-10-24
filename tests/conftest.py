"""
Shared fixtures.

The embedding index is loaded once per session, not per test - building
it takes ~2s and nothing in the suite mutates it.

No test may reach the network. Tests that touch the advisor either run
with use_llm=False or have GEMINI_API_KEY removed from the environment.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend/python"))


@pytest.fixture(scope="session")
def retriever():
    from retriever import INDEX_DIR, TicketRetriever

    marker = INDEX_DIR / "retriever_embedding.joblib"

    if not marker.exists():
        pytest.skip(
            "No saved index. Run: python3 main.py --rebuild-index 'test'"
        )

    return TicketRetriever.load(backend="embedding")


@pytest.fixture(scope="session")
def advisor(retriever):
    from advisor import Advisor
    return Advisor(retriever=retriever)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """
    Belt and braces: strip the API key from every test's environment so
    an accidental LLM call degrades instead of hitting the network.
    Tests that specifically exercise the LLM path set it back.
    """
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)