"""Shared pytest fixtures."""

import pytest


@pytest.fixture(autouse=True)
def mock_tavily_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a fake Tavily API key for unit tests."""
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-api-key")