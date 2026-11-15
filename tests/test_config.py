"""Configuration validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_chatbot.config.settings import Settings


def test_settings_validate_requires_google_api_key() -> None:
    """The app should refuse configuration without a Google API key."""
    settings = Settings(google_api_key=None, tavily_api_key=None, db_path=Path("chatbot_state.db"))
    with pytest.raises(RuntimeError):
        settings.validate()
