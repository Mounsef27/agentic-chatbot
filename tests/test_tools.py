"""Tests for password generation and tool metadata."""

from __future__ import annotations

import string

import pytest

from agentic_chatbot.tools.password import generate_password
from agentic_chatbot.tools.search import build_search_tool


def test_generate_password_respects_length() -> None:
    """Password generation should create a secure password of the requested length."""
    password = generate_password(16)
    assert len(password) == 16
    allowed = set(string.ascii_letters + string.digits + "!@#$%&*")
    assert set(password).issubset(allowed)


def test_generate_password_rejects_invalid_lengths() -> None:
    """Passwords shorter than 8 or longer than 128 characters should be rejected."""
    with pytest.raises(ValueError):
        generate_password(7)
    with pytest.raises(ValueError):
        generate_password(129)


def test_search_tool_is_registered() -> None:
    """The search tool should expose a stable LangChain name."""
    tool = build_search_tool()
    assert tool.name
    assert tool.name == "tavily_search"
