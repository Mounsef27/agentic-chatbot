"""Conversation formatting tests."""

from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from agentic_chatbot.services.conversations import conversation_messages


class FakeChatbot:
    """Simple chatbot stub used to test saved state formatting."""

    def __init__(self, messages):
        self._messages = messages

    def get_state(self, config):
        return SimpleNamespace(values={"messages": self._messages})


def test_conversation_messages_formatting() -> None:
    """Human and AI messages should be rendered in a UI-friendly format."""
    chatbot = FakeChatbot(
        [
            HumanMessage(content="hello"),
            AIMessage(content="hi there"),
        ]
    )

    history = conversation_messages(chatbot, "thread-123")
    assert history == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
