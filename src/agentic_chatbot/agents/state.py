"""Shared state definition for the chatbot graph."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ChatState(TypedDict):
    """State for the LangGraph conversation graph.

    Attributes:
        messages (list[BaseMessage]): The ordered message history.
    """

    messages: Annotated[list[BaseMessage], add_messages]
