"""Streaming helpers for graph responses."""

from __future__ import annotations

from collections.abc import Iterator

from langgraph.types import Command


def stream_response(chatbot: object, graph_input: dict | Command, config: dict) -> Iterator[str]:
    """Stream custom events from the compiled graph.

    Args:
        chatbot (object): Compiled LangGraph chatbot.
        graph_input (dict | Command): Graph input payload or resume command.
        config (dict): Runnable configuration including thread ID.

    Yields:
        str: Text chunks emitted by the graph.
    """
    for event in chatbot.stream(graph_input, config=config, stream_mode="custom"):
        if isinstance(event, str):
            yield event
