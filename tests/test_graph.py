"""Graph routing and approval tests."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from agentic_chatbot.agents.nodes import approval_node, route_after_chat
from agentic_chatbot.tools.search import build_search_tool


def test_route_after_chat_requires_approval() -> None:
    """AI messages with tool calls should pause for approval before execution."""
    tool = build_search_tool()
    state = {
        "messages": [
            AIMessage(
                content="I need to search the web.",
                tool_calls=[{"id": "call_1", "name": tool.name, "args": {"query": "python"}}],
            )
        ]
    }
    assert route_after_chat(state) == "approval"


def test_approval_node_handles_approve_and_reject() -> None:
    """Approval should resume execution or reject the tool call safely."""
    tool = build_search_tool()
    state = {
        "messages": [
            AIMessage(
                content="I need a search.",
                tool_calls=[{"id": "call_1", "name": tool.name, "args": {"query": "python"}}],
            )
        ]
    }

    approved = approval_node(state, interrupt_func=lambda value: True, search_tool=tool)
    assert approved.goto == "tools"

    rejected = approval_node(state, interrupt_func=lambda value: False, search_tool=tool)
    assert rejected.goto == "chat_node"
    assert rejected.update["messages"][0].content == "Tool call rejected by user. Skipping tool execution."
