"""LangGraph node implementations and routing logic."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.runnables import Runnable
from langgraph.config import get_stream_writer
from langgraph.types import Command, interrupt

from agentic_chatbot.agents.state import ChatState
from agentic_chatbot.services.conversations import text_parts
from agentic_chatbot.services.llm import build_llm_with_tools
from agentic_chatbot.tools.password import password_tool
from agentic_chatbot.tools.search import build_search_tool, is_search_tool_call


def chat_node(state: ChatState, llm_with_tools: Runnable | None = None) -> dict[str, list[BaseMessage]]:
    """Generate an assistant reply from the current chat history.

    Args:
        state (ChatState): Current graph state.
        llm_with_tools (Runnable | None): Optional model instance with tools bound.

    Returns:
        dict[str, list[BaseMessage]]: A single message update containing the assistant reply.

    Raises:
        RuntimeError: If the model returns no content.
    """
    writer = get_stream_writer()
    model = llm_with_tools or build_llm_with_tools(build_search_tool(), password_tool)
    response: BaseMessage | None = None

    for chunk in model.stream(state["messages"]):
        for text in text_parts(chunk.content):
            if writer is not None:
                writer(text)
        response = chunk if response is None else response + chunk

    if response is None:
        raise RuntimeError("Gemini returned an empty response.")

    return {"messages": [response]}


def approval_node(
    state: ChatState,
    interrupt_func: Callable[[dict[str, Any]], Any] = interrupt,
    search_tool: Any | None = None,
) -> Command:
    """Handle Human-in-the-Loop approval for tool calls.

    Args:
        state (ChatState): The current graph state.
        interrupt_func (Callable[[dict[str, Any]], Any]): Interrupt callback used by tests.
        search_tool (Any | None): Tool instance used to identify search requests.

    Returns:
        Command: A LangGraph command to continue execution or resume the chat loop.
    """
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []
    resolved_search_tool = search_tool or build_search_tool()

    search_calls = [
        call for call in tool_calls if is_search_tool_call(call) or call.get("name") == resolved_search_tool.name
    ]
    if not search_calls:
        return Command(goto="tools")

    approved = interrupt_func(
        {
            "action": "approve_tool_calls",
            "tool_calls": [{"name": call["name"], "args": call["args"]} for call in search_calls],
            "message": "Do you approve the searches? (yes/no)",
        }
    )

    if approved:
        return Command(goto="tools")

    rejected_messages = [
        ToolMessage(
            content="Tool call rejected by user. Skipping tool execution.",
            tool_call_id=call.get("id"),
        )
        for call in tool_calls
    ]
    return Command(update={"messages": rejected_messages}, goto="chat_node")


def route_after_chat(state: ChatState) -> Literal["approval", "__end__"]:
    """Decide whether the graph should pause for approval.

    Args:
        state (ChatState): Current graph state.

    Returns:
        Literal["approval", "__end__"]: Approval route when tool calls exist, otherwise end.
    """
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "approval"
    return "__end__"
