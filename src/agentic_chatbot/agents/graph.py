"""Graph assembly for the chatbot workflow."""

from __future__ import annotations

from functools import partial

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agentic_chatbot.agents.nodes import approval_node, chat_node, route_after_chat
from agentic_chatbot.agents.state import ChatState
from agentic_chatbot.persistence.checkpoint import get_checkpointer
from agentic_chatbot.services.llm import build_llm_with_tools
from agentic_chatbot.tools.password import password_tool
from agentic_chatbot.tools.search import build_search_tool


def build_graph(
    *,
    checkpointer: object | None = None,
    search_tool: object | None = None,
    llm_with_tools: object | None = None,
) -> object:
    """Build the LangGraph chatbot workflow.

    Args:
        checkpointer (object | None): Optional SQL checkpoint backend.
        search_tool (object | None): Optional search tool override.
        llm_with_tools (object | None): Optional bound LLM tool wrapper.

    Returns:
        object: A compiled LangGraph application.
    """
    resolved_search_tool = search_tool or build_search_tool()
    resolved_llm = llm_with_tools or build_llm_with_tools(resolved_search_tool, password_tool)
    graph = StateGraph(ChatState)
    graph.add_node("chat_node", partial(chat_node, llm_with_tools=resolved_llm))
    graph.add_node("approval_node", partial(approval_node, search_tool=resolved_search_tool))
    graph.add_node("tools", ToolNode(tools=[resolved_search_tool, password_tool]))
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", route_after_chat, {"approval": "approval_node", "__end__": END})
    graph.add_edge("tools", "chat_node")
    return graph.compile(checkpointer=checkpointer or get_checkpointer())


def get_chatbot() -> object:
    """Return the cached chatbot graph instance.

    Returns:
        object: The compiled graph singleton.
    """
    return build_graph()
