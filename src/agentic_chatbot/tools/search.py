"""Search integration for Tavily."""

from __future__ import annotations

from typing import Any

from langchain_tavily import TavilySearch

from agentic_chatbot.config.settings import get_settings


def build_search_tool() -> TavilySearch:
    """Create a Tavily search tool bound to the configured API key.

    Returns:
        TavilySearch: A configured Tavily search tool instance.
    """
    settings = get_settings()
    return TavilySearch(
        api_key=settings.tavily_api_key,
        max_results=5,
        topic="general",
        search_depth="advanced",
    )


def is_search_tool_call(tool_call: dict[str, Any]) -> bool:
    """Return whether a tool call targets the search tool.

    Args:
        tool_call (dict[str, Any]): A LangChain-style tool call dictionary.

    Returns:
        bool: True when the tool call is for the Tavily search tool.
    """
    search_tool = build_search_tool()
    return tool_call.get("name") == search_tool.name
