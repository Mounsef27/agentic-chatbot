"""Google Gemini and tool binding helpers."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI

from agentic_chatbot.config.settings import get_settings


def build_llm() -> ChatGoogleGenerativeAI:
    """Create a Google Gemini model client.

    Returns:
        ChatGoogleGenerativeAI: An initialized Gemini chat model.
    """
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.model_name,
        temperature=1.0,
        max_tokens=None,
        timeout=60,
        max_retries=1,
        google_api_key=settings.google_api_key,
    )


def build_llm_with_tools(*tools: BaseTool | Any) -> Any:
    """Bind a list of tools to the configured Gemini model.

    Args:
        *tools: One or more LangChain tools to attach to the model.

    Returns:
        Any: A model instance configured with tool binding.
    """
    llm = build_llm()
    return llm.bind_tools(list(tools))
