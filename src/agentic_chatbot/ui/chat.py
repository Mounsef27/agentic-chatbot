"""Helpers for rendering chat history and streaming responses."""

from __future__ import annotations

import logging

import httpx
import streamlit as st
from google.genai.errors import APIError

from agentic_chatbot.config.settings import get_settings
from agentic_chatbot.services.streaming import stream_response

logger = logging.getLogger(__name__)


def render_conversation_history(messages: list[dict[str, str]]) -> None:
    """Render persisted conversation history in the chat container.

    Args:
        messages (list[dict[str, str]]): Message entries for the current thread.
    """
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def render_user_message(prompt: str) -> None:
    """Render a user prompt before execution.

    Args:
        prompt (str): User input.
    """
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)


def execute_graph(chatbot: object, config: dict, graph_input: dict | object) -> bool:
    """Execute the graph and stream results into the UI.

    Args:
        chatbot (object): Compiled LangGraph chatbot.
        config (dict): Runnable config including thread ID.
        graph_input (dict | object): Input for the graph or a resume command.

    Returns:
        bool: True when the graph executed without a terminal API or transport error.
    """
    response: str | None = None
    with st.chat_message("assistant"), st.status("Running the agent...", expanded=False) as status:
        try:
            response = st.write_stream(stream_response(chatbot, graph_input, config))
        except APIError as error:
            status.update(label="Gemini request failed", state="error")
            detail = redact_api_key(str(error))
            st.error(f"Google API error ({error.code or error.status}): {detail}")
            return False
        except httpx.TimeoutException:
            status.update(label="Gemini did not respond in time", state="error")
            st.error("The request exceeded the 60-second timeout.")
            return False
        except httpx.HTTPError as error:
            status.update(label="Unable to reach Google", state="error")
            st.error(redact_api_key(str(error)))
            return False
        snapshot = chatbot.get_state(config)
        if snapshot.interrupts:
            status.update(label="Human approval required", state="complete")
        else:
            status.update(label="Execution complete", state="complete")

    if isinstance(response, str) and response.strip():
        st.session_state.messages.append({"role": "assistant", "content": response})

    return True


def redact_api_key(message: str) -> str:
    """Redact the configured Google API key from user-facing text.

    Args:
        message (str): Message potentially containing a key.

    Returns:
        str: Sanitized message safe for display.
    """
    key = get_settings().google_api_key
    if key:
        return message.replace(key, "[hidden]")
    return message
