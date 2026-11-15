"""Streamlit entry point for the agentic chatbot."""

from __future__ import annotations

import logging
from uuid import uuid4

import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from agentic_chatbot.agents.graph import get_chatbot
from agentic_chatbot.config.settings import get_settings
from agentic_chatbot.persistence.checkpoint import get_checkpointer
from agentic_chatbot.ui.approval import render_approval_banner
from agentic_chatbot.ui.chat import execute_graph, render_conversation_history
from agentic_chatbot.ui.sidebar import render_sidebar

logger = logging.getLogger(__name__)


def main() -> None:
    """Render the Streamlit UI and coordinate the chatbot workflow."""
    st.set_page_config(page_title="Agentic Chatbot", page_icon="💬")
    st.title("Agentic Chatbot")
    st.caption("Chat with your AI assistant powered by Google Gemini.")

    settings = get_settings()
    if not settings.google_api_key:
        st.error("Google API key not found. Add GEMINI_API_KEY or GOOGLE_API_KEY to your environment or .env file.")
        st.stop()

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid4())
        st.session_state.messages = []

    chatbot = get_chatbot()
    render_sidebar(chatbot, get_checkpointer())
    render_conversation_history(st.session_state.get("messages", []))

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    def execute_graph_wrapper(graph_input: dict | Command) -> bool:
        return execute_graph(chatbot, config, graph_input)

    snapshot = chatbot.get_state(config)
    if snapshot.interrupts:
        render_approval_banner(snapshot, execute_graph=execute_graph_wrapper)

    prompt = st.chat_input("Type your message...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        success = execute_graph_wrapper({"messages": [HumanMessage(content=prompt)]})
        if success:
            st.rerun()


if __name__ == "__main__":
    main()
