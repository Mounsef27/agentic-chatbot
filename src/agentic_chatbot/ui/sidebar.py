"""Sidebar controls for conversation management."""

from __future__ import annotations

import streamlit as st

from agentic_chatbot.services.conversations import conversation_messages, saved_conversations


def render_sidebar(chatbot: object, checkpointer: object) -> None:
    """Render the sidebar and available conversation threads.

    Args:
        chatbot (object): Compiled LangGraph chatbot.
        checkpointer (object): SQLite checkpointer used to list conversations.
    """
    with st.sidebar:
        st.header("Conversations")
        if st.button("New conversation", use_container_width=True):
            st.session_state.thread_id = str(__import__("uuid").uuid4())
            st.session_state.messages = []
            st.rerun()

        for thread_id, title in saved_conversations(checkpointer):
            if st.button(
                title,
                key=f"conversation-{thread_id}",
                type="primary" if thread_id == st.session_state.thread_id else "secondary",
                use_container_width=True,
            ):
                st.session_state.thread_id = thread_id
                st.session_state.messages = conversation_messages(chatbot, thread_id)
                st.rerun()

        st.caption("Model: gemini-3.5-flash-lite")
