"""Approval workflow for tool requests."""

from __future__ import annotations

from typing import Any

import streamlit as st
from langgraph.types import Command


def render_approval_banner(snapshot: Any, *, execute_graph: Any) -> bool:
    """Render the Human-in-the-Loop approval panel and resume the graph.

    Args:
        snapshot (Any): LangGraph snapshot containing pending interrupts.
        execute_graph (Any): Callback that runs the graph with a command payload.

    Returns:
        bool: True when an approval decision is rendered and handled.
    """
    interrupts = getattr(snapshot, "interrupts", None) or []
    if not interrupts:
        return False

    interrupt_data = interrupts[0].value
    st.warning(interrupt_data.get("message", "Tool execution requires approval."))
    for call in interrupt_data.get("tool_calls", []):
        st.write(f"**Tool:** {call['name']}")
        st.json(call["args"])

    col1, col2 = st.columns(2)
    with col1:
        approve = st.button("Approve", type="primary", use_container_width=True)
    with col2:
        reject = st.button("Reject", use_container_width=True)

    if approve or reject:
        success = execute_graph(Command(resume=bool(approve)))
        if success:
            st.rerun()
        return True

    st.stop()
    return True
