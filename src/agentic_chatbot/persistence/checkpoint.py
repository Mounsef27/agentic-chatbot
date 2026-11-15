"""SQLite checkpoint configuration for LangGraph."""

from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from agentic_chatbot.config.settings import get_settings


@lru_cache(maxsize=1)
def get_checkpointer(db_path: str | Path | None = None) -> SqliteSaver:
    """Return a cached SQLite-backed checkpointer for LangGraph.

    Args:
        db_path (str | Path | None): Optional custom checkpoint database path.

    Returns:
        SqliteSaver: A SQLite checkpointer shared across the app lifecycle.
    """
    resolved = Path(db_path) if db_path is not None else get_settings().db_path
    connection = sqlite3.connect(str(resolved), check_same_thread=False)
    return SqliteSaver(connection)
