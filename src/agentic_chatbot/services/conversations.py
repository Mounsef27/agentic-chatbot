"""Conversation history helpers for storing and reading thread state."""

from __future__ import annotations

from collections.abc import Iterator

from langchain_core.messages import HumanMessage


def text_parts(content: object) -> Iterator[str]:
    """Yield text fragments from message content dictionaries.

    Args:
        content (object): Message content from LangChain.

    Yields:
        str: Extracted text values.
    """
    if isinstance(content, str):
        if content:
            yield content
        return

    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                yield block["text"]


def conversation_messages(chatbot: object, thread_id: str) -> list[dict[str, str]]:
    """Read the visible conversation history for one thread.

    Args:
        chatbot (object): Compiled LangGraph chatbot instance.
        thread_id (str): Thread identifier to read.

    Returns:
        list[dict[str, str]]: User and assistant messages formatted for UI rendering.
    """
    config = {"configurable": {"thread_id": thread_id}}
    state = chatbot.get_state(config)
    messages = state.values.get("messages", [])
    return [
        {
            "role": "user" if message.type == "human" else "assistant",
            "content": "".join(text_parts(message.content)),
        }
        for message in messages
        if message.type in ("human", "ai")
    ]


def saved_conversations(checkpointer: object) -> list[tuple[str, str]]:
    """List saved conversation titles from the SQLite checkpoint database.

    Args:
        checkpointer (object): LangGraph SQLite checkpointer instance.

    Returns:
        list[tuple[str, str]]: Pairs of thread IDs and conversation titles.
    """
    conversations: list[tuple[str, str]] = []
    seen_thread_ids: set[str] = set()

    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config["configurable"]["thread_id"]
        if thread_id in seen_thread_ids:
            continue
        seen_thread_ids.add(thread_id)

        messages = checkpoint.checkpoint.get("channel_values", {}).get("messages", [])
        first_user_message = next((message for message in messages if isinstance(message, HumanMessage)), None)
        title = "".join(text_parts(first_user_message.content)).strip() if first_user_message is not None else ""
        if not title:
            title = f"Conversation {thread_id[:8]}"
        elif len(title) > 40:
            title = f"{title[:37]}..."
        conversations.append((thread_id, title))

    return conversations
