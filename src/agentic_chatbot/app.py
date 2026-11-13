import os
import sqlite3
from typing import Annotated, Iterator, TypedDict
from uuid import uuid4

import httpx
import streamlit as st
from dotenv import load_dotenv
from google.genai.errors import APIError
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver

conn = sqlite3.connect("chatbot_state.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=1.0,  # Gemini 3.0+ defaults to 1.0
    max_tokens=None,
    timeout=60,
    max_retries=1,
    google_api_key=google_api_key,
)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState) -> dict[str, list[BaseMessage]]:
    writer = get_stream_writer()
    response: BaseMessage | None = None
    for chunk in llm.stream(state["messages"]):
        for text in text_parts(chunk.content):
            writer(text)
        response = chunk if response is None else response + chunk

    if response is None:
        raise RuntimeError("Gemini returned an empty response.")

    return {"messages": [response]}


def text_parts(content: object) -> Iterator[str]:
    if isinstance(content, str):
        if content:
            yield content
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                yield block["text"]


def conversation_messages(thread_id: str) -> list[dict[str, str]]:
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    state = get_chatbot().get_state(config)
    messages = state.values.get("messages", [])
    return [
        {
            "role": "user" if message.type == "human" else "assistant",
            "content": "".join(text_parts(message.content)),
        }
        for message in messages
        if message.type in ("human", "ai")
    ]


def saved_conversations() -> list[tuple[str, str]]:
    conversations: list[tuple[str, str]] = []
    seen_thread_ids: set[str] = set()

    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config["configurable"]["thread_id"]
        if thread_id in seen_thread_ids:
            continue
        seen_thread_ids.add(thread_id)

        messages = checkpoint.checkpoint.get("channel_values", {}).get("messages", [])
        first_user_message = next(
            (message for message in messages if isinstance(message, HumanMessage)),
            None,
        )
        title = (
            "".join(text_parts(first_user_message.content)).strip()
            if first_user_message is not None
            else ""
        )
        if not title:
            title = f"Conversation {thread_id[:8]}"
        elif len(title) > 40:
            title = f"{title[:37]}..."
        conversations.append((thread_id, title))

    return conversations


def redact_api_key(message: str) -> str:
    if google_api_key:
        return message.replace(google_api_key, "[clé masquée]")
    return message


def stream_response(prompt: str, config: RunnableConfig) -> Iterator[str]:
    events = get_chatbot().stream(
        {"messages": [HumanMessage(content=prompt)]},
        config=config,
        stream_mode="custom",
    )
    for event in events:
        if isinstance(event, str):
            yield event


@st.cache_resource(show_spinner=False)
def get_chatbot():
    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_edge(START, "chat_node")
    graph.add_edge("chat_node", END)
    return graph.compile(checkpointer=checkpointer)


def main() -> None:
    st.set_page_config(page_title="Agentic Chatbot", page_icon="💬")
    st.title("Agentic Chatbot")
    st.caption("Discutez avec votre assistant IA, propulsé par Google Gemini.")

    if not google_api_key:
        st.error(
            "Clé API Google introuvable. Ajoutez GEMINI_API_KEY ou "
            "GOOGLE_API_KEY dans le fichier .env."
        )
        st.stop()

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid4())
        st.session_state.messages = []

    with st.sidebar:
        st.header("Conversations")
        if st.button("Nouvelle conversation", use_container_width=True):
            st.session_state.thread_id = str(uuid4())
            st.session_state.messages = []
            st.rerun()

        for thread_id, title in saved_conversations():
            if st.button(
                title,
                key=f"conversation-{thread_id}",
                type=(
                    "primary"
                    if thread_id == st.session_state.thread_id
                    else "secondary"
                ),
                use_container_width=True,
            ):
                st.session_state.thread_id = thread_id
                st.session_state.messages = conversation_messages(thread_id)
                st.rerun()

        st.caption("Modèle : gemini-3.5-flash-lite")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Écrivez votre message...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        config: RunnableConfig = {
            "configurable": {"thread_id": st.session_state.thread_id}
        }
        response: str | None = None
        with st.chat_message("assistant"):
            with st.status("Connexion à Gemini...", expanded=False) as status:
                try:
                    response = st.write_stream(stream_response(prompt, config))
                except APIError as error:
                    status.update(label="Échec de la requête Gemini", state="error")
                    detail = redact_api_key(error.message or str(error))
                    st.error(
                        f"Erreur API Google ({error.code or error.status}): {detail}"
                    )
                except httpx.TimeoutException:
                    status.update(label="Gemini ne répond pas", state="error")
                    st.error(
                        "La requête a dépassé le délai de 60 secondes. "
                        "Vérifiez votre connexion et réessayez."
                    )
                except httpx.HTTPError as error:
                    status.update(label="Connexion à Google impossible", state="error")
                    detail = redact_api_key(str(error))
                    st.error(f"Erreur réseau pendant l'appel à Gemini : {detail}")
                else:
                    status.update(label="Réponse reçue", state="complete")

        if response is not None:
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )
            st.rerun()


if __name__ == "__main__":
    main()
