# Agentic Chatbot

A modular Streamlit application that combines Google Gemini, LangGraph, SQLite persistence, and human approval for tool execution.

## Overview

This project implements a conversational agent that can:

- answer user questions with a Gemini-backed LLM,
- perform web search through Tavily when needed,
- generate secure passwords on demand,
- store conversation state across sessions with LangGraph checkpointing,
- pause before executing search tools and wait for explicit user approval.

## Key Features

- LangGraph state-machine workflow with a clean node structure
- Google Gemini model integration via `langchain-google-genai`
- Tavily search tool integrated as a first-class agent tool
- Secure random password generation tool
- SQLite-based conversation persistence with `SqliteSaver`
- Streamlit UI with side-bar conversation history and live streaming
- Human-in-the-loop approval flow for search operations
- Lightweight pytest coverage for graph routing and tool behavior

## Technology Stack

- Python 3.13+
- Streamlit
- LangChain
- LangGraph
- Google Gemini API
- Tavily Search API
- SQLite
- pytest + Ruff

## Architecture Overview

The project follows a pragmatic modular layout:

- `config/` for environment configuration
- `agents/` for graph state, routing, and node logic
- `tools/` for domain tools like search and password generation
- `services/` for model initialization, streaming, and conversation management
- `persistence/` for SQLite checkpointer configuration
- `ui/` for Streamlit rendering and approval UX
- `app.py` as the minimal application entry point

## Project Structure

```text
agentic-chatbot/
├── .env.example
├── README.md
├── pyproject.toml
├── src/
│   └── agentic_chatbot/
│       ├── __init__.py
│       ├── app.py
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── graph.py
│       │   ├── nodes.py
│       │   └── state.py
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py
│       ├── persistence/
│       │   ├── __init__.py
│       │   └── checkpoint.py
│       ├── services/
│       │   ├── __init__.py
│       │   ├── conversations.py
│       │   ├── llm.py
│       │   └── streaming.py
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── password.py
│       │   └── search.py
│       └── ui/
│           ├── __init__.py
│           ├── approval.py
│           ├── chat.py
│           └── sidebar.py
└── tests/
    ├── test_config.py
    ├── test_conversations.py
    ├── test_graph.py
    └── test_tools.py
```

## Installation

Use `uv` to install dependencies and create the project environment:

```bash
uv sync
```

## Environment Variables

Copy the sample environment file and provide real values:

```bash
cp .env.example .env
```

Then configure the required keys:

- `GEMINI_API_KEY` or `GOOGLE_API_KEY` for Google Gemini access
- `TAVILY_API_KEY` for web search requests
- `CHATBOT_DB_PATH` optionally overrides the default SQLite checkpoint file

## Running the Application

Start the Streamlit app with:

```bash
uv run streamlit run src/agentic_chatbot/app.py
```

## LangGraph Workflow

The application uses a `StateGraph` to structure the workflow:

1. The LLM responds to the current message.
2. If a tool call is present, the graph routes to the approval node.
3. Search-related tool calls are paused for human approval.
4. Approved calls execute normally through the tool node.
5. The graph returns to the chat node and continues the conversation.

## Human-in-the-Loop Approval

When the agent requests a web search, the application interrupts the graph and asks for explicit approval. The UI shows:

- the tool name,
- the arguments being sent,
- approve/reject actions.

If the user rejects the request, the tool execution is skipped and the agent continues safely without the tool result.

## SQLite Conversation Persistence

The project stores LangGraph checkpoints in SQLite to retain conversation history across app restarts. The default database path is `chatbot_state.db`.

Configuration is centralized in `persistence/checkpoint.py`, and previous checkpoint data remains compatible with the existing thread IDs and conversation flow.

## Development Commands

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Testing

The test suite uses pytest and avoids external API calls by mocking or faking the relevant inputs. It covers:

- password generation and validation,
- tool registration,
- graph routing,
- approval and rejection flows,
- conversation formatting,
- configuration validation.

## Notes

- The app preserves the existing LangGraph architecture rather than replacing it with a higher-level agent wrapper.
- Search approval uses the actual registered Tavily tool name rather than a guessed variable name.
- The visible UI text is kept in English to match the current requirement.
