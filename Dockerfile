# Use the same Python version as your project
FROM python:3.13-slim

WORKDIR /agentic_chatbot

COPY --from=ghcr.io/astral-sh/uv:0.12.7 /uv /uvx /bin/

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "src/agentic_chatbot/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
