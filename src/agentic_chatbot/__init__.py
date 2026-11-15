"""Public package interface for the agentic chatbot."""


def main() -> None:
    """Run the Streamlit application entry point."""
    from agentic_chatbot.app import main as app_main

    app_main()


__all__ = ["main"]
