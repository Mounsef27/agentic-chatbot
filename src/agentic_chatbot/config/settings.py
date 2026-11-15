"""Runtime configuration for the chatbot application."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Strongly typed application settings.

    Attributes:
        google_api_key (str | None): Google Gemini API key loaded from the environment.
        tavily_api_key (str | None): Tavily API key loaded from the environment.
        db_path (Path): Path to the SQLite checkpoint database.
        model_name (str): Gemini model identifier.
    """

    google_api_key: str | None
    tavily_api_key: str | None
    db_path: Path
    model_name: str = "gemini-3.5-flash-lite"

    def validate(self) -> None:
        """Validate required runtime settings.

        Raises:
            RuntimeError: If the required Google API key is missing.
        """
        if not self.google_api_key:
            raise RuntimeError("Missing Google API key. Set GEMINI_API_KEY or GOOGLE_API_KEY in your environment.")


def get_settings(env: dict[str, str] | None = None) -> Settings:
    """Load runtime settings from environment variables.

    Args:
        env (dict[str, str] | None): Optional environment mapping used for testing.

    Returns:
        Settings: The resolved settings instance.
    """
    load_dotenv()
    source = os.environ if env is None else env
    google_api_key = source.get("GOOGLE_API_KEY") or source.get("GEMINI_API_KEY")
    tavily_api_key = source.get("TAVILY_API_KEY")
    db_path = Path(source.get("CHATBOT_DB_PATH", "chatbot_state.db"))
    return Settings(
        google_api_key=google_api_key,
        tavily_api_key=tavily_api_key,
        db_path=db_path,
        model_name=source.get("GEMINI_MODEL", "gemini-3.5-flash-lite"),
    )
