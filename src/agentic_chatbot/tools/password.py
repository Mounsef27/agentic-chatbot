"""Secure password generation tool."""

from __future__ import annotations

import secrets
import string

from langchain_core.tools import tool


def generate_password(length: int = 16) -> str:
    """Generate a cryptographically secure password.

    Args:
        length (int): Desired password length. Must be between 8 and 128 characters.

    Returns:
        str: A random password composed of letters, digits, and special characters.

    Raises:
        ValueError: If the requested length is outside the supported range.
    """
    if not 8 <= length <= 128:
        raise ValueError("Password length must be between 8 and 128.")

    characters = string.ascii_letters + string.digits + "!@#$%&*"
    return "".join(secrets.choice(characters) for _ in range(length))


password_tool = tool(generate_password)
