"""Resolve the resource owner's supplied identity without guessing an account."""

import os
import re


def resolve_user_id(explicit: str | None = None, recorded: str | None = None) -> str:
    """Prefer explicit input, then the recorded contract, then legacy USER_ID."""
    value = explicit if explicit is not None else recorded
    if value is None:
        value = os.environ.get("USER_ID", "")
    if not isinstance(value, str):
        raise ValueError("user_id must be a string supplied by the user")
    value = value.strip()
    if value and not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", value):
        raise ValueError("user_id must contain lowercase letters, digits, and internal '-' or '.'")
    return value
