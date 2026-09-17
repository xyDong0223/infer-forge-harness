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


def environment_user_id(environment: dict, supplied: str | None = None) -> str | None:
    """Keep both successful and failed durable handoffs under one explicit owner."""
    recorded = {
        resolve_user_id(recorded=handoff["user_id"])
        for key in ("environment_proof", "failed_environment_proof")
        if isinstance(handoff := environment.get(key), dict) and handoff.get("user_id")
    }
    if len(recorded) > 1:
        raise ValueError("environment handoffs contain conflicting user_id values")
    owner = next(iter(recorded), None)
    requested = resolve_user_id(supplied) if supplied is not None else None
    if owner and requested is not None and requested != owner:
        raise ValueError("user_id does not match the recorded environment owner")
    return owner if owner is not None else requested
