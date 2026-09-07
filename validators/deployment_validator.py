"""Independent KDP-001 acceptance checks."""

from __future__ import annotations

from typing import Any


def validate_deployment_status(status: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if status.get("state") != "DEPLOYMENT_READY":
        errors.append("state must be DEPLOYMENT_READY")
    checks = status.get("checks", {})
    required = {
        "pod_ready": True,
        "health_check": 200,
        "chat_completion": "non_empty",
        "expected_backend": "kunlun",
        "unexpected_fallback": False,
    }
    for key, expected in required.items():
        if checks.get(key) != expected:
            errors.append(f"checks.{key} must equal {expected!r}")
    if not status.get("artifacts"):
        errors.append("artifacts must contain at least one evidence path")
    return errors
