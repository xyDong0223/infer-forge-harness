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


def validate_environment_status(status: dict[str, Any]) -> list[str]:
    """Acceptance for the environment half of the proof.

    Deliberately says nothing about a server. What this phase delivers is a pod
    whose stack actually imports, plus the fingerprint that says which stack —
    and downstream Tasks (model scan, failure triage) run inside that pod, so the
    pod name and the fingerprint are part of the deliverable, not decoration.
    """
    errors: list[str] = []
    if status.get("state") != "ENVIRONMENT_READY":
        errors.append("state must be ENVIRONMENT_READY")
    checks = status.get("checks", {})
    for key in ("pod_ready", "runtime_importable", "code_ready", "device_ready"):
        if checks.get(key) is not True:
            errors.append(f"checks.{key} must be True")
    if not status.get("pod"):
        errors.append("pod must be recorded so a later phase can import it")
    artifacts = status.get("artifacts") or []
    for required in (
        "environment_fingerprint.txt",
        "runtime_import.txt",
        "code_readiness.json",
        "device_readiness.json",
    ):
        if required not in artifacts:
            errors.append(f"{required} must be part of the evidence bundle")
    return errors
