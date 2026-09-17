"""Independent KDP-001 acceptance checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from core.paths import REPO_ROOT


ENVIRONMENT_ARTIFACTS = (
    "environment_fingerprint.txt", "runtime_import.txt", "code_readiness.json",
    "device_readiness.json", "base_model_identity.json", "base_server_log.txt",
    "base_health_result.txt", "base_chat_result.json",
)


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

    Requires the configured MiniMax base-model service smoke as well as a pod
    whose stack imports, plus the fingerprint that says which stack —
    and downstream Tasks (model scan, failure triage) run inside that pod, so the
    pod name and the fingerprint are part of the deliverable, not decoration.
    """
    errors: list[str] = []
    if status.get("state") != "ENVIRONMENT_READY":
        errors.append("state must be ENVIRONMENT_READY")
    checks = status.get("checks", {})
    for key in (
        "pod_ready", "runtime_importable", "code_ready", "device_ready",
        "base_model_loaded", "base_prefill", "base_decode",
    ):
        if checks.get(key) is not True:
            errors.append(f"checks.{key} must be True")
    for key, expected in (("base_health_check", 200), ("base_chat_completion", "non_empty"), ("unexpected_fallback", False)):
        if checks.get(key) != expected:
            errors.append(f"checks.{key} must equal {expected!r}")
    if not status.get("pod"):
        errors.append("pod must be recorded so a later phase can import it")
    artifacts = status.get("artifacts") or []
    for required in ENVIRONMENT_ARTIFACTS:
        if required not in artifacts:
            errors.append(f"{required} must be part of the evidence bundle")
    return errors


def validate_environment_identity(artifact_root: Path) -> list[str]:
    """Verify persisted baseline identity, including imported proof bundles."""
    try:
        profile = yaml.safe_load((REPO_ROOT / "config/clusters/p800-cluster.yaml").read_text())
        expected = profile["validation"]["base_model"]
        identity = json.loads((Path(artifact_root) / "base_model_identity.json").read_text())
        if not isinstance(identity, dict):
            return ["base_model_identity.json must contain an object"]
        return [f"base_model_identity.{key} must equal configured {expected[key]!r}"
                for key in ("name", "path", "served_model_name")
                if not expected.get(key) or identity.get(key) != expected[key]]
    except (OSError, ValueError, KeyError, TypeError) as error:
        return [f"cannot validate base_model_identity.json: {error}"]
