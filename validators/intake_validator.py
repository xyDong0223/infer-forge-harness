"""Independent MAT-001 acceptance checks.

The intake Tool produces a ModelRequest; this decides whether it may be called
an identity fact. The rules are read from the Task Contract rather than
duplicated here, so tightening the contract tightens the judgment — a validator
that hardcodes its own list drifts from the contract it is supposed to enforce.
"""

from __future__ import annotations

import re
from typing import Any

HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
# Wording that would claim more coverage than a sampled digest provides.
FULL_HASH_CLAIMS = ("full", "complete", "every byte", "whole file")


def _get(payload: Any, path: str) -> Any:
    for part in path.split("."):
        if not isinstance(payload, dict) or part not in payload:
            return None
        payload = payload[part]
    return payload


def _keys(payload: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            found.add(key)
            found |= _keys(value)
    elif isinstance(payload, list):
        for item in payload:
            found |= _keys(item)
    return found


def validate_model_request(
    request: dict[str, Any], contract: dict[str, Any], probe: dict[str, Any] | None = None
) -> list[str]:
    errors: list[str] = []
    acceptance = contract.get("acceptance") or {}
    checks = contract.get("checks") or {}

    for path in acceptance.get("required_fields") or []:
        value = _get(request, path)
        if value is None or value == "" or value == []:
            errors.append(f"{path} is required by the contract but is empty")

    present = _keys(request)
    for forbidden in acceptance.get("forbidden_fields") or []:
        if forbidden in present:
            errors.append(
                f"{forbidden!r} must not appear in an intake artifact: it is a runtime decision, "
                "not a fact about the checkpoint"
            )

    revision = _get(request, "model.revision")
    if isinstance(revision, str) and not HEX64.match(revision):
        errors.append("model.revision must be a sha256 hex digest")
    method = _get(request, "model.revision_method") or ""
    if (checks.get("revision") or {}).get("reject_full_hash_claim") and any(
        claim in method.lower() for claim in FULL_HASH_CLAIMS
    ):
        errors.append(
            f"model.revision_method claims full coverage ({method!r}) but the digest is sampled"
        )

    commit = _get(request, "target.vllm_kunlun_commit")
    ref = _get(request, "target.vllm_kunlun_ref")
    if not isinstance(commit, str) or not HEX40.match(commit or ""):
        errors.append("target.vllm_kunlun_commit must be a resolved 40-char commit sha")
    if commit and ref and commit == ref:
        errors.append("target.vllm_kunlun_commit still holds the ref: a branch moves")

    identity = request.get("identity") or {}
    architectures = identity.get("architectures")
    if not isinstance(architectures, list) or not architectures:
        errors.append("identity.architectures must list at least one architecture")
    remote_code = identity.get("remote_code") or []
    if remote_code and not _get(request, "model.trust_remote_code_required"):
        errors.append(
            "identity.remote_code is non-empty but trust_remote_code_required is false: "
            "the checkpoint carries its own modelling code"
        )

    if acceptance.get("evidence_required"):
        if not probe:
            errors.append("evidence_required: the probe result must accompany the request")
        else:
            if probe.get("state") != "INTAKE_READY":
                errors.append(f"probe state is {probe.get('state')!r}, not INTAKE_READY")
            if not probe.get("probed_in"):
                errors.append("probe must record where it ran (probed_in)")
            if probe.get("revision") != revision:
                errors.append("model.revision does not match the probe's own digest")
    return errors
