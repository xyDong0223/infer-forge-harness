"""Independent MAT-003 acceptance checks.

A capability match is the easiest artifact in this repository to over-read: it
looks like a support statement and is not one. Qwen3-8B matches every axis on
P800 and still fails in an attention kernel, so the checks below exist to keep
the artifact honest about what it did and did not establish.
"""

from __future__ import annotations

from typing import Any

AXIS_VERDICTS = {
    "PROVIDED",
    "PROVIDED_MODULE_ONLY",
    "NOT_PROVIDED",
    "NOT_REQUIRED",
    "UNKNOWN",
}
OVERALL = {"MATCHED", "MATCHED_WITH_UNKNOWNS", "MISMATCH"}
EVIDENCE = {"REGISTRY", "MODULE", "ABSENT", "N/A"}
# Wording that would turn a static match into a support claim.
BANNED_CLAIMS = ("supported", "works", "verified", "ready")


def validate_capability_match(match: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    checks = contract.get("checks") or {}
    acceptance = contract.get("acceptance") or {}

    if match.get("state") != "MATCH_READY":
        errors.append(f"state is {match.get('state')!r}, not MATCH_READY")
    if match.get("runtime_verified") is not False:
        errors.append(
            "runtime_verified must be present and false: this Task executes nothing, so it cannot "
            "claim a runtime result"
        )
    if not match.get("matched_in"):
        errors.append("matched_in must record the pod whose installation was inspected")

    verdict = match.get("verdict")
    if verdict not in OVERALL:
        errors.append(f"unknown overall verdict {verdict!r}")
    if any(claim in str(verdict).lower() for claim in BANNED_CLAIMS):
        errors.append(f"verdict {verdict!r} reads as a support claim")

    axes = match.get("axes") or []
    if not axes:
        errors.append("no capability axis was matched")
    for axis in axes:
        name = axis.get("axis", "?")
        if axis.get("verdict") not in AXIS_VERDICTS:
            errors.append(f"{name}: unknown verdict {axis.get('verdict')!r}")
        if axis.get("evidence") not in EVIDENCE:
            errors.append(f"{name}: evidence must be graded, got {axis.get('evidence')!r}")
        if axis.get("verdict") == "PROVIDED" and axis.get("evidence") != "REGISTRY":
            errors.append(
                f"{name}: PROVIDED requires registry evidence; a module on disk only supports "
                "PROVIDED_MODULE_ONLY"
            )
        if axis.get("verdict") == "NOT_PROVIDED" and axis.get("evidence") != "ABSENT":
            errors.append(f"{name}: NOT_PROVIDED must be backed by absent evidence")

    blocking = {axis["axis"] for axis in axes if axis.get("verdict") == "NOT_PROVIDED"}
    unknown = {axis["axis"] for axis in axes if axis.get("verdict") == "UNKNOWN"}
    if set(match.get("blocking_axes") or []) != blocking:
        errors.append("blocking_axes disagrees with the per-axis verdicts")
    if set(match.get("unknown_axes") or []) != unknown:
        errors.append("unknown_axes disagrees with the per-axis verdicts")
    if blocking and verdict != "MISMATCH":
        errors.append(f"{sorted(blocking)} are NOT_PROVIDED but the overall verdict is {verdict!r}")
    if not blocking and unknown and verdict != "MATCHED_WITH_UNKNOWNS":
        errors.append(
            f"{sorted(unknown)} were not introspected, so MATCHED would hide an open question"
        )

    if checks.get("require_graded_evidence") and any(
        axis.get("evidence") == "N/A" and axis.get("verdict") not in ("NOT_REQUIRED", "UNKNOWN")
        for axis in axes
    ):
        errors.append("an axis claims a result without evidence")
    if acceptance.get("evidence_required") and not match.get("provided"):
        errors.append("the installation's provided-capability inventory must be recorded")
    return errors
