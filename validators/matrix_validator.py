"""Independent MAT-016 acceptance checks.

The matrix is read instead of the evidence, so the only interesting failure is an
entry that claims more than its evidence supports. `validated` therefore requires
both a deployment proof and an accuracy differential, and every row must carry the
conditions that make it falsifiable — a revision, a stack commit, and whatever the
result depended on.
"""

from __future__ import annotations

from typing import Any

STATUSES = {"validated", "runs_unverified_accuracy", "not_yet_validated", "unsupported"}


def validate_matrix_entry(entry: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    acceptance = contract.get("acceptance") or {}
    rules = (contract.get("checks") or {}).get("status_requires") or {}

    status = entry.get("status")
    if status not in STATUSES:
        errors.append(f"unknown status {status!r}")
    for field in ("model", "hardware"):
        if not entry.get(field):
            errors.append(f"{field} is required")

    evidence = entry.get("evidence") or {}
    required = rules.get(str(status)) or []
    for name in required:
        if not evidence.get(name):
            errors.append(
                f"status {status!r} requires {name} evidence, which is absent — a claim without it "
                "is not falsifiable"
            )
    if status == "validated":
        if evidence.get("deployment_proof") != "DEPLOYMENT_READY":
            errors.append("validated requires a DEPLOYMENT_READY proof")
        if evidence.get("accuracy_differential") != "ACCURACY_PASS":
            errors.append(
                "validated requires ACCURACY_PASS: a server that answers is not a server that is "
                "right"
            )

    if acceptance.get("require_validity_conditions") and status != "not_yet_validated":
        for field in ("revision", "stack_commit"):
            if not entry.get(field):
                errors.append(f"{field} is required: a row is only true of one {field}")
        if entry.get("conditions") is None:
            errors.append("conditions must be recorded, even when empty")
    if acceptance.get("require_limitations_when_conditional"):
        conditions = " ".join(entry.get("conditions") or [])
        if "enforce_eager=True" in conditions and not entry.get("limitations"):
            errors.append(
                "the result depended on eager execution, so that limitation must appear in the row"
            )
    return errors
