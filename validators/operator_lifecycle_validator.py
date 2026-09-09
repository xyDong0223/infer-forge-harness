"""Validation gates for asynchronous operator lifecycle artifacts."""

from __future__ import annotations

from typing import Any


def validate_dispatch(report: dict[str, Any]) -> list[str]:
    errors = []
    if report.get("state") not in {"DISPATCHED", "DISPATCH_SKIPPED"}:
        errors.append("dispatch state is invalid")
    if report.get("state") == "DISPATCHED" and not report.get("requests"):
        errors.append("DISPATCHED requires request ids")
    return errors


def validate_baseline(report: dict[str, Any]) -> list[str]:
    errors = []
    if report.get("state") != "BASELINE_FROZEN":
        errors.append("baseline is not frozen")
    if not report.get("baseline_id"):
        errors.append("baseline_id is required")
    return errors


def validate_integration(report: dict[str, Any]) -> list[str]:
    state = report.get("state")
    if state not in {"WAITING_FOR_CANDIDATE", "READY_FOR_INTEGRATION", "CANDIDATE_REJECTED"}:
        return ["integration state is invalid"]
    return []
