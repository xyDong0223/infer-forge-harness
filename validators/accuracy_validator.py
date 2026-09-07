"""Accuracy validator interface for future reference/candidate differential runs."""

from __future__ import annotations

from typing import Any


def validate_accuracy_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "ACCURACY_PASS":
        errors.append("accuracy report is not ACCURACY_PASS")
    if not report.get("cases"):
        errors.append("accuracy report must include case-level evidence")
    if report.get("threshold_source") in (None, ""):
        errors.append("threshold_source is required")
    return errors
