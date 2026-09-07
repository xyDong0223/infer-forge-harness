"""Performance validator interface for reproducible baseline/candidate comparisons."""

from __future__ import annotations

from typing import Any


def validate_performance_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in ("workload", "baseline", "candidate", "raw_result"):
        if not report.get(field):
            errors.append(f"{field} evidence is required")
    if report.get("decision") not in {"PASS", "NO_GO", "REWORK", "NEEDS_HUMAN"}:
        errors.append("decision must be a controlled terminal state")
    return errors
