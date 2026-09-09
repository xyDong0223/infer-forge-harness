"""Validate a RuntimeDriftReport.

The gate is deliberately blunt: any plugin module that fails to import against the
installed engine is drift, and drift found here is drift not found one model load
at a time. The validator's job is to refuse a report that cannot support that
claim -- an empty scan, a missing engine identity, or a failure list with no
per-failure evidence.
"""

from __future__ import annotations

from typing import Any

# Modules whose import failure says nothing about drift: they re-register custom
# ops and legitimately refuse a second registration in an already-initialised
# process. Recorded rather than silently dropped.
BENIGN_ERROR_TYPES = {"RuntimeError"}


def validate_drift_report(report: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    state = report.get("state")
    if state not in ("DRIFT_CLEAR", "DRIFT_FOUND"):
        errors.append(f"state {state!r} is neither DRIFT_CLEAR nor DRIFT_FOUND")

    engine = report.get("engine") or {}
    for field in ("name", "path"):
        if not engine.get(field):
            errors.append(f"engine.{field} is required: drift is relative to a specific install")

    plugin = report.get("plugin") or {}
    scanned = plugin.get("modules_scanned")
    if not isinstance(scanned, int) or scanned <= 0:
        errors.append("plugin.modules_scanned must be a positive count: an empty scan proves nothing")

    failures = report.get("failures")
    if failures is None:
        errors.append("failures is required, even when empty")
    else:
        for index, failure in enumerate(failures):
            for field in ("module", "error_type", "message"):
                if not failure.get(field):
                    errors.append(f"failures[{index}].{field} is required")
            if "resolution" not in failure:
                errors.append(
                    f"failures[{index}] has no resolution field: a drift report that does not "
                    "say where the symbol went leaves the next Task guessing"
                )

    if state == "DRIFT_CLEAR" and failures:
        errors.append("state is DRIFT_CLEAR but failures were recorded")
    if state == "DRIFT_FOUND" and not failures:
        errors.append("state is DRIFT_FOUND but no failure was recorded")

    acceptance = (contract or {}).get("acceptance") or {}
    if acceptance.get("require_resolution_candidates"):
        for index, failure in enumerate(failures or []):
            resolution = failure.get("resolution") or {}
            if resolution.get("kind") in ("MISSING_SYMBOL", "MISSING_MODULE") and not resolution.get(
                "candidates"
            ):
                # Not an error: a symbol can be genuinely gone, which is a real
                # answer. It must be stated as such rather than left blank.
                if not resolution.get("gone_upstream"):
                    errors.append(
                        f"failures[{index}].resolution has no candidates and is not marked "
                        "gone_upstream: say which of the two it is"
                    )

    return errors
