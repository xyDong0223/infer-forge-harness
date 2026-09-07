"""Independent MAT-004 acceptance checks.

A classification is a routing decision, so the failure that costs the most is a
class whose next action does not follow from it — someone writes a model file for
a version lag, or reads `NO_STATIC_GAP` as "it works".
"""

from __future__ import annotations

from typing import Any

CLASSES = {
    "NO_STATIC_GAP",
    "REGISTRATION_MISSING",
    "VERSION_LAG",
    "CAPABILITY_MISSING",
    "UNVERIFIED",
}
# The action each class must lead to, as a substring of the recorded next action.
REQUIRED_ACTION = {
    "REGISTRATION_MISSING": "model implementation",
    "VERSION_LAG": "cherry-pick",
    "NO_STATIC_GAP": "triage",
}


def validate_gap_classification(report: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    acceptance = contract.get("acceptance") or {}

    if report.get("state") != "CLASSIFICATION_READY":
        errors.append(f"state is {report.get('state')!r}, not CLASSIFICATION_READY")
    if report.get("runtime_verified") is not False:
        errors.append(
            "runtime_verified must be false: this Task reads two static findings and executes "
            "nothing"
        )

    classification = report.get("classification")
    if classification not in CLASSES:
        errors.append(f"unknown classification {classification!r}")
    action = str(report.get("next_action") or "")
    if not action:
        errors.append("next_action is required: a class without an action is not a routing decision")
    expected = REQUIRED_ACTION.get(str(classification))
    if expected and expected not in action:
        errors.append(
            f"{classification} must lead to an action mentioning {expected!r}, got {action!r}"
        )

    gaps = report.get("gaps")
    if gaps is None:
        errors.append("gaps must be present, even when empty")
    else:
        for gap in gaps:
            if gap.get("class") not in CLASSES:
                errors.append(f"gap {gap.get('axis')!r} has unknown class {gap.get('class')!r}")
            for field in ("axis", "detail", "next_action", "evidence"):
                if not gap.get(field):
                    errors.append(f"gap {gap.get('axis', '?')!r} is missing {field}")
        blocking = {gap["axis"] for gap in gaps
                    if gap.get("class") in ("REGISTRATION_MISSING", "CAPABILITY_MISSING")}
        if set(report.get("blocking") or []) != blocking:
            errors.append("blocking disagrees with the per-gap classes")
        if not gaps and classification != "NO_STATIC_GAP":
            errors.append(f"no gap was found but the classification is {classification!r}")
        if gaps and classification == "NO_STATIC_GAP":
            errors.append("gaps were found, so NO_STATIC_GAP would hide them")

    if acceptance.get("require_input_provenance"):
        inputs = report.get("inputs") or {}
        for field in ("scan_state", "match_verdict"):
            if not inputs.get(field):
                errors.append(f"inputs.{field} must record what this classification was derived from")
        if not (inputs.get("scanned_in") or inputs.get("matched_in")):
            errors.append("inputs must name the pod whose installation the findings came from")
    return errors
