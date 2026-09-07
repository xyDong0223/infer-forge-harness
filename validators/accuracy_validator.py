"""Independent MAT-013 acceptance checks.

Reads its thresholds from the contract. The failure this guards against is a
plausible-looking report: fluent output, a passing status, and no way to tell what
it was compared against — at which point "accuracy verified" means nothing.

Note the status handling. An ACCURACY_FAIL is a valid, well-formed report and must
not be rejected as malformed; the Validator's job is to check that a *pass* is
earned, not to insist every run passes.
"""

from __future__ import annotations

from typing import Any


def validate_accuracy_report(
    report: dict[str, Any], thresholds: dict[str, Any] | None = None
) -> list[str]:
    thresholds = thresholds or {}
    errors: list[str] = []

    if report.get("status") not in ("ACCURACY_PASS", "ACCURACY_FAIL"):
        errors.append(f"status {report.get('status')!r} is neither a pass nor a fail")
    cases = report.get("cases")
    if not cases:
        errors.append("accuracy report must include case-level evidence")
    if not report.get("threshold_source"):
        errors.append("threshold_source is required: a report without it is an opinion")

    reference = report.get("reference") or {}
    for field in ("implementation", "device"):
        if not reference.get(field):
            errors.append(f"reference.{field} is required: a differential names what it differs from")
    if reference.get("device") and reference["device"] == (report.get("candidate") or {}).get("device"):
        errors.append(
            "the reference shares the candidate's device, so agreement proves nothing about the "
            "accelerator"
        )
    if not report.get("metric"):
        errors.append("metric is required")

    for case in cases or []:
        label = case.get("prompt", "?")
        for field in ("top1_candidate", "top1_reference", "candidate_top5", "reference_top5"):
            if case.get(field) in (None, []):
                errors.append(f"case {label!r} is missing {field}")
        if case.get("top1_match") is None:
            errors.append(f"case {label!r} does not state whether the top token matched")

    if cases and report.get("status") == "ACCURACY_PASS":
        if thresholds.get("require_top1_match_on_all_cases"):
            failed = [case.get("prompt") for case in cases if not case.get("top1_match")]
            if failed:
                errors.append(f"top-1 disagreed on {failed} but the status is ACCURACY_PASS")
        floor = thresholds.get("min_top5_overlap")
        if floor is not None:
            weak = [
                case.get("prompt") for case in cases
                if int(case.get("top5_overlap", 0)) < int(floor)
            ]
            if weak:
                errors.append(
                    f"top-5 overlap is below {floor} on {weak} but the status is ACCURACY_PASS"
                )
    return errors
