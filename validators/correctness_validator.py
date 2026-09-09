"""Shared correctness gates for model composition and platform kernels."""

from __future__ import annotations

from typing import Any


def validate_kernel_grade(report: dict[str, Any]) -> list[str]:
    errors = []
    if not isinstance(report, dict):
        return ["kernel grade must be an object"]
    if report.get("state") not in {"PASS", "FAIL", "AMBIGUOUS"}:
        errors.append("kernel grade state must be PASS, FAIL, or AMBIGUOUS")
    for field in ("relative_l2", "max_abs_error", "reference_provenance"):
        if report.get(field) in (None, {}, ""):
            errors.append(f"kernel grade requires {field}")
    for field in ("relative_l2", "max_abs_error", "max_relative_l2"):
        if field in report and not isinstance(report[field], (int, float)):
            errors.append(f"kernel grade {field} must be numeric")
    if report.get("shape_candidate") != report.get("shape_reference"):
        errors.append("kernel grade candidate and reference shapes must match")
    provenance = report.get("reference_provenance") or {}
    for field in ("implementation", "device", "dtype"):
        if not provenance.get(field):
            errors.append(f"kernel grade reference provenance requires {field}")
    if provenance.get("independently_written") is not True:
        errors.append("kernel grade reference must be independently written")
    if report.get("control_discriminates") is not True:
        errors.append("kernel grade requires a discriminating negative control")
    if report.get("state") == "PASS":
        if report.get("max_relative_l2") is None:
            errors.append("kernel grade PASS requires max_relative_l2")
        elif report.get("relative_l2", 1.0) > report["max_relative_l2"]:
            errors.append("kernel grade pass exceeds its declared relative-L2 gate")
    return errors


def validate_end_to_end(report: dict[str, Any]) -> list[str]:
    errors = []
    if report.get("state") not in ("ACCURACY_PASS", "ACCURACY_FAIL"):
        errors.append("end-to-end report must state ACCURACY_PASS or ACCURACY_FAIL")
    cases = report.get("cases") or []
    if not cases:
        errors.append("end-to-end report requires case-level evidence")
    if report.get("composition") != "integrated_serving_path":
        errors.append("end-to-end evidence must use the integrated serving path")
    if report.get("reference_provenance", {}).get("device") == report.get(
        "candidate", {}
    ).get("device"):
        errors.append("end-to-end reference must be independent of candidate device")
    for case in cases:
        if not isinstance(case, dict) or not case.get("evidence"):
            errors.append("each end-to-end case must carry evidence")
    return errors


def validate_long_context(report: dict[str, Any]) -> list[str]:
    errors = []
    geometry = report.get("geometry") or {}
    context_len = int(geometry.get("context_len", 0))
    boundary = int(geometry.get("block_size", 0)) * int(geometry.get("topk", 0))
    if context_len <= boundary:
        errors.append("long-context test must cross block_size * topk")
    if not report.get("selected_blocks"):
        errors.append("long-context report must record selected blocks")
    if report.get("path") != "sparse":
        errors.append("long-context report must prove the sparse path was exercised")
    if report.get("state") == "PASS" and report.get("relative_l2", 1.0) > report.get(
        "max_relative_l2", 0.0
    ):
        errors.append("long-context pass exceeds its declared relative-L2 gate")
    return errors
