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


READY_GATES = (
    # package_swap and path_proof are the GLM-5.2 lesson: a candidate that
    # cannot show how it was built into the pod and that every rank actually
    # took its code path stood behind four failed swaps before a green one.
    ("kernel_grade", "kernel_grade_report"),
    ("dispatch_report", "dispatch_report_path"),
    ("package_swap", "package_swap_report"),
    ("path_proof", "worker_path_log"),
    ("service_regression", "service_regression_report"),
    ("accuracy_regression", "accuracy_regression_report"),
)


def validate_integration(report: dict[str, Any]) -> list[str]:
    state = report.get("state")
    if state not in {"WAITING_FOR_CANDIDATE", "READY_FOR_INTEGRATION", "CANDIDATE_REJECTED"}:
        return ["integration state is invalid"]
    errors: list[str] = []
    candidates = report.get("candidates") or []
    if state in {"READY_FOR_INTEGRATION", "CANDIDATE_REJECTED"} and not candidates:
        errors.append(f"{state} requires graded candidates")
    if state == "READY_FOR_INTEGRATION":
        for candidate in candidates:
            failed = candidate.get("failed_gates") or {}
            for gate, evidence in READY_GATES:
                # package_swap and path_proof are the GLM-5.2 lesson: a candidate
                # that cannot show how it was built into the pod and that every
                # rank actually took its code path stood behind four failed swaps
                # before a green one. grade_candidate folds a missing evidence
                # file into failed_gates as "evidence:<gate>".
                if f"evidence:{gate}" in failed:
                    errors.append(
                        f"READY_FOR_INTEGRATION claims {gate} for {candidate.get('candidate_id')} "
                        f"but records no {evidence}: an integration without its evidence file "
                        "is a claim, not a fact"
                    )
        if not report.get("bisect_schedule"):
            errors.append(
                "READY_FOR_INTEGRATION must record a bisect schedule: a failed batch is "
                "attributed by bisection, not by waiting for a human"
            )
    if state == "CANDIDATE_REJECTED" and not report.get("failed_candidates"):
        errors.append("CANDIDATE_REJECTED must name the candidates whose gates failed")
    return errors
