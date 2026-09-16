"""Validation gates for asynchronous operator lifecycle artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sqlite3
from typing import Any


def _scheduled_dispatch(report: dict[str, Any]) -> list[str]:
    from engine.scheduler import EventStore

    errors = []
    try:
        store = EventStore(report["scheduler_state"], readonly=True)
        try:
            run = store.run(report["run_id"])
            if run is None:
                return ["dispatch references an unknown adaptation run"]
            for key, expected in {
                "subject": run.model_id,
                "evidence_mode": run.metadata.get("evidence_mode", "real"),
                "environment_fingerprint": run.environment.get("environment_proof", {}).get("fingerprint"),
                "artifact_root": run.metadata.get("artifact_root"),
            }.items():
                if report.get(key) != expected:
                    errors.append(f"dispatch {key} does not match the persisted run")
            if not run.metadata.get("graph_environment_required"):
                errors.append("dispatch requires an activated graph scheduler binding")
            if report["state"] != "DISPATCH_BLOCKED" and run.status != "ENVIRONMENT_READY":
                errors.append("dispatch requires a currently ready environment")
            task_ids, keys = report.get("task_ids"), report.get("operator_keys")
            if (not isinstance(task_ids, list) or not isinstance(keys, list)
                    or not all(isinstance(value, str) for value in [*task_ids, *keys])):
                return [*errors, "dispatch requires task_ids and operator_keys lists"]
            tasks = [task for task in store.tasks(run.run_id) if task.task_id in task_ids]
            if sorted(task.task_id for task in tasks) != sorted(task_ids):
                errors.append("dispatch task ids do not identify tasks owned by this run")
            if sorted({task.operator_key for task in tasks}) != sorted(keys):
                errors.append("dispatch operator keys do not match persisted tasks")
            if any(task.stage != "torch" for task in tasks):
                errors.append("dispatch must identify the operator's initial torch task")
            if report.get("requests") != task_ids or report.get("request_count") != len(task_ids):
                errors.append("dispatch requests must match persistent task ids")
            if report["state"] == "DISPATCH_SKIPPED" and task_ids:
                errors.append("DISPATCH_SKIPPED cannot contain dispatched tasks")
            sources = report.get("reports")
            if not isinstance(sources, list) or (not sources and report["state"] != "DISPATCH_BLOCKED"):
                errors.append("dispatch requires hashed source report references")
            else:
                for source in sources:
                    path = Path(source["path"])
                    if (path.is_symlink() or not path.is_file()
                            or hashlib.sha256(path.read_bytes()).hexdigest() != source.get("sha256")):
                        errors.append("dispatch source report evidence is missing or changed")
        finally:
            store.close()
    except (OSError, ValueError, KeyError, TypeError, sqlite3.DatabaseError) as exc:
        errors.append(f"cannot validate scheduled dispatch: {exc}")
    return errors


def validate_dispatch(report: dict[str, Any]) -> list[str]:
    errors = []
    scheduled = "scheduler_state" in report or "run_id" in report or report.get("state") == "DISPATCH_BLOCKED"
    allowed = {"DISPATCHED", "DISPATCH_SKIPPED"}
    if scheduled:
        allowed.add("DISPATCH_BLOCKED")
    if report.get("state") not in allowed:
        errors.append("dispatch state is invalid")
    if report.get("state") == "DISPATCHED" and not report.get("requests"):
        errors.append("DISPATCHED requires request ids")
    if scheduled:
        errors.extend(_scheduled_dispatch(report))
        if report.get("state") == "DISPATCH_BLOCKED" and not report.get("errors"):
            errors.append("DISPATCH_BLOCKED must explain unresolved evidence")
        if report.get("state") != "DISPATCH_BLOCKED" and report.get("errors"):
            errors.append("successful dispatch cannot contain unresolved errors")
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
    if state in {"OPERATORS_READY", "WAITING_FOR_OPERATORS", "OPERATORS_BLOCKED"}:
        from engine.graph_bridge import GraphSchedulerBridge

        errors: list[str] = []
        try:
            bridge = GraphSchedulerBridge(
                Path(report["scheduler_state"]), report["run_id"], report["subject"],
                Path(report["artifact_root"]), {}, execute=False,
            )
            try:
                gate = bridge.delivery_status()
                try:
                    baseline = bridge.validate_baseline(Path(report["baseline_path"]))
                    if report.get("baseline_id") != baseline["baseline_id"]:
                        errors.append("integration baseline id does not match frozen evidence")
                except (OSError, ValueError, KeyError, TypeError, sqlite3.DatabaseError) as exc:
                    gate["state"] = "OPERATORS_BLOCKED"
                    gate["errors"].append(str(exc))
                    if state != "OPERATORS_BLOCKED":
                        errors.append(f"integration baseline evidence rejected: {exc}")
                for key in ("state", "run_id", "evidence_mode", "environment_fingerprint",
                            "task_ids", "operator_keys", "task_results", "snapshot_token"):
                    if report.get(key) != gate.get(key):
                        errors.append(f"integration {key} does not match the live scheduler gate")
                if state == "OPERATORS_BLOCKED" and not report.get("errors"):
                    errors.append("OPERATORS_BLOCKED must explain failed gates")
                if state == "OPERATORS_READY" and report.get("errors"):
                    errors.append("OPERATORS_READY cannot contain failed gates")
            finally:
                bridge.close()
        except (OSError, ValueError, KeyError, TypeError, sqlite3.DatabaseError) as exc:
            errors.append(f"cannot validate scheduled integration: {exc}")
        return errors
    if "scheduler_state" in report or "run_id" in report:
        return ["scheduled integration cannot use legacy candidate states"]
    if state not in {"WAITING_FOR_CANDIDATE", "READY_FOR_INTEGRATION", "CANDIDATE_REJECTED"}:
        return ["integration state is invalid"]
    errors: list[str] = []
    if state == "READY_FOR_INTEGRATION":
        for gate, evidence in READY_GATES:
            if not report.get(evidence):
                errors.append(
                    f"READY_FOR_INTEGRATION claims {gate} but records no {evidence}: "
                    "an integration without its evidence file is a claim, not a fact"
                )
    if state == "CANDIDATE_REJECTED" and not report.get("failed_gates"):
        errors.append("CANDIDATE_REJECTED must name the gates that failed")
    return errors
