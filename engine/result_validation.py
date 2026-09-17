"""Evidence gates for persisted worker results, independent of platform actions."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .contracts import AdaptationRun, DiagnosticTask, OperatorTask


STAGE_EVIDENCE = {
    "torch": ("reference_artifact", "focused_tests", "independent_validation"),
    "xpu": (
        "build_record", "registration_record", "device_test", "dispatch_report",
        "independent_validation",
    ),
    "integration": (
        "integration_report", "service_regression", "accuracy_regression",
        "fallback_report", "independent_validation",
    ),
    "diagnosis": ("diagnosis_report",),
}
NEXT_ACTIONS = {
    "REDISCOVER_OPERATOR", "DISPATCH_TORCH_FIX", "DISPATCH_XPU_FIX", "RETRY", "BLOCKED",
}


def _validate_legacy_result(
    task: OperatorTask | DiagnosticTask,
    run: AdaptationRun,
    result: dict[str, Any],
    worker: str,
) -> list[str]:
    """Check identity and evidence, not just the worker's success-shaped verdict."""
    errors: list[str] = []
    if type(result.get("schema_version")) is not int or result["schema_version"] != 1:
        errors.append("schema_version must be 1")
    verdicts = [result[key] for key in ("status", "verdict") if key in result]
    if not verdicts or any(str(value).upper() != "PASS" for value in verdicts):
        errors.append("an explicit PASS status/verdict is required")
    if any(result.get(key) is False for key in ("ok", "success", "passed")):
        errors.append("result contains an explicit failed check")
    identity = {
        "task_id": task.task_id,
        "operator_key": task.operator_key,
        "stage": task.stage,
        "attempt": task.attempt,
        "environment_fingerprint": run.environment.get("environment_proof", {}).get("fingerprint"),
    }
    mode = run.metadata.get("evidence_mode", "real")
    if mode not in ("real", "simulation") or (
        mode == "simulation" and run.metadata.get("environment_required")
    ):
        errors.append("simulation evidence cannot satisfy an environment-backed run")
    environment_required = mode == "real" or run.metadata.get("graph_environment_required")
    if environment_required and not identity["environment_fingerprint"]:
        errors.append("environment-backed results require a bound environment fingerprint")
    if environment_required and task.stage != "diagnosis" and run.status != "ENVIRONMENT_READY":
        errors.append("environment-backed results require a currently ready environment")
    identity["evidence_mode"] = mode
    for key, value in identity.items():
        if key not in result or result[key] != value:
            errors.append(f"result {key} does not match the claimed task/run")
    if type(result.get("attempt")) is not int:
        errors.append("result attempt must be an integer")

    evidence = result.get("evidence")
    hashes = result.get("evidence_sha256")
    if not isinstance(evidence, dict) or not isinstance(hashes, dict):
        return [*errors, "evidence and evidence_sha256 must be objects"]
    for key in STAGE_EVIDENCE[task.stage]:
        if key not in evidence:
            errors.append(f"missing {task.stage} evidence: {key}")
    contents: dict[str, bytes] = {}
    for key, value in evidence.items():
        if not isinstance(value, str) or not value.strip():
            errors.append(f"evidence.{key} must be a file path")
            continue
        path = Path(value)
        if not path.is_absolute():
            root = run.metadata.get("artifact_root")
            if not root:
                errors.append(f"relative evidence.{key} requires run metadata.artifact_root")
                continue
            path = Path(root) / path
        workspace_output = task.input.get("workspace", {}).get("output")
        if workspace_output and not path.resolve().is_relative_to(Path(workspace_output).resolve()):
            errors.append(f"evidence.{key} must belong to the claimed attempt output")
            continue
        if path.is_symlink() or not path.is_file():
            errors.append(f"evidence.{key} must be a regular non-symlink file")
            continue
        try:
            content = path.read_bytes()
        except OSError as exc:
            errors.append(f"cannot read evidence.{key}: {exc}")
            continue
        if not content:
            errors.append(f"evidence.{key} is empty")
        if hashes.get(key) != hashlib.sha256(content).hexdigest():
            errors.append(f"evidence.{key} hash mismatch")
        contents[key] = content

    report_key = "diagnosis_report" if task.stage == "diagnosis" else "independent_validation"
    if report_key not in contents:
        return [*errors, f"missing readable {report_key}"]
    try:
        report = json.loads(contents[report_key])
    except (ValueError, UnicodeDecodeError):
        return [*errors, f"{report_key} must contain JSON"]
    if not isinstance(report, dict):
        return [*errors, f"{report_key} must be an object"]
    for key, value in identity.items():
        if key not in report or report[key] != value:
            errors.append(f"{report_key} {key} does not match the claimed task/run")
    if type(report.get("attempt")) is not int:
        errors.append(f"{report_key} attempt must be an integer")
    if str(report.get("verdict", "")).upper() != "PASS":
        errors.append(f"{report_key} verdict must be PASS")
    if task.stage == "diagnosis":
        for key in ("diagnosis", "repair_conclusion", "next_action", "confidence"):
            if result.get(key) != report.get(key) or key not in report:
                errors.append(f"diagnosis result must match report field {key}")
        if not report.get("diagnosis") or not report.get("repair_conclusion"):
            errors.append("diagnosis and repair_conclusion are required")
        if not isinstance(report.get("next_action"), str) or report["next_action"] not in NEXT_ACTIONS:
            errors.append("diagnosis next_action is invalid")
        confidence = report.get("confidence")
        if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                or not math.isfinite(confidence) or not 0 <= confidence <= 1):
            errors.append("diagnosis confidence must be a finite value between 0 and 1")
    else:
        validator = report.get("validator")
        if not isinstance(validator, str) or not validator.strip() or validator == worker:
            errors.append("validation requires a recorded validator distinct from the producer")
        checks = report.get("checks")
        if not isinstance(checks, list) or not checks or any(
            not isinstance(check, dict) or not check.get("name") or check.get("passed") is not True
            for check in checks
        ):
            errors.append("independent validation requires nonempty passing named checks")
        expected_hashes = {key: hashes.get(key) for key in evidence if key != report_key}
        if report.get("evidence_sha256") != expected_hashes:
            errors.append("validation report does not bind the submitted evidence hashes")
    return errors


def validate_result(task, run, result: dict[str, Any], worker: str) -> list[str]:
    """Apply the recorded run protocol; uploaded reports cannot enable v2 trust."""
    from .managed_validation import requires_managed_validation, validate_managed_result

    producer = worker
    if requires_managed_validation(run, task):
        validation_id = result.get("managed_validation_id")
        receipt = (run.metadata.get("managed_validations", {}).get(validation_id)
                   if isinstance(validation_id, str) else None)
        if receipt:
            producer = receipt.get("producer", worker)
    errors = _validate_legacy_result(task, run, result, producer)
    errors.extend(validate_managed_result(task, run, result, worker))
    return errors
