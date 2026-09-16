"""Synthetic worker-result fixtures for scheduler tests, never device evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from engine.contracts import AdaptationRun, DiagnosticTask, OperatorTask
from engine.result_validation import STAGE_EVIDENCE


def simulation_result(
    task: OperatorTask | DiagnosticTask,
    artifact_dir: str | Path,
    *,
    worker_id: str = "worker",
    run: AdaptationRun | None = None,
    diagnosis: str = "Simulated fixture failure",
    repair_conclusion: Any = None,
    next_action: str = "RETRY",
    confidence: float = 1.0,
) -> dict[str, Any]:
    """Write hashed fixture artifacts bound to the supplied claimed task.

    Pass ``run`` when the simulation has an environment fingerprint. This helper
    refuses real runs; its artifacts only exercise scheduler bookkeeping.
    """
    if run is not None and (
        run.metadata.get("evidence_mode") != "simulation"
        or run.metadata.get("environment_required")
    ):
        raise ValueError("simulation_result requires an explicit simulation run")
    return _write_stage_result(
        Path(artifact_dir), task, worker=worker_id,
        environment_fingerprint=(
            run.environment.get("environment_proof", {}).get("fingerprint")
            if run is not None else None
        ),
        evidence_mode="simulation", diagnosis=diagnosis,
        repair_conclusion=repair_conclusion, next_action=next_action, confidence=confidence,
    )


def stage_result(
    root: Path,
    task: OperatorTask | DiagnosticTask,
    *,
    worker: str = "worker",
    environment_fingerprint: str | None = None,
    evidence_mode: str = "simulation",
) -> dict[str, Any]:
    """Build a test-only envelope; ``real`` models a real-mode contract in tests.

    Every artifact remains labeled as synthetic fixture data. Neither mode
    constitutes hardware evidence or may be used to promote an adaptation run.
    """
    return _write_stage_result(
        root, task, worker=worker, environment_fingerprint=environment_fingerprint,
        evidence_mode=evidence_mode,
    )


def _write_stage_result(
    root: Path,
    task: OperatorTask | DiagnosticTask,
    *,
    worker: str,
    environment_fingerprint: str | None,
    evidence_mode: str,
    diagnosis: str = "Simulated fixture failure",
    repair_conclusion: Any = None,
    next_action: str = "RETRY",
    confidence: float = 1.0,
) -> dict[str, Any]:
    if evidence_mode not in ("real", "simulation"):
        raise ValueError("fixture evidence_mode must be real or simulation")
    identity = {
        "task_id": task.task_id,
        "operator_key": task.operator_key,
        "stage": task.stage,
        "attempt": task.attempt,
        "evidence_mode": evidence_mode,
        "environment_fingerprint": environment_fingerprint,
    }
    root = (
        Path(task.input.get("workspace", {}).get("output", root)).resolve()
        / hashlib.sha256(task.task_id.encode()).hexdigest()[:16]
        / str(task.attempt)
    )
    root.mkdir(parents=True, exist_ok=True)
    result = {"schema_version": 1, "status": "PASS", "verdict": "PASS", **identity}
    evidence: dict[str, str] = {}
    hashes: dict[str, str] = {}
    report_key = "diagnosis_report" if task.stage == "diagnosis" else "independent_validation"

    def write(key: str, payload: dict[str, Any]) -> None:
        path = root / f"{key}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        evidence[key] = str(path)
        hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()

    fixture = {
        **identity,
        "verdict": "PASS",
        "simulation_notice": "Scheduler test fixture; no model, hardware or service validation.",
    }
    for key in STAGE_EVIDENCE[task.stage]:
        if key != report_key:
            write(key, {**fixture, "artifact_kind": key})
    if task.stage == "diagnosis":
        conclusion = {
            "diagnosis": diagnosis,
            "repair_conclusion": (
                repair_conclusion if repair_conclusion is not None
                else {"action": "retry simulated fixture"}
            ),
            "next_action": next_action,
            "confidence": confidence,
        }
        result.update(conclusion)
        write(report_key, {**fixture, **conclusion})
    else:
        write(report_key, {
            **fixture,
            "validator": f"{worker}:simulation-validator",
            "checks": [{"name": "simulated_fixture_contract", "passed": True}],
            "evidence_sha256": dict(hashes),
        })
    return {**result, "evidence": evidence, "evidence_sha256": hashes}
