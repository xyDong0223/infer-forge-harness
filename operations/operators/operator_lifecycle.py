"""Coordinate asynchronous XPU operator work and serialized integration.

This module deliberately does not generate or grade kernels. It creates durable
requests for the xpu-op-gen subagent, freezes a serving baseline, and records
whether one candidate is ready to be integrated against that baseline.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from core.paths import REPO_ROOT
from validators.operator_lifecycle_validator import (
    validate_baseline, validate_dispatch, validate_integration,
)



SUCCESS_STATES = {"DISPATCHED", "DISPATCH_SKIPPED", "BASELINE_FROZEN", "WAITING_FOR_CANDIDATE"}
TERMINAL_INTEGRATION_STATES = {"INTEGRATED_PASS", "INTEGRATED_REGRESSION", "ROLLED_BACK"}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _status(out: Path, state: str, **fields: Any) -> dict[str, Any]:
    payload = {"state": state, **fields}
    validator = {
        "dispatch_status.json": validate_dispatch,
        "baseline_status.json": validate_baseline,
        "integration_status.json": validate_integration,
    }.get(out.name)
    if validator is not None:
        errors = validator(payload)
        payload["validator"] = {"passed": not errors, "errors": errors}
        if errors and "scheduler_state" in payload:
            payload["state"] = "DISPATCH_BLOCKED" if out.name == "dispatch_status.json" else "OPERATORS_BLOCKED"
            payload["errors"] = [*payload.get("errors", []), *errors]
    _write(out, payload)
    return payload


def dispatch(
    gaps_path: Path, out_dir: Path, subject: str, baseline_id: str | None = None, *,
    scheduler_state: Path | None = None, run_id: str | None = None,
    operator_report: Path | None = None,
) -> dict:
    """Create one independently executable request per operator gap."""
    if scheduler_state is not None or run_id is not None:
        bridge = _scheduled_bridge(scheduler_state, run_id, subject)
        try:
            return bridge.dispatch(gaps_path, out_dir, operator_report)
        finally:
            bridge.close()
    if operator_report is not None:
        raise ValueError("--operator-report requires --scheduler-state and --run-id")
    source = _load(gaps_path)
    gaps = source.get("gaps", source.get("findings", [])) if isinstance(source, dict) else source
    gaps = gaps if isinstance(gaps, list) else []
    requests = []
    for index, gap in enumerate(gaps, start=1):
        if not isinstance(gap, dict):
            continue
        operator = gap.get("operator") or gap.get("symbol") or gap.get("name")
        if not operator and gap.get("class") == "CAPABILITY_MISSING":
            operator = gap.get("axis")
        if not operator:
            continue
        request = {
            "request_id": f"{subject}-op-{index:03d}",
            "subject": subject,
            "operator": operator,
            "gap": gap,
            "baseline_id": baseline_id,
            "status": "PENDING",
            "required_inputs": [
                "op_contract.json",
                "reference_provenance.json",
                "captured_arguments.jsonl",
                "environment_fingerprint.json",
            ],
        "completion_requires": [
            "generation_manifest.json",
            "build_report.json",
            "kernel_grade.json",
            "dispatch_report.json",
            ],
        }
        requests.append(request)
        _write(out_dir / "requests" / f"{request['request_id']}.json", request)
    state = "DISPATCHED" if requests else "DISPATCH_SKIPPED"
    return _status(
        out_dir / "dispatch_status.json",
        state,
        subject=subject,
        baseline_id=baseline_id,
        request_count=len(requests),
        requests=[request["request_id"] for request in requests],
        source=str(gaps_path),
    )


def freeze_baseline(
    service_path: Path,
    accuracy_path: Path,
    out_dir: Path,
    subject: str,
    environment: dict[str, str],
) -> dict:
    """Freeze the exact evidence that later candidate integrations must preserve."""
    service = _load(service_path)
    accuracy = _load(accuracy_path)
    service_state = service.get("state")
    accuracy_state = accuracy.get("state", accuracy.get("status"))
    baseline_id = f"{subject}-baseline-{_digest({'service': service, 'accuracy': accuracy})[:12]}"
    manifest = {
        "baseline_id": baseline_id,
        "subject": subject,
        "service": {"state": service_state, "artifact": str(service_path.resolve()),
                    "sha256": hashlib.sha256(service_path.read_bytes()).hexdigest()},
        "accuracy": {"state": accuracy_state, "artifact": str(accuracy_path.resolve()),
                     "sha256": hashlib.sha256(accuracy_path.read_bytes()).hexdigest()},
        "environment": dict(environment),
        "status": "FROZEN",
        "integration_policy": {
            "one_candidate_at_a_time": True,
            "require_kernel_pass": True,
            "require_dispatch_confirmation": True,
            "require_service_regression": True,
            "require_accuracy_regression": True,
            "rollback_on_failure": True,
        },
    }
    if service_state != "DEPLOYMENT_READY" or accuracy_state != "ACCURACY_PASS":
        return _status(
            out_dir / "baseline_status.json",
            "BASELINE_REJECTED",
            subject=subject,
            service_state=service_state,
            accuracy_state=accuracy_state,
            reason="service and accuracy must pass before freezing a baseline",
        )
    _write(out_dir / "baseline_manifest.json", manifest)
    return _status(
        out_dir / "baseline_status.json",
        "BASELINE_FROZEN",
        baseline_id=baseline_id,
        manifest=str(out_dir / "baseline_manifest.json"),
    )


def integration_decision(
    baseline_path: Path,
    candidate_path: Path | None,
    out_dir: Path,
    subject: str,
    *,
    scheduler_state: Path | None = None,
    run_id: str | None = None,
) -> dict:
    """Grade candidate readiness; actual service mutation remains an explicit step."""
    if scheduler_state is not None or run_id is not None:
        bridge = _scheduled_bridge(scheduler_state, run_id, subject)
        try:
            status = bridge.delivery_status()
            status["baseline_path"] = str(baseline_path.resolve())
            try:
                baseline = bridge.validate_baseline(baseline_path)
                status["baseline_id"] = baseline["baseline_id"]
            except (OSError, ValueError, KeyError, TypeError) as exc:
                status["state"] = "OPERATORS_BLOCKED"
                status["errors"].append(str(exc))
            return _status(out_dir / "integration_status.json", **status)
        finally:
            bridge.close()
    baseline = _load(baseline_path)
    if candidate_path is None:
        return _status(
            out_dir / "integration_status.json",
            "WAITING_FOR_CANDIDATE",
            subject=subject,
            baseline_id=baseline.get("baseline_id"),
            next_action="provide one candidate manifest",
        )
    candidate = _load(candidate_path)
    required = {
        "kernel_grade": "KERNEL_PASS",
        "dispatch_report": "DISPATCH_CONFIRMED",
        "package_swap": "PASS",
        "path_proof": "PASS",
        "service_regression": "PASS",
        "accuracy_regression": "PASS",
    }
    failures = {
        key: candidate.get(key)
        for key, expected in required.items()
        if candidate.get(key) != expected
    }
    evidence_fields = {
        "kernel_grade": "kernel_grade_report",
        "dispatch_report": "dispatch_report_path",
        "package_swap": "package_swap_report",
        "path_proof": "worker_path_log",
        "service_regression": "service_regression_report",
        "accuracy_regression": "accuracy_regression_report",
    }
    missing_evidence = {
        status: field for status, field in evidence_fields.items() if not candidate.get(field)
    }
    failures.update({f"evidence:{key}": value for key, value in missing_evidence.items()})
    state = "READY_FOR_INTEGRATION" if not failures else "CANDIDATE_REJECTED"
    return _status(
        out_dir / "integration_status.json",
        state,
        subject=subject,
        baseline_id=baseline.get("baseline_id"),
        candidate=str(candidate_path),
        failed_gates=failures,
        **{field: candidate.get(field) for field in evidence_fields.values()},
        next_action="integrate and run regression" if not failures else "return candidate for rework",
    )


def _scheduled_bridge(state: Path | None, run_id: str | None, subject: str):
    from engine.graph_bridge import GraphSchedulerBridge
    from engine.scheduler import EventStore

    if state is None or not run_id:
        raise ValueError("--scheduler-state and --run-id must be supplied together")
    store = EventStore(state, readonly=True)
    try:
        run = store.run(run_id)
        if run is None:
            raise ValueError(f"unknown adaptation run: {run_id}")
        root = run.metadata.get("artifact_root")
        if not root:
            raise ValueError("scheduled graph requires a run artifact_root")
    finally:
        store.close()
    return GraphSchedulerBridge(state, run_id, subject, Path(root), {})


def execute(args) -> int:

    if args.action == "dispatch":
        dispatch(
            args.gaps, args.out, args.subject, args.baseline_id,
            scheduler_state=args.scheduler_state, run_id=args.run_id,
            operator_report=args.operator_report,
        )
    elif args.action == "freeze-baseline":
        environment = dict(pair.split("=", 1) for pair in args.env)
        freeze_baseline(args.service, args.accuracy, args.out, args.subject, environment)
    else:
        integration_decision(
            args.baseline,
            Path(args.candidate) if args.candidate else None,
            args.out,
            args.subject,
            scheduler_state=args.scheduler_state,
            run_id=args.run_id,
        )
    return 0
