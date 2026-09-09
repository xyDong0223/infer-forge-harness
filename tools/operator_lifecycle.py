"""Coordinate asynchronous XPU operator work and serialized integration.

This module deliberately does not generate or grade kernels. It creates durable
requests for the xpu-op-gen subagent, freezes a serving baseline, and records
whether one candidate is ready to be integrated against that baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


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
    _write(out, payload)
    return payload


def dispatch(gaps_path: Path, out_dir: Path, subject: str, baseline_id: str | None = None) -> dict:
    """Create one independently executable request per operator gap."""
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
    accuracy_state = accuracy.get("state")
    baseline_id = f"{subject}-baseline-{_digest({'service': service, 'accuracy': accuracy})[:12]}"
    manifest = {
        "baseline_id": baseline_id,
        "subject": subject,
        "service": {"state": service_state, "artifact": str(service_path)},
        "accuracy": {"state": accuracy_state, "artifact": str(accuracy_path)},
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
) -> dict:
    """Grade candidate readiness; actual service mutation remains an explicit step."""
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
        next_action="integrate and run regression" if not failures else "return candidate for rework",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    dispatch_parser = sub.add_parser("dispatch")
    dispatch_parser.add_argument("--gaps", type=Path, required=True)
    dispatch_parser.add_argument("--out", type=Path, required=True)
    dispatch_parser.add_argument("--subject", required=True)
    dispatch_parser.add_argument("--baseline-id")

    baseline_parser = sub.add_parser("freeze-baseline")
    baseline_parser.add_argument("--service", type=Path, required=True)
    baseline_parser.add_argument("--accuracy", type=Path, required=True)
    baseline_parser.add_argument("--out", type=Path, required=True)
    baseline_parser.add_argument("--subject", required=True)
    baseline_parser.add_argument("--env", action="append", default=[])

    integration_parser = sub.add_parser("integrate")
    integration_parser.add_argument("--baseline", type=Path, required=True)
    integration_parser.add_argument("--candidate", type=Path)
    integration_parser.add_argument("--out", type=Path, required=True)
    integration_parser.add_argument("--subject", required=True)
    args = parser.parse_args()

    if args.action == "dispatch":
        dispatch(args.gaps, args.out, args.subject, args.baseline_id)
    elif args.action == "freeze-baseline":
        environment = dict(pair.split("=", 1) for pair in args.env)
        freeze_baseline(args.service, args.accuracy, args.out, args.subject, environment)
    else:
        integration_decision(
            args.baseline,
            Path(args.candidate) if args.candidate else None,
            args.out,
            args.subject,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
