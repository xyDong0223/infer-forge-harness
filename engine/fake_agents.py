"""Deterministic fake Agent harness for validating orchestration end-to-end.

The harness deliberately performs real local computations and writes evidence
artifacts.  A stage can advance only when :class:`EvidenceGate` validates those
artifacts; a worker cannot promote a task by returning ``{"status": "pass"}``
alone.  This provides a safe, hardware-independent rehearsal of the future
PyTorch/XPU Agent contracts. Only explicitly configured simulation runs are
accepted; these artifacts never prove real device or service readiness.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.storage import ArtifactStore, ensure_external

from .contracts import OperatorSpec, OperatorTask
from .scheduler import TaskScheduler


class EvidenceError(ValueError):
    """Raised when a fake Agent result lacks independently checked evidence."""


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    return str(ArtifactStore(path.parent).write_json(path.name, payload, overwrite=True))


def _read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise EvidenceError(f"missing evidence artifact: {p}")
    try:
        value = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"invalid evidence artifact: {p}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"evidence artifact must be an object: {p}")
    return value


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _numel(shape: Any) -> int:
    if isinstance(shape, int):
        return max(1, shape)
    if isinstance(shape, (list, tuple)):
        n = 1
        for dim in shape:
            if not isinstance(dim, int) or dim <= 0:
                raise EvidenceError(f"fake shape must contain positive integers: {shape!r}")
            n *= dim
        return n
    # Symbolic shapes are valid contracts, but use a small deterministic probe.
    return 4


def _apply_fake_op(values: list[float], spec: OperatorSpec) -> list[float]:
    op = str(spec.semantics.get("fake_op", "identity")).lower()
    if op == "identity":
        return list(values)
    if op == "double":
        return [2.0 * x for x in values]
    if op == "negate":
        return [-x for x in values]
    if op == "square":
        return [x * x for x in values]
    if op == "add":
        # A scalarized add is useful for exercising a multi-input contract.
        return [sum(values)]
    raise EvidenceError(f"unsupported fake_op: {op}")


def _close(a: list[float], b: list[float], tol: float = 1e-8) -> bool:
    return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))


class EvidenceGate:
    """Validate stage-specific evidence before a task is completed."""

    @staticmethod
    def _common(result: dict[str, Any]) -> None:
        if str(result.get("verdict", "")).lower() != "pass":
            raise EvidenceError("result verdict must be pass")
        evidence = result.get("evidence")
        if not isinstance(evidence, dict):
            raise EvidenceError("result must include an evidence object")

    @classmethod
    def torch(cls, result: dict[str, Any], spec: OperatorSpec) -> None:
        cls._common(result)
        evidence = result["evidence"]
        ref_path = evidence.get("reference_artifact")
        validation_path = evidence.get("independent_validation")
        if not ref_path or not validation_path:
            raise EvidenceError("torch evidence requires reference and independent validation artifacts")
        ref = _read_json(ref_path)
        validation = _read_json(validation_path)
        if ref.get("operator_key") != spec.operator_key:
            raise EvidenceError("reference artifact operator key mismatch")
        if str(validation.get("verdict", "")).lower() != "pass" or not validation.get("checks"):
            raise EvidenceError("independent torch validation did not pass")
        if result.get("evidence_sha256", {}).get("reference_artifact") != _sha256(ref_path):
            raise EvidenceError("reference artifact hash mismatch")
        if not all(c.get("passed") is True for c in validation["checks"]):
            raise EvidenceError("torch validation contains a failed check")

    @classmethod
    def xpu(cls, result: dict[str, Any], spec: OperatorSpec) -> None:
        cls._common(result)
        evidence = result["evidence"]
        device_path = evidence.get("device_test")
        validation_path = evidence.get("independent_validation")
        if not device_path or not validation_path:
            raise EvidenceError("xpu evidence requires device test and independent validation artifacts")
        device = _read_json(device_path)
        validation = _read_json(validation_path)
        if device.get("operator_key") != spec.operator_key:
            raise EvidenceError("device artifact operator key mismatch")
        if device.get("executed") is not True or not str(device.get("device", "")).startswith("fake-xpu"):
            raise EvidenceError("device test was not executed on fake XPU")
        if str(validation.get("verdict", "")).lower() != "pass" or not validation.get("checks"):
            raise EvidenceError("independent device validation did not pass")
        if not all(c.get("passed") is True for c in device.get("checks", [])):
            raise EvidenceError("device test contains a failed check")
        if not all(c.get("passed") is True for c in validation["checks"]):
            raise EvidenceError("device validation contains a failed check")

    @classmethod
    def integration(cls, result: dict[str, Any], spec: OperatorSpec) -> None:
        cls._common(result)
        evidence = result["evidence"]
        report_path = evidence.get("integration_report")
        if not report_path:
            raise EvidenceError("integration evidence requires an integration report")
        report = _read_json(report_path)
        if report.get("operator_key") != spec.operator_key or report.get("service_smoke") is not True:
            raise EvidenceError("service smoke evidence is missing")
        checks = report.get("checks")
        if not checks or not all(c.get("passed") is True for c in checks):
            raise EvidenceError("integration checks did not pass")
        if not isinstance(report.get("latency_ms"), (float, int)) or report["latency_ms"] <= 0:
            raise EvidenceError("integration report lacks measured latency")


@dataclass
class FakeAgentCall:
    agent_id: str
    stage: str
    task_id: str
    action: str
    evidence: dict[str, Any] = field(default_factory=dict)


class FakeAgentHarness:
    """Run torch Agent, XPU Agent, and independent validators locally."""

    def __init__(self, scheduler: TaskScheduler, artifact_dir: str | Path):
        self.scheduler = scheduler
        self.artifact_dir = ensure_external(artifact_dir)
        self.calls: list[FakeAgentCall] = []

    def _artifact(self, task: OperatorTask, name: str) -> Path:
        workspace = task.input.get("workspace")
        if workspace is not None:
            return ArtifactStore(workspace["output"]).path(f"{name}.json")
        task_key = hashlib.sha256(task.task_id.encode()).hexdigest()[:16]
        return ArtifactStore(self.artifact_dir).path(f"{task_key}/{task.attempt}/{name}.json")

    def _identity(self, task: OperatorTask) -> dict[str, Any]:
        run = self.scheduler.store.run(task.run_id)
        if run is None or run.metadata.get("evidence_mode") != "simulation" or run.metadata.get("environment_required"):
            raise EvidenceError("fake harness requires an explicit simulation run")
        return {
            "task_id": task.task_id,
            "operator_key": task.operator_key,
            "stage": task.stage,
            "attempt": task.attempt,
            "evidence_mode": "simulation",
            "environment_fingerprint": run.environment.get("environment_proof", {}).get("fingerprint"),
        }

    def _write_artifact(self, task: OperatorTask, name: str, payload: dict[str, Any]) -> str:
        return _write_json(self._artifact(task, name), {
            **payload,
            **self._identity(task),
            "simulation_notice": "Local Python rehearsal; not real XPU or service evidence.",
        })

    def _envelope(self, task: OperatorTask, spec: OperatorSpec, result: dict[str, Any]) -> dict[str, Any]:
        """Bind the already-gated local checks and artifacts to this lease attempt."""
        evidence = dict(result["evidence"])
        if task.stage == "integration":
            report = _read_json(evidence["integration_report"])
            validation = {
                "validator": "fake-integration-validator",
                "checks": [{
                    "name": "simulated_service_output_recomputed",
                    "passed": _close(report["outputs"], _apply_fake_op(report["inputs"], spec)),
                }],
            }
        else:
            validation = _read_json(evidence["independent_validation"])
        checks = validation["checks"]
        if not checks or not all(check.get("passed") is True for check in checks):
            raise EvidenceError("independent simulation validation failed")
        hashes = {key: _sha256(path) for key, path in evidence.items() if key != "independent_validation"}
        evidence["independent_validation"] = self._write_artifact(task, "independent-validation", {
            **validation,
            "schema_version": 1,
            "verdict": "PASS",
            "evidence_sha256": hashes,
        })
        return {
            "schema_version": 1,
            "status": "PASS",
            "verdict": "PASS",
            **self._identity(task),
            "evidence": evidence,
            "evidence_sha256": {key: _sha256(path) for key, path in evidence.items()},
        }

    def _complete(self, task: OperatorTask, worker: str, result: dict[str, Any]) -> None:
        completed = self.scheduler.complete(
            task.task_id, worker_id=worker, lease_token=task.lease_token, result=result,
        )
        if completed.status != "succeeded":
            raise EvidenceError(f"scheduler rejected simulated {task.stage} evidence: {completed.output}")

    def _claim(self, stage: str, worker: str) -> OperatorTask:
        claimed = self.scheduler.claim_ready(worker, stage=stage, limit=1)
        if len(claimed) != 1:
            raise RuntimeError(f"no ready {stage} task for {worker}")
        return claimed[0]

    def run(self, run_id: str, spec: OperatorSpec) -> dict[str, Any]:
        """Discover and run one operator through all fake stages.

        Each Agent writes an artifact, and an independent validator reads that
        artifact and recomputes the expected output.  The scheduler receives a
        result only after the stage evidence gate accepts it.
        """
        run = self.scheduler.store.run(run_id)
        if run is None:
            raise KeyError(f"unknown run: {run_id}")
        if run.metadata.get("evidence_mode") != "simulation" or run.metadata.get("environment_required"):
            raise EvidenceError("fake harness requires an explicit simulation run")
        self.scheduler.discover_operator(run_id, spec)
        torch_task = self._claim("torch", "fake-torch-agent")
        torch_result = self._torch_agent(torch_task, spec)
        EvidenceGate.torch(torch_result, spec)
        torch_result = self._envelope(torch_task, spec, torch_result)
        self._complete(torch_task, "fake-torch-agent", torch_result)

        xpu_task = self._claim("xpu", "fake-xpu-agent")
        xpu_result = self._xpu_agent(xpu_task, spec, torch_result)
        EvidenceGate.xpu(xpu_result, spec)
        xpu_result = self._envelope(xpu_task, spec, xpu_result)
        self._complete(xpu_task, "fake-xpu-agent", xpu_result)

        integration_task = self._claim("integration", "fake-integration-agent")
        integration_result = self._integration_agent(integration_task, spec, torch_result, xpu_result)
        EvidenceGate.integration(integration_result, spec)
        integration_result = self._envelope(integration_task, spec, integration_result)
        self._complete(integration_task, "fake-integration-agent", integration_result)
        return {
            "run_id": run_id,
            "operator_key": spec.operator_key,
            "verdict": "pass",
            "evidence_mode": "simulation",
            "calls": [call.__dict__ for call in self.calls],
            "tasks": [
                {"stage": stage, "status": self.scheduler.store.get_task(f"{run_id}:{spec.operator_key}:{stage}").status}
                for stage in ("torch", "xpu", "integration")
            ],
            "artifacts": {"torch": torch_result["evidence"], "xpu": xpu_result["evidence"], "integration": integration_result["evidence"]},
        }

    def _torch_agent(self, task: OperatorTask, spec: OperatorSpec) -> dict[str, Any]:
        values = [round((i + 1) / 10, 4) for i in range(_numel(spec.inputs[0].shape))]
        output = _apply_fake_op(values, spec)
        ref_path = self._write_artifact(task, "torch.reference", {"inputs": values, "outputs": output, "implementation": "fake-pytorch"})
        self.calls.append(FakeAgentCall("fake-torch-agent", "torch", task.task_id, "generate_reference", {"path": str(ref_path)}))

        reference = _read_json(ref_path)
        expected = _apply_fake_op(reference["inputs"], spec)
        checks = [{"name": "reference_matches_independent_recompute", "passed": _close(reference["outputs"], expected)}]
        focused_path = self._write_artifact(task, "torch.focused-tests", {"checks": checks})
        validation_path = self._write_artifact(task, "independent-validation", {"verdict": "pass" if all(c["passed"] for c in checks) else "fail", "checks": checks, "validator": "fake-independent-validator"})
        self.calls.append(FakeAgentCall("fake-independent-validator", "torch", task.task_id, "validate_reference", {"path": str(validation_path)}))
        return {
            "verdict": "pass",
            "evidence": {"reference_artifact": ref_path, "focused_tests": focused_path, "independent_validation": validation_path},
            "evidence_sha256": {"reference_artifact": _sha256(ref_path)},
        }

    def _xpu_agent(self, task: OperatorTask, spec: OperatorSpec, torch_result: dict[str, Any]) -> dict[str, Any]:
        ref = _read_json(torch_result["evidence"]["reference_artifact"])
        values = list(ref["inputs"])
        output = _apply_fake_op(values, spec)
        checks = [{"name": "xpu_matches_torch_reference", "passed": _close(output, list(ref["outputs"]))}]
        device_path = self._write_artifact(task, "xpu.device-test", {"device": "fake-xpu:simulator", "executed": True, "actual_xpu_execution": False, "outputs": output, "checks": checks})
        self.calls.append(FakeAgentCall("fake-xpu-agent", "xpu", task.task_id, "run_device_test", {"path": str(device_path), "device": "fake-xpu:simulator"}))

        device = _read_json(device_path)
        validation_checks = [{"name": "device_execution_marker", "passed": device.get("executed") is True}, {"name": "device_output_recomputed", "passed": _close(device["outputs"], _apply_fake_op(values, spec))}]
        validation_path = self._write_artifact(task, "independent-validation", {"verdict": "pass" if all(c["passed"] for c in validation_checks) else "fail", "checks": validation_checks, "validator": "fake-device-validator"})
        self.calls.append(FakeAgentCall("fake-device-validator", "xpu", task.task_id, "validate_device_test", {"path": str(validation_path)}))
        return {"verdict": "pass", "evidence": {
            "device_test": device_path,
            "independent_validation": validation_path,
            "build_record": self._write_artifact(task, "xpu.build", {"implementation": "local-python-simulator", "compiled": False}),
            "registration_record": self._write_artifact(task, "xpu.registration", {"dispatch": "_apply_fake_op", "torch_dispatcher_registered": False}),
            "dispatch_report": self._write_artifact(task, "xpu.dispatch", {"dispatch": "_apply_fake_op", "actual_xpu_execution": False, "checks": checks}),
        }}

    def _integration_agent(self, task: OperatorTask, spec: OperatorSpec, torch_result: dict[str, Any], xpu_result: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        torch = _read_json(torch_result["evidence"]["reference_artifact"])
        device = _read_json(xpu_result["evidence"]["device_test"])
        checks = [{"name": "service_output_matches_device", "passed": _close(list(torch["outputs"]), list(device["outputs"]))}, {"name": "dispatch_marker_present", "passed": device.get("executed") is True}]
        latency_ms = max((time.perf_counter() - started) * 1000.0, 0.001)
        report_path = self._write_artifact(task, "integration.report", {"service_smoke": all(c["passed"] for c in checks), "latency_ms": latency_ms, "checks": checks, "implementation": "fake-service", "inputs": torch["inputs"], "outputs": device["outputs"]})
        self.calls.append(FakeAgentCall("fake-integration-agent", "integration", task.task_id, "run_service_smoke", {"path": str(report_path)}))
        return {"verdict": "pass", "evidence": {
            "integration_report": report_path,
            "service_regression": self._write_artifact(task, "integration.service", {"implementation": "fake-service", "actual_service_request": False, "checks": checks}),
            "accuracy_regression": self._write_artifact(task, "integration.accuracy", {"checks": checks}),
            "fallback_report": self._write_artifact(task, "integration.fallback", {"execution_path": "local-python-simulation", "cpu_execution": True, "actual_xpu_execution": False}),
        }}


def run_fake_adaptation(scheduler: TaskScheduler, run_id: str, spec: OperatorSpec, artifact_dir: str | Path) -> dict[str, Any]:
    """Convenience wrapper used by demos and tests."""
    return FakeAgentHarness(scheduler, artifact_dir).run(run_id, spec)


__all__ = ["EvidenceError", "EvidenceGate", "FakeAgentCall", "FakeAgentHarness", "run_fake_adaptation"]
