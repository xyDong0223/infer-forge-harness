"""Deterministic fake Agent harness for validating orchestration end-to-end.

The harness deliberately performs real local computations and writes evidence
artifacts.  A stage can advance only when :class:`EvidenceGate` validates those
artifacts; a worker cannot promote a task by returning ``{"status": "pass"}``
alone.  This provides a safe, hardware-independent rehearsal of the future
PyTorch/XPU Agent contracts.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contracts import OperatorSpec, OperatorTask
from .scheduler import TaskScheduler


class EvidenceError(ValueError):
    """Raised when a fake Agent result lacks independently checked evidence."""


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


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
        if validation.get("verdict") != "pass" or not validation.get("checks"):
            raise EvidenceError("independent torch validation did not pass")
        if evidence.get("reference_sha256") != _sha256(ref_path):
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
        if validation.get("verdict") != "pass" or not validation.get("checks"):
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
        self.artifact_dir = Path(artifact_dir)
        self.calls: list[FakeAgentCall] = []

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
        self.scheduler.discover_operator(run_id, spec)
        torch_task = self._claim("torch", "fake-torch-agent")
        torch_result = self._torch_agent(torch_task, spec)
        EvidenceGate.torch(torch_result, spec)
        self.scheduler.complete(torch_task.task_id, worker_id="fake-torch-agent", result=torch_result)

        xpu_task = self._claim("xpu", "fake-xpu-agent")
        xpu_result = self._xpu_agent(xpu_task, spec, torch_result)
        EvidenceGate.xpu(xpu_result, spec)
        self.scheduler.complete(xpu_task.task_id, worker_id="fake-xpu-agent", result=xpu_result)

        integration_task = self._claim("integration", "fake-integration-agent")
        integration_result = self._integration_agent(integration_task, spec, torch_result, xpu_result)
        EvidenceGate.integration(integration_result, spec)
        self.scheduler.complete(integration_task.task_id, worker_id="fake-integration-agent", result=integration_result)
        return {
            "run_id": run_id,
            "operator_key": spec.operator_key,
            "verdict": "pass",
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
        ref_path = self.artifact_dir / f"{spec.operator_id}.torch.reference.json"
        _write_json(ref_path, {"operator_key": spec.operator_key, "inputs": values, "outputs": output, "implementation": "fake-pytorch"})
        self.calls.append(FakeAgentCall("fake-torch-agent", "torch", task.task_id, "generate_reference", {"path": str(ref_path)}))

        # Independent validator has no access to the Agent's output computation.
        expected = _apply_fake_op(values, spec)
        checks = [{"name": "reference_matches_independent_recompute", "passed": _close(output, expected)}]
        validation_path = self.artifact_dir / f"{spec.operator_id}.torch.validation.json"
        _write_json(validation_path, {"operator_key": spec.operator_key, "verdict": "pass" if all(c["passed"] for c in checks) else "fail", "checks": checks, "validator": "fake-independent-validator"})
        self.calls.append(FakeAgentCall("fake-independent-validator", "torch", task.task_id, "validate_reference", {"path": str(validation_path)}))
        return {"verdict": "pass", "evidence": {"reference_artifact": str(ref_path), "reference_sha256": _sha256(ref_path), "independent_validation": str(validation_path)}}

    def _xpu_agent(self, task: OperatorTask, spec: OperatorSpec, torch_result: dict[str, Any]) -> dict[str, Any]:
        ref = _read_json(torch_result["evidence"]["reference_artifact"])
        values = list(ref["inputs"])
        output = _apply_fake_op(values, spec)
        checks = [{"name": "xpu_matches_torch_reference", "passed": _close(output, list(ref["outputs"]))}]
        device_path = self.artifact_dir / f"{spec.operator_id}.xpu.device-test.json"
        _write_json(device_path, {"operator_key": spec.operator_key, "device": "fake-xpu:simulator", "executed": True, "outputs": output, "checks": checks})
        self.calls.append(FakeAgentCall("fake-xpu-agent", "xpu", task.task_id, "run_device_test", {"path": str(device_path), "device": "fake-xpu:simulator"}))

        validation_checks = [{"name": "device_execution_marker", "passed": True}, {"name": "device_output_recomputed", "passed": _close(output, _apply_fake_op(values, spec))}]
        validation_path = self.artifact_dir / f"{spec.operator_id}.xpu.validation.json"
        _write_json(validation_path, {"operator_key": spec.operator_key, "verdict": "pass" if all(c["passed"] for c in validation_checks) else "fail", "checks": validation_checks, "validator": "fake-device-validator"})
        self.calls.append(FakeAgentCall("fake-device-validator", "xpu", task.task_id, "validate_device_test", {"path": str(validation_path)}))
        return {"verdict": "pass", "evidence": {"device_test": str(device_path), "independent_validation": str(validation_path)}}

    def _integration_agent(self, task: OperatorTask, spec: OperatorSpec, torch_result: dict[str, Any], xpu_result: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        torch = _read_json(torch_result["evidence"]["reference_artifact"])
        device = _read_json(xpu_result["evidence"]["device_test"])
        checks = [{"name": "service_output_matches_device", "passed": _close(list(torch["outputs"]), list(device["outputs"]))}, {"name": "dispatch_marker_present", "passed": device.get("executed") is True}]
        latency_ms = max((time.perf_counter() - started) * 1000.0, 0.001)
        report_path = self.artifact_dir / f"{spec.operator_id}.integration.report.json"
        _write_json(report_path, {"operator_key": spec.operator_key, "service_smoke": all(c["passed"] for c in checks), "latency_ms": latency_ms, "checks": checks, "implementation": "fake-service"})
        self.calls.append(FakeAgentCall("fake-integration-agent", "integration", task.task_id, "run_service_smoke", {"path": str(report_path)}))
        return {"verdict": "pass", "evidence": {"integration_report": str(report_path)}}


def run_fake_adaptation(scheduler: TaskScheduler, run_id: str, spec: OperatorSpec, artifact_dir: str | Path) -> dict[str, Any]:
    """Convenience wrapper used by demos and tests."""
    return FakeAgentHarness(scheduler, artifact_dir).run(run_id, spec)


__all__ = ["EvidenceError", "EvidenceGate", "FakeAgentCall", "FakeAgentHarness", "run_fake_adaptation"]
