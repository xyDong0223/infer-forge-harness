import hashlib
import json
from pathlib import Path

import pytest

from engine import (
    AdaptationRun,
    EvidenceError,
    EvidenceGate,
    EventStore,
    FakeAgentHarness,
    IOSpec,
    OperatorSpec,
    TaskScheduler,
)
from engine.result_validation import STAGE_EVIDENCE, validate_result


def _spec() -> OperatorSpec:
    return OperatorSpec(
        operator_id="demo_double",
        model_id="DeepSeek-V4.1",
        model_revision="rev-1",
        plugin_revision="kunlun-1",
        backend="kunlun-p800",
        inputs=[IOSpec("x", "float32", [2, 2], "contiguous")],
        outputs=[IOSpec("y", "float32", [2, 2], "contiguous")],
        semantics={"reference": "torch", "fake_op": "double"},
    )


def test_fake_agents_call_independent_validator_and_fake_xpu(tmp_path: Path):
    scheduler = TaskScheduler(EventStore(tmp_path / "state.db"))
    run = scheduler.create_run(AdaptationRun(
        run_id="r", model_id="DeepSeek-V4.1", backend="kunlun-p800",
        metadata={"evidence_mode": "simulation"},
    ))
    result = FakeAgentHarness(scheduler, tmp_path / "artifacts").run("r", _spec())

    assert result["verdict"] == "pass"
    assert result["evidence_mode"] == "simulation"
    assert [(c["agent_id"], c["action"]) for c in result["calls"]] == [
        ("fake-torch-agent", "generate_reference"),
        ("fake-independent-validator", "validate_reference"),
        ("fake-xpu-agent", "run_device_test"),
        ("fake-device-validator", "validate_device_test"),
        ("fake-integration-agent", "run_service_smoke"),
    ]
    assert all(item["status"] == "succeeded" for item in result["tasks"])
    for task in scheduler.store.tasks("r"):
        output = task.output
        worker = output["_submission"]["worker"]
        assert validate_result(task, run, output, worker) == []
        assert output["schema_version"] == 1
        assert output["status"] == output["verdict"] == "PASS"
        assert set(output["evidence"]) == set(STAGE_EVIDENCE[task.stage])
        report = json.loads(Path(output["evidence"]["independent_validation"]).read_text())
        assert report["validator"] != worker
        for key, path in output["evidence"].items():
            artifact = json.loads(Path(path).read_text())
            assert artifact["evidence_mode"] == "simulation"
            assert artifact["environment_fingerprint"] is None
            assert artifact["task_id"] == task.task_id
            assert artifact["attempt"] == task.attempt
            assert "not real XPU or service evidence" in artifact["simulation_notice"]
            assert output["evidence_sha256"][key] == hashlib.sha256(Path(path).read_bytes()).hexdigest()
    device = json.loads(Path(result["artifacts"]["xpu"]["device_test"]).read_text())
    assert device["executed"] is True
    assert device["device"].startswith("fake-xpu")
    assert device["actual_xpu_execution"] is False
    build = json.loads(Path(result["artifacts"]["xpu"]["build_record"]).read_text())
    assert build["compiled"] is False
    fallback = json.loads(Path(result["artifacts"]["integration"]["fallback_report"]).read_text())
    assert fallback["cpu_execution"] is True
    assert fallback["actual_xpu_execution"] is False


@pytest.mark.parametrize("metadata", [{}, {"evidence_mode": "simulation", "environment_required": True}])
def test_fake_harness_refuses_non_simulation_and_environment_backed_runs(tmp_path: Path, metadata):
    scheduler = TaskScheduler(tmp_path / "state.db")
    scheduler.create_run(run_id="r", model_id="m", metadata=metadata)
    before = scheduler.store.run("r").to_dict()
    harness = FakeAgentHarness(scheduler, tmp_path / "artifacts")

    with pytest.raises(EvidenceError, match="explicit simulation run"):
        harness.run("r", _spec())

    assert scheduler.store.run("r").to_dict() == before
    assert scheduler.store.tasks("r") == []
    assert harness.calls == []
    assert not (tmp_path / "artifacts").exists()


def test_fake_harness_preserves_evidence_from_other_runs(tmp_path: Path):
    scheduler = TaskScheduler(tmp_path / "state.db")
    for run_id in ("first", "second"):
        scheduler.create_run(run_id=run_id, model_id="m", metadata={"evidence_mode": "simulation"})
        FakeAgentHarness(scheduler, tmp_path / "artifacts").run(run_id, _spec())

    for run_id in ("first", "second"):
        run = scheduler.store.run(run_id)
        for task in scheduler.store.tasks(run_id):
            assert validate_result(task, run, task.output, task.output["_submission"]["worker"]) == []


def test_fake_harness_does_not_report_success_after_scheduler_rejects_result(tmp_path: Path, monkeypatch):
    scheduler = TaskScheduler(tmp_path / "state.db")
    scheduler.create_run(run_id="r", model_id="m", metadata={"evidence_mode": "simulation"})
    harness = FakeAgentHarness(scheduler, tmp_path / "artifacts")
    monkeypatch.setattr(harness, "_envelope", lambda task, spec, result: {"status": "PASS"})

    with pytest.raises(EvidenceError, match="scheduler rejected simulated torch evidence"):
        harness.run("r", _spec())

    assert scheduler.pending_tasks("r", stage="xpu") == []
    assert len(scheduler.pending_tasks("r", stage="diagnosis")) == 1


def test_evidence_gate_rejects_handwritten_pass():
    with pytest.raises(EvidenceError):
        EvidenceGate.xpu({"verdict": "pass", "evidence": {}}, _spec())


def test_evidence_gate_rejects_missing_or_failed_device_check(tmp_path: Path):
    device = tmp_path / "device.json"
    validation = tmp_path / "validation.json"
    device.write_text(json.dumps({"operator_key": _spec().operator_key, "device": "fake-xpu:simulator", "executed": True, "checks": [{"passed": False}]}))
    validation.write_text(json.dumps({"verdict": "pass", "checks": [{"passed": True}]}))
    with pytest.raises(EvidenceError):
        EvidenceGate.xpu({"verdict": "pass", "evidence": {"device_test": str(device), "independent_validation": str(validation)}}, _spec())
