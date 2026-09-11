from pathlib import Path

import pytest

from orchestration import (
    AdaptationRun,
    EvidenceError,
    EvidenceGate,
    EventStore,
    FakeAgentHarness,
    IOSpec,
    OperatorSpec,
    TaskScheduler,
)


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
    scheduler.create_run(AdaptationRun(run_id="r", model_id="DeepSeek-V4.1", backend="kunlun-p800"))
    result = FakeAgentHarness(scheduler, tmp_path / "artifacts").run("r", _spec())

    assert result["verdict"] == "pass"
    assert [(c["agent_id"], c["action"]) for c in result["calls"]] == [
        ("fake-torch-agent", "generate_reference"),
        ("fake-independent-validator", "validate_reference"),
        ("fake-xpu-agent", "run_device_test"),
        ("fake-device-validator", "validate_device_test"),
        ("fake-integration-agent", "run_service_smoke"),
    ]
    assert all(item["status"] == "succeeded" for item in result["tasks"])
    device = __import__("json").loads(Path(result["artifacts"]["xpu"]["device_test"]).read_text())
    assert device["executed"] is True
    assert device["device"].startswith("fake-xpu")


def test_evidence_gate_rejects_handwritten_pass():
    with pytest.raises(EvidenceError):
        EvidenceGate.xpu({"verdict": "pass", "evidence": {}}, _spec())


def test_evidence_gate_rejects_missing_or_failed_device_check(tmp_path: Path):
    device = tmp_path / "device.json"
    validation = tmp_path / "validation.json"
    device.write_text(__import__("json").dumps({"operator_key": _spec().operator_key, "device": "fake-xpu:simulator", "executed": True, "checks": [{"passed": False}]}))
    validation.write_text(__import__("json").dumps({"verdict": "pass", "checks": [{"passed": True}]}))
    with pytest.raises(EvidenceError):
        EvidenceGate.xpu({"verdict": "pass", "evidence": {"device_test": str(device), "independent_validation": str(validation)}}, _spec())
