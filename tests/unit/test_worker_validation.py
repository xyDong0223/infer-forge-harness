"""Actual subprocess measurements, independent reference and receipt-bound output."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from engine.contracts import OperatorSpec
from engine.managed_validation import freeze_candidate
from engine.result_validation import STAGE_EVIDENCE
from engine.scheduler import TaskScheduler
from operations.validation.managed_worker import (
    ValidationContractError, check_supported_mode, grade_observations, validation_plan,
)
from operations.validation.worker_probe import measure
from runners.worker_validation import validate_worker


CANDIDATE = "def run_case(inputs):\n    return {'y': [value * 2.0 for value in inputs['x']]}\n"
REFERENCE = "def run_case(inputs):\n    return {'y': [value + value for value in inputs['x']]}\n"


def make_spec(root):
    root.mkdir(parents=True, exist_ok=True)
    reference = root / "independent_reference.py"
    reference.write_text(REFERENCE)
    geometry = {"shape": [4], "dtype": "float64", "layout": "contiguous"}
    contract = {
        "schema_version": 1, "candidate_entry": "candidate.py",
        "reference": {"entry": str(reference), "files": {
            str(reference): hashlib.sha256(reference.read_bytes()).hexdigest()},
            "provenance": "Independently written addition oracle, not candidate multiplication"},
        "cases": [{"id": "signed", "inputs": {"x": {
            **geometry, "values": [-1.0, 0.0, 1.0, 2.0]}}, "outputs": {"y": geometry}}],
        "thresholds": {"y": {"max_relative_l2": 1e-8}},
        "negative_control": {"kind": "offset", "value": 1.0},
        "expected_dispatch": {"symbol": "run_case", "device": "simulation-cpu", "ranks": [0]},
        "fallback": {"allowed_devices": ["simulation-cpu"]},
        "service": {"require_http_status": 200},
    }
    return OperatorSpec(
        operator_id="scale", model_id="demo", model_revision="model-v1",
        plugin_revision="plugin-v1", backend="kunlun",
        inputs=[{"name": "x", **geometry}], outputs=[{"name": "y", **geometry}],
        semantics={"formula": "y = 2 * x", "fake_op": "double", "validation": contract},
    )


def prepare_candidate(scheduler, task, source=CANDIDATE):
    output = Path(task.input["workspace"]["output"])
    root = output / "candidate"
    root.mkdir()
    (root / "candidate.py").write_text(source)
    evidence = {}
    for kind in STAGE_EVIDENCE[task.stage]:
        if kind == "independent_validation":
            continue
        if kind == "reference_artifact":
            path = root / "candidate.py"
        else:
            path = output / f"{kind}.json"
            path.write_text(json.dumps({"producer_note": kind, "simulation": True}))
        evidence[kind] = str(path)
    candidate = freeze_candidate(scheduler, task.task_id, "controller", task.lease_token,
                                 "implementer", root, "plugin-v1")
    return candidate, evidence


@pytest.fixture
def workflow(tmp_path):
    scheduler = TaskScheduler(tmp_path / "state.sqlite")
    scheduler.create_run(run_id="r", model_id="demo", model_revision="model-v1",
                         plugin_revision="plugin-v1", backend="kunlun", metadata={
        "evidence_mode": "simulation", "worker_protocol": "managed-v2",
        "environment_required": False, "artifact_root": str(tmp_path / "run"),
    })
    spec = make_spec(tmp_path / "reference")
    scheduler.discover_operator("r", spec)
    yield scheduler, spec
    scheduler.store.close()


def execute_validation(scheduler, task, candidate, evidence, identifier="v1"):
    return validate_worker(scheduler, task.task_id, "controller", task.lease_token,
                           "independent-validator", candidate["candidate_id"], identifier,
                           evidence, timeout=20)


def test_actual_three_stage_measurements_and_readonly_replay(workflow):
    scheduler, _ = workflow
    for stage in ("torch", "xpu", "integration"):
        task = scheduler.claim_ready("controller", stage=stage, run_id="r")[0]
        candidate, evidence = prepare_candidate(scheduler, task)
        validation = execute_validation(scheduler, task, candidate, evidence, f"v-{stage}")
        assert validation["status"] == "PASS", validation
        assert validation["receipt"]["state"] == "succeeded"
        result = validation["result"]
        assert len(result["execution_ids"]) == 3
        raw = json.loads(Path(result["evidence"]["measurement_candidate"]).read_text())
        assert raw["cases"][0]["outputs"]["y"]["values"] == [-2.0, 0.0, 2.0, 4.0]
        if stage == "integration":
            assert raw["cases"][0]["service"]["http_status"] == 200
            assert raw["cases"][0]["service"]["transport"] == "http-loopback-simulation"
        done = scheduler.complete(task.task_id, worker_id="controller",
                                  lease_token=task.lease_token, result=result)
        assert done.status == "succeeded", done.output
        before = scheduler.store.run("r").to_dict()
        replay = execute_validation(scheduler, task, candidate, evidence, f"v-{stage}")
        assert replay["replayed"] is True
        assert replay["result"] == result
        assert scheduler.store.run("r").to_dict() == before


def test_wrong_numerics_are_measured_not_promoted(workflow):
    scheduler, _ = workflow
    task = scheduler.claim_ready("controller", stage="torch", run_id="r")[0]
    candidate, evidence = prepare_candidate(scheduler, task, CANDIDATE.replace("2.0", "3.0"))
    validation = execute_validation(scheduler, task, candidate, evidence)
    assert validation["status"] == "FAIL", validation
    assert validation["blocked"] is True
    assert validation["receipt"]["state"] == "failed"
    report = json.loads(Path(validation["result"]["evidence"]["independent_validation"]).read_text())
    assert any(check["name"].endswith(":numerical") and not check["passed"] for check in report["checks"])
    assert scheduler.store.get_task(task.task_id).status == "running"


def test_pass_shaped_module_output_is_not_a_measurement(workflow):
    scheduler, _ = workflow
    task = scheduler.claim_ready("controller", stage="torch", run_id="r")[0]
    candidate, evidence = prepare_candidate(scheduler, task, "def run_case(inputs):\n    return {'verdict': 'PASS'}\n")
    validation = execute_validation(scheduler, task, candidate, evidence)
    assert validation["status"] == "BLOCKED"
    assert validation["receipt"]["state"] == "failed"
    assert validation["result"] is None


def test_candidate_that_mutates_its_frozen_source_is_rejected(workflow):
    scheduler, _ = workflow
    task = scheduler.claim_ready("controller", stage="torch", run_id="r")[0]
    source = "from pathlib import Path\n" + CANDIDATE.replace(
        "    return", "    Path(__file__).write_text('changed during validation')\n    return")
    candidate, evidence = prepare_candidate(scheduler, task, source)
    validation = execute_validation(scheduler, task, candidate, evidence)
    assert validation["status"] == "BLOCKED"
    assert "candidate tree changed" in validation["error"]


@pytest.mark.parametrize("mutation", [
    lambda contract: contract.pop("thresholds"),
    lambda contract: contract.update(cases=[]),
    lambda contract: contract["thresholds"]["y"].update(max_relative_l2=True),
    lambda contract: contract["thresholds"]["y"].update(max_relative_l2=float("nan")),
    lambda contract: contract["negative_control"].update(value=0),
])
def test_missing_or_ambiguous_contract_is_rejected(tmp_path, mutation):
    spec = make_spec(tmp_path)
    mutation(spec.semantics["validation"])
    with pytest.raises(ValidationContractError):
        validation_plan(spec, "torch")


@pytest.mark.parametrize("stage", ["xpu", "integration"])
def test_real_device_service_has_no_simulation_shortcut(tmp_path, stage):
    plan = validation_plan(make_spec(tmp_path), stage)
    with pytest.raises(ValidationContractError, match="trusted real model-dispatch/service"):
        check_supported_mode(plan, "real")


def observations(tmp_path, stage="integration"):
    spec = make_spec(tmp_path)
    path = tmp_path / "candidate.py"
    path.write_text(CANDIDATE)
    plan = validation_plan(spec, stage)
    base = {"validation_id": "unit", "evidence_mode": "simulation", "plan": plan}
    result = {}
    for role, entry in (("candidate", path), ("reference", Path(plan["contract"]["reference"]["entry"]))):
        result[role] = measure({**base, "role": role, "entry": {
            "path": str(entry), "sha256": hashlib.sha256(entry.read_bytes()).hexdigest()}})
    reference = tmp_path / "reference-observations.json"
    reference.write_text(json.dumps(result["reference"]))
    result["control"] = measure({**base, "role": "control", "reference_observations": str(reference)})
    return plan, result


@pytest.mark.parametrize("field", ["dispatch", "fallback", "service"])
def test_missing_raw_execution_observation_is_rejected(tmp_path, field):
    plan, raw = observations(tmp_path)
    raw["candidate"]["cases"][0].pop(field)
    with pytest.raises(ValidationContractError):
        grade_observations(plan, **raw, evidence_mode="simulation", validation_id="unit")


def test_control_that_cannot_discriminate_is_not_pass(tmp_path):
    plan, raw = observations(tmp_path, "torch")
    raw["control"]["cases"] = deepcopy(raw["reference"]["cases"])
    result = grade_observations(plan, **raw, evidence_mode="simulation", validation_id="unit")
    assert result["verdict"] == "FAIL"


@pytest.mark.parametrize("field,value", [("dtype", "float32"), ("shape", [2, 2]),
                                          ("layout", "noncontiguous"), ("device", "fake-xpu")])
def test_tensor_metadata_must_match_the_observed_contract(tmp_path, field, value):
    plan, raw = observations(tmp_path, "torch")
    raw["candidate"]["cases"][0]["outputs"]["y"][field] = value
    with pytest.raises(ValidationContractError):
        grade_observations(plan, **raw, evidence_mode="simulation", validation_id="unit")
