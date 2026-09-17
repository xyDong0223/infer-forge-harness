"""Local fixtures exercise durable gates, not device or model correctness."""

import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from engine import IOSpec, OperatorSpec, TaskScheduler
from tests.scheduler_helpers import stage_result


def spec():
    return OperatorSpec(
        operator_id="identity", model_id="model", model_revision="1",
        plugin_revision="1", backend="device",
        inputs=[IOSpec("x", "float32", [1], "contiguous")],
        outputs=[IOSpec("y", "float32", [1], "contiguous")],
        semantics={"operation": "identity"},
    )


def claimed(tmp_path, *, managed=False):
    scheduler = TaskScheduler(tmp_path / "state.db")
    scheduler.create_run(
        run_id="run", model_id="model", backend="device",
        metadata={"evidence_mode": "simulation", **(
            {"artifact_root": str(tmp_path / "run")} if managed else {}
        )},
    )
    scheduler.discover_operator("run", spec())
    return scheduler, scheduler.claim_ready("worker")[0]


def test_managed_claims_isolate_retry_and_register_result(tmp_path):
    scheduler, first = claimed(tmp_path, managed=True)
    first_output = Path(first.input["workspace"]["output"])
    (first_output / "interrupted.txt").write_text("preserve this attempt")
    scheduler.recover(force=True)
    second = scheduler.claim_ready("worker")[0]
    assert second.input["workspace"]["root"] != first.input["workspace"]["root"]
    assert (first_output / "interrupted.txt").read_text() == "preserve this attempt"
    workspace = second.input["workspace"]
    assert json.loads((Path(workspace["input"]) / "task.json").read_text())["attempt"] == 2
    result = stage_result(tmp_path / "unused", second)
    outcome = scheduler.complete(second.task_id, "worker", result, second.lease_token)
    assert outcome.status == "succeeded"
    manifest = json.loads(Path(outcome.output["artifact_manifest"]).read_text())
    assert manifest["identity"]["attempt"] == 2
    assert manifest["outcome"] == "PASS"
    assert "result.json" in {item["path"] for item in manifest["artifacts"]}


def test_managed_evidence_cannot_reference_unowned_output(tmp_path):
    scheduler, task = claimed(tmp_path, managed=True)
    result = stage_result(tmp_path / "unused", task)
    source = Path(result["evidence"]["reference_artifact"])
    outside = tmp_path / "unowned.json"
    outside.write_bytes(source.read_bytes())
    result["evidence"]["reference_artifact"] = str(outside)
    outcome = scheduler.complete(task.task_id, "worker", result, task.lease_token)
    assert outcome.status == "failed"
    assert any("claimed attempt output" in error for error in
               outcome.output["error"]["metadata"]["validation_errors"])
    assert json.loads(Path(outcome.output["artifact_manifest"]).read_text())["outcome"] == "FAILED"


def test_inventory_error_is_recorded_as_diagnosis_not_success(tmp_path):
    scheduler, task = claimed(tmp_path, managed=True)
    result = stage_result(tmp_path / "unused", task)
    output = Path(task.input["workspace"]["output"])
    (output / "linked.json").symlink_to(result["evidence"]["reference_artifact"])
    outcome = scheduler.complete(task.task_id, "worker", result, task.lease_token)
    assert outcome.status == "failed"
    assert outcome.output["error"]["metadata"]["reason"] == "artifact_registration_failed"
    assert "artifact_registration_error" in outcome.output["error"]["metadata"]
    assert scheduler.diagnosis_for(task.task_id) is not None


@pytest.mark.parametrize("result", [{}, {"status": "PASS"}, {"status": "UNKNOWN"}, {"ok": True}])
def test_success_shaped_results_create_diagnosis_instead_of_advancing(tmp_path, result):
    scheduler, task = claimed(tmp_path)
    outcome = scheduler.complete(task.task_id, "worker", result, task.lease_token)
    assert outcome.status == "failed"
    assert not scheduler.pending_tasks("run", "xpu")
    assert scheduler.diagnosis_for(task.task_id) is not None
    assert outcome.output["error"]["metadata"]["validation_errors"]


@pytest.mark.parametrize("damage", [
    "missing_file", "modified_file", "wrong_task", "wrong_attempt", "same_validator",
    "empty_checks", "missing_stage_evidence", "conflicting_verdict", "simulation_as_real",
])
def test_evidence_is_bound_to_this_candidate_attempt_and_validator(tmp_path, damage):
    scheduler, task = claimed(tmp_path)
    result = stage_result(tmp_path / "evidence", task)
    report_path = Path(result["evidence"]["independent_validation"])
    report = json.loads(report_path.read_text())
    if damage == "missing_file":
        Path(result["evidence"]["reference_artifact"]).unlink()
    elif damage == "modified_file":
        Path(result["evidence"]["reference_artifact"]).write_text("changed")
    elif damage == "wrong_task":
        report["task_id"] = "another-task"
    elif damage == "wrong_attempt":
        report["attempt"] += 1
    elif damage == "same_validator":
        report["validator"] = "worker"
    elif damage == "empty_checks":
        report["checks"] = []
    elif damage == "missing_stage_evidence":
        del result["evidence"]["focused_tests"]
    elif damage == "conflicting_verdict":
        result.update(status="PASS", verdict="FAIL")
    elif damage == "simulation_as_real":
        result["evidence_mode"] = "real"
    report_path.write_text(json.dumps(report))
    result["evidence_sha256"]["independent_validation"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    outcome = scheduler.complete(task.task_id, "worker", result, task.lease_token)
    assert outcome.status == "failed"
    assert not scheduler.pending_tasks("run", "xpu")


def test_completion_rollback_leaves_no_half_transition(tmp_path, monkeypatch):
    scheduler, task = claimed(tmp_path)
    result = stage_result(tmp_path / "evidence", task)
    events = scheduler.store.events("run")

    def crash(*args):
        raise RuntimeError("crash before successor")

    monkeypatch.setattr(scheduler, "_ensure_next", crash)
    with pytest.raises(RuntimeError, match="crash"):
        scheduler.complete(task.task_id, "worker", result, task.lease_token)
    reopened = TaskScheduler(tmp_path / "state.db")
    assert reopened.task_status(task.task_id).status == "running"
    assert len(reopened.store.events("run")) == len(events)
    assert not reopened.pending_tasks("run", "xpu")


def test_failure_rollback_leaves_no_missing_diagnosis(tmp_path, monkeypatch):
    scheduler, task = claimed(tmp_path)
    events = scheduler.store.events("run")

    def crash(*args):
        raise RuntimeError("crash before diagnosis")

    monkeypatch.setattr(scheduler, "_ensure_diagnosis", crash)
    with pytest.raises(RuntimeError, match="crash"):
        scheduler.fail(task.task_id, "worker", "failure", task.lease_token)
    assert scheduler.task_status(task.task_id).status == "running"
    assert scheduler.diagnosis_for(task.task_id) is None
    assert len(scheduler.store.events("run")) == len(events)


def test_event_failure_rolls_back_task_and_successor(tmp_path, monkeypatch):
    scheduler, task = claimed(tmp_path)
    result = stage_result(tmp_path / "evidence", task)
    original = scheduler.store.append_event

    def crash(event):
        if event.event_type == "task_created":
            raise RuntimeError("event write interrupted")
        return original(event)

    monkeypatch.setattr(scheduler.store, "append_event", crash)
    with pytest.raises(RuntimeError, match="interrupted"):
        scheduler.complete(task.task_id, "worker", result, task.lease_token)
    assert scheduler.task_status(task.task_id).status == "running"
    assert not scheduler.pending_tasks("run", "xpu")
    assert scheduler.store.events("run")[-1].event_type == "task_claimed"


def test_reconcile_repairs_missing_successor_once_and_requires_evidence(tmp_path, monkeypatch):
    scheduler, task = claimed(tmp_path)
    result = stage_result(tmp_path / "evidence", task)
    with monkeypatch.context() as patch:
        patch.setattr(scheduler, "_ensure_next", lambda *args: None)
        scheduler.complete(task.task_id, "worker", result, task.lease_token)
    evidence = Path(result["evidence"]["reference_artifact"])
    original = evidence.read_bytes()
    evidence.write_text("changed after validation")
    blocked = scheduler.reconcile("run")
    assert blocked["blocked"] and not blocked["created"]
    evidence.write_bytes(original)
    repaired = scheduler.reconcile("run")
    assert len(repaired["created"]) == 1
    assert not repaired["blocked"]
    assert scheduler.reconcile("run") == {"created": [], "blocked": []}


def test_reconcile_repairs_missing_diagnosis(tmp_path, monkeypatch):
    scheduler, task = claimed(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(scheduler, "_ensure_diagnosis", lambda *args: None)
        scheduler.fail(task.task_id, "worker", "historical failure", task.lease_token)
    repaired = scheduler.reconcile("run")
    assert repaired["created"] == [task.task_id + ":diagnosis"]
    assert scheduler.diagnosis_for(task.task_id).input["bug_report"]["message"] == "historical failure"


def test_reconcile_cannot_forget_producer_after_lease_is_cleared(tmp_path, monkeypatch):
    scheduler, task = claimed(tmp_path)
    result = stage_result(tmp_path / "evidence", task)
    report_path = Path(result["evidence"]["independent_validation"])
    report = json.loads(report_path.read_text())
    report["validator"] = "worker"
    report_path.write_text(json.dumps(report))
    result["evidence_sha256"]["independent_validation"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    with monkeypatch.context() as patch:
        # Reproduce the old permissive gate and interrupted successor creation.
        patch.setattr("engine.scheduler.validate_result", lambda *args: [])
        patch.setattr(scheduler, "_ensure_next", lambda *args: None)
        scheduler.complete(task.task_id, "worker", result, task.lease_token)
    repaired = scheduler.reconcile("run")
    assert not repaired["created"]
    assert any("distinct from the producer" in error for error in repaired["blocked"][0]["validation_errors"])


def test_expired_and_superseded_leases_cannot_submit_with_reused_worker_name(tmp_path, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("engine.scheduler.time.time", lambda: now[0])
    scheduler, task = claimed(tmp_path)
    result = stage_result(tmp_path / "evidence", task)
    with pytest.raises(ValueError, match="lease token"):
        scheduler.complete(task.task_id, worker_id="worker", result=result)
    now[0] += 301
    with pytest.raises(ValueError, match="unexpired"):
        scheduler.complete(task.task_id, "worker", result, task.lease_token)
    reclaimed = scheduler.claim_ready("worker")[0]
    assert reclaimed.lease_token != task.lease_token
    with pytest.raises(ValueError, match="lease token"):
        scheduler.fail(task.task_id, "worker", "late failure", task.lease_token)
    assert scheduler.task_status(task.task_id).status == "running"


def test_renewal_preserves_attempt_and_extends_lease(tmp_path, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("engine.scheduler.time.time", lambda: now[0])
    scheduler, task = claimed(tmp_path)
    now[0] += 250
    renewed = scheduler.renew_lease(task.task_id, "worker", task.lease_token, 300)
    assert renewed.lease_token == task.lease_token
    assert renewed.attempt == task.attempt
    assert renewed.lease_expires == 1550
    now[0] += 100
    result = stage_result(tmp_path / "evidence", task)
    assert scheduler.complete(task.task_id, "worker", result, task.lease_token).status == "succeeded"


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_invalid_lease_durations_are_explicit_errors(tmp_path, seconds):
    scheduler, task = claimed(tmp_path)
    with pytest.raises(ValueError):
        scheduler.renew_lease(task.task_id, "worker", task.lease_token, seconds)
    with pytest.raises(ValueError):
        scheduler.claim_ready("worker", lease_seconds=seconds)


def test_independent_connections_claim_once(tmp_path):
    scheduler, task = claimed(tmp_path)
    scheduler.recover(force=True)

    def claim(worker):
        other = TaskScheduler(tmp_path / "state.db")
        try:
            return other.claim_ready(worker)
        finally:
            other.store.close()

    with ThreadPoolExecutor(max_workers=2) as workers:
        batches = list(workers.map(claim, ["one", "two"]))
    assert [entry.task_id for batch in batches for entry in batch] == [task.task_id]


def environment_proof(root):
    root.mkdir(parents=True, exist_ok=True)
    artifacts = [
        "environment_fingerprint.txt", "runtime_import.txt", "code_readiness.json",
        "device_readiness.json", "base_model_identity.json", "base_health_result.txt",
        "base_chat_result.json", "base_server_log.txt",
    ]
    for name in artifacts:
        (root / name).write_text("local test fixture: " + name)
    from tests.scheduler_helpers import write_base_model_identity
    write_base_model_identity(root)
    return {
        "state": "ENVIRONMENT_READY", "pod": "test-pod", "artifact_root": str(root),
        "user_id": "fixture-owner",
        "checks": {
            "pod_ready": True, "runtime_importable": True, "code_ready": True,
            "device_ready": True, "base_model_loaded": True, "base_prefill": True,
            "base_decode": True, "base_health_check": 200,
            "base_chat_completion": "non_empty", "unexpected_fallback": False,
        },
        "artifacts": artifacts,
    }


def test_real_runs_cannot_skip_environment_via_python_api(tmp_path):
    scheduler = TaskScheduler(tmp_path / "state.db")
    run = scheduler.create_run(run_id="run", model_id="model")
    assert run.status == "WAITING_FOR_ENVIRONMENT"
    with pytest.raises(ValueError, match="environment proof"):
        scheduler.discover_operator("run", spec())


@pytest.mark.parametrize("damage", ["state", "artifact", "validator", "fingerprint"])
def test_environment_import_checks_actual_status_and_readable_evidence(tmp_path, damage):
    scheduler = TaskScheduler(tmp_path / "state.db")
    scheduler.create_run(run_id="run", model_id="model")
    proof = environment_proof(tmp_path / "proof")
    if damage == "state":
        proof["state"] = "ENVIRONMENT_FAILED"
    elif damage == "validator":
        proof["validator"] = {"passed": False}
    elif damage == "fingerprint":
        (tmp_path / "proof" / "environment_fingerprint.txt").write_text("")
    else:
        (tmp_path / "proof" / "device_readiness.json").unlink()
    with pytest.raises(ValueError):
        scheduler.bind_environment("run", proof)
    assert scheduler.store.run("run").status == "WAITING_FOR_ENVIRONMENT"


def test_environment_binding_requires_recorded_user_id(tmp_path):
    scheduler = TaskScheduler(tmp_path / "state.db")
    scheduler.create_run(run_id="run", model_id="model", backend="device")
    proof = environment_proof(tmp_path / "proof")
    proof.pop("user_id")
    with pytest.raises(ValueError, match="user_id"):
        scheduler.bind_environment("run", proof)


def test_environment_binding_fingerprints_artifacts_and_rejects_cross_environment_spec(tmp_path):
    scheduler = TaskScheduler(tmp_path / "state.db")
    scheduler.create_run(run_id="run", model_id="model", backend="device")
    proof = environment_proof(tmp_path / "proof")
    run = scheduler.bind_environment("run", proof)
    bound = run.environment["environment_proof"]
    assert bound["fingerprint"] == hashlib.sha256(
        (tmp_path / "proof" / "environment_fingerprint.txt").read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="fingerprint"):
        scheduler.discover_operator("run", spec())
    matching = spec()
    matching.environment = bound
    scheduler.discover_operator("run", matching)
    (tmp_path / "proof" / "environment_fingerprint.txt").write_text("changed environment")
    with pytest.raises(ValueError, match="environment changed"):
        scheduler.bind_environment("run", proof)
    assert scheduler.store.run("run").environment["environment_proof"] == bound
    scheduler.record_environment_failure("run", {"state": "FAILED"}, "failed reproof")
    with pytest.raises(ValueError, match="environment changed"):
        scheduler.bind_environment("run", proof)
    assert scheduler.store.run("run").environment["environment_proof"] == bound


def test_legacy_cli_discovery_uses_bound_environment(tmp_path):
    scheduler = TaskScheduler(tmp_path / "state.db")
    scheduler.create_run(run_id="run", model_id="model", backend="device")
    run = scheduler.bind_environment("run", environment_proof(tmp_path / "proof"))
    report = tmp_path / "gaps.json"
    report.write_text(json.dumps({
        "plugin_revision": "1",
        "entries": [{
            "operator_id": "op", "inputs": [spec().inputs[0].to_dict()],
            "outputs": [spec().outputs[0].to_dict()], "semantics": {"description": "identity"},
        }],
    }))
    script = Path(__file__).resolve().parents[2] / "cli" / "scheduler.py"
    completed = subprocess.run([
        sys.executable, str(script), "--state", str(tmp_path / "state.db"),
        "discover", "--run-id", "run", "--model", "model", "--backend", "device",
        "--report", str(report),
    ], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    discovered = json.loads(completed.stdout)["tasks"][0]
    assert discovered["input"]["operator_spec"]["environment"]["fingerprint"] == run.environment["environment_proof"]["fingerprint"]


def test_failed_reproof_pauses_real_work_until_same_environment_is_ready(tmp_path):
    scheduler = TaskScheduler(tmp_path / "state.db")
    scheduler.create_run(run_id="run", model_id="model", backend="device")
    proof = environment_proof(tmp_path / "proof")
    run = scheduler.bind_environment("run", proof)
    matching = spec()
    matching.environment = run.environment["environment_proof"]
    task = scheduler.discover_operator("run", matching)
    scheduler.record_environment_failure("run", {"state": "FAILED"}, "failed reproof")
    assert not scheduler.claim_ready("worker", stage="torch")
    scheduler.bind_environment("run", proof)
    assert scheduler.claim_ready("worker", stage="torch")[0].task_id == task.task_id
