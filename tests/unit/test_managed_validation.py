"""Managed protocol trust-boundary tests; all evidence is labeled simulation."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import pytest

from engine import IOSpec, OperatorSpec, TaskScheduler
from engine.context import task_packet
from engine.execution import begin_execution, finish_execution
from engine.graph_bridge import GraphSchedulerBridge
from engine.interaction import canonical_digest
from engine.managed_validation import (
    RECIPE_FILES, RECIPE_PATH, begin_validation, finish_validation, freeze_candidate,
    get_candidate, managed_result_binding, reconcile_managed_validation, validate_candidate,
)
from engine.result_validation import validate_result
from tests.scheduler_helpers import simulation_result
from tests.unit.test_scheduler_reliability import environment_proof, spec as legacy_spec


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def managed(tmp_path, request):
    reference = tmp_path / "reference.py"
    reference.write_text("def run_case(inputs):\n    return {**inputs}\n" if
                         getattr(request, "param", None) == "duplicate_reference" else
                         "def run_case(inputs):\n    return dict(inputs)\n")
    contract = {
        "schema_version": 1, "candidate_entry": "entry.py",
        "reference": {"entry": str(reference), "files": {str(reference): digest(reference)},
                      "provenance": "Independent simulation fixture, not device evidence"},
    }
    spec = OperatorSpec(
        operator_id="identity", model_id="model", model_revision="1", plugin_revision="1",
        backend="device", inputs=[IOSpec("x", "float64", [1], "contiguous")],
        outputs=[IOSpec("y", "float64", [1], "contiguous")],
        semantics={"operation": "identity", "validation": contract},
    )
    scheduler = TaskScheduler(tmp_path / "state.db")
    scheduler.create_run(run_id="run", model_id="model", backend="device", metadata={
        "evidence_mode": "simulation", "worker_protocol": "managed-v2",
        "artifact_root": str(tmp_path / "run"),
    })
    scheduler.discover_operator("run", spec)
    task = scheduler.claim_ready("controller", lease_seconds=300)[0]
    root = Path(task.input["workspace"]["output"]) / "candidate"
    root.mkdir()
    (root / "entry.py").write_text("def run_case(inputs):\n    return {**inputs}\n")
    yield scheduler, task, root, contract
    scheduler.store.close()


def freeze(managed):
    scheduler, task, root, _ = managed
    return freeze_candidate(scheduler, task.task_id, "controller", task.lease_token,
                            "producer", root, "revision-1")


def request(managed, *, validation_id="validation-1"):
    scheduler, task, _, contract = managed
    candidate = freeze(managed)
    return dict(
        task_id=task.task_id, worker="controller", lease_token=task.lease_token,
        validation_id=validation_id, candidate_id=candidate["candidate_id"],
        validator="producer:simulation-validator",
        recipe={"path": str(RECIPE_PATH), "sha256": digest(RECIPE_PATH),
                "files": {name: digest(name) for name in RECIPE_FILES},
                "controller_process": {"host_id": "fixture-host", "pid": 111,
                                       "process_identity": "simulation-controller-birth"},
                "candidate_entry": "entry.py", "reference_entry": contract["reference"]["entry"]},
        contract_sha256=canonical_digest(contract), reference_files=contract["reference"]["files"],
    )


def validated(managed, *, last_active=False):
    scheduler, task, root, _ = managed
    receipt, _ = begin_validation(scheduler, **request(managed))
    result = simulation_result(task, root.parent, worker_id="producer", run=scheduler.store.run("run"))
    result.update(managed_validation_id=receipt["validation_id"], execution_ids=receipt["execution_ids"],
                  candidate_id=receipt["candidate_id"])
    for identity in receipt["execution_ids"]:
        begin_execution(scheduler, "run", identity, {
            "argv": ["unit-test-simulation", identity], "cwd": str(root),
            "output_dir": str(root.parent),
        }, task_id=task.task_id, worker="controller", lease_token=task.lease_token)
        if not last_active or identity != receipt["execution_ids"][-1]:
            finish_execution(scheduler, "run", identity, state="SUCCEEDED", returncode=0,
                             termination_confirmed=True, worker="controller", lease_token=task.lease_token)
    receipt = finish_validation(scheduler, "run", receipt["validation_id"], result)
    return receipt, result


def test_frozen_candidate_is_complete_idempotent_and_current(managed):
    scheduler, task, root, _ = managed
    (root / ".hidden.py").write_text("hidden files are still inventoried")
    record = freeze(managed)
    assert record == freeze(managed)
    assert {item["path"] for item in record["files"]} == {"entry.py", ".hidden.py"}
    assert record == get_candidate(scheduler, task.task_id)
    packet = task_packet(scheduler.store, scheduler.store.run("run"), task)
    assert packet["acceptance"]["managed_validation_required"] is True
    assert packet["managed_candidates"] == [record]
    assert not validate_candidate(task, scheduler.store.run("run"), record)


@pytest.mark.parametrize("damage", ["additional", "changed", "removed", "manifest", "symlink"])
def test_frozen_candidate_rejects_tree_and_manifest_mutation(managed, damage):
    scheduler, task, root, _ = managed
    record = freeze(managed)
    if damage == "additional":
        (root / ".undeclared").write_text("not present at freeze")
    elif damage == "changed":
        (root / "entry.py").write_text("changed")
    elif damage == "removed":
        (root / "entry.py").unlink()
    elif damage == "manifest":
        Path(record["manifest_path"]).write_text("{}")
    else:
        (root / "link.py").symlink_to(root / "entry.py")
    assert validate_candidate(task, scheduler.store.run("run"), record)
    with pytest.raises(ValueError):
        freeze(managed)
    with pytest.raises(ValueError):
        get_candidate(scheduler, task.task_id, record["candidate_id"])


def test_candidate_requires_exact_current_attempt_and_live_claim(managed):
    scheduler, task, root, _ = managed
    with pytest.raises(ValueError, match="output/candidate"):
        freeze_candidate(scheduler, task.task_id, "controller", task.lease_token,
                         "producer", root.parent, "revision-1")
    with pytest.raises(ValueError):
        freeze_candidate(scheduler, task.task_id, "controller", "wrong-token", "producer", root, "r")
    record = freeze(managed)
    assert scheduler.recover(force=True) == 1
    newer = scheduler.claim_ready("controller")[0]
    assert validate_candidate(newer, scheduler.store.run("run"), record)


@pytest.mark.parametrize("damage", [
    "same_producer", "contract", "candidate_entry", "reference_entry", "reference_hash",
    "recipe_hash", "recipe_files", "recipe_file_hash", "duplicate_reference",
])
def test_validation_acceptance_rejects_unbound_or_nonindependent_recipe(managed, damage):
    scheduler, _, root, contract = managed
    args = request(managed)
    if damage == "same_producer":
        args["validator"] = "producer"
    elif damage == "contract":
        args["contract_sha256"] = "0" * 64
    elif damage == "candidate_entry":
        args["recipe"]["candidate_entry"] = "other.py"
    elif damage == "reference_entry":
        args["recipe"]["reference_entry"] = str(root / "entry.py")
    elif damage == "reference_hash":
        Path(contract["reference"]["entry"]).write_text("changed reference")
    elif damage == "recipe_hash":
        args["recipe"]["sha256"] = "0" * 64
    elif damage == "recipe_files":
        args["recipe"]["files"].pop(str(RECIPE_PATH))
    elif damage == "recipe_file_hash":
        args["recipe"]["files"][str(RECIPE_PATH)] = "0" * 64
    else:
        # Entry equivalence is checked even if both files have different paths.
        Path(contract["reference"]["entry"]).write_bytes((root / "entry.py").read_bytes())
    with pytest.raises(ValueError):
        begin_validation(scheduler, **args)
    assert not scheduler.store.run("run").metadata.get("managed_validations")


@pytest.mark.parametrize("managed", ["duplicate_reference"], indirect=True)
def test_reference_cannot_be_same_candidate_content_at_another_path(managed):
    scheduler, _, _, _ = managed
    with pytest.raises(ValueError, match="duplicate the candidate"):
        begin_validation(scheduler, **request(managed))


def test_acceptance_is_durable_idempotent_and_never_reclaims_uncertain_execution(managed, monkeypatch):
    scheduler, task, _, _ = managed
    args = request(managed)
    receipt, is_new = begin_validation(scheduler, **args)
    assert is_new and receipt["state"] == "executing"
    assert receipt["execution_status"] == "IN_PROGRESS_OR_UNKNOWN"
    assert task.lease_token not in json.dumps(receipt)
    reopened = TaskScheduler(scheduler.store.path)
    assert begin_validation(reopened, **args) == (receipt, False)
    assert reopened.recover(force=True) == 0
    monkeypatch.setattr("engine.scheduler.time.time", lambda: task.lease_expires + 1)
    assert reopened.claim_ready("second-controller") == []
    assert begin_validation(reopened, **args) == (receipt, False)
    with pytest.raises(ValueError):
        begin_validation(reopened, **{**args, "validator": "another-validator"})
    reopened.store.close()


def test_acceptance_has_one_winner_across_scheduler_connections(managed):
    scheduler, _, _, _ = managed
    args = request(managed)

    def accept(_):
        other = TaskScheduler(scheduler.store.path)
        try:
            return begin_validation(other, **args)[1]
        finally:
            other.store.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(accept, range(2))) == [False, True]


def test_active_validation_blocks_complete_fail_and_other_validation(managed):
    scheduler, task, _, _ = managed
    args = request(managed)
    begin_validation(scheduler, **args)
    for action in (
        lambda: scheduler.complete(task.task_id, "controller", {}, task.lease_token),
        lambda: scheduler.fail(task.task_id, "controller", "failure", task.lease_token),
        lambda: begin_validation(scheduler, **{**args, "validation_id": "another-id"}),
    ):
        with pytest.raises(ValueError, match="active|uncertain"):
            action()


def test_legacy_report_does_not_satisfy_managed_protocol(managed):
    scheduler, task, root, _ = managed
    result = simulation_result(task, root.parent)
    done = scheduler.complete(task.task_id, "controller", result, task.lease_token)
    assert done.status == "failed"
    assert "managed_validation_id" in str(done.output)
    assert not scheduler.pending_tasks("run", "xpu")
    diagnosis = scheduler.claim_ready("diagnostician", stage="diagnosis")[0]
    result = simulation_result(diagnosis, root.parent, worker_id="diagnostician")
    assert scheduler.complete(diagnosis.task_id, "diagnostician", result, diagnosis.lease_token).status == "succeeded"


def test_finished_receipt_is_replayable_and_binds_final_submission(managed):
    scheduler, task, _, _ = managed
    receipt, result = validated(managed)
    assert receipt["result"] == result
    assert finish_validation(scheduler, "run", receipt["validation_id"], result) == receipt
    assert not validate_result(task, scheduler.store.run("run"), result, "controller")
    done = scheduler.complete(task.task_id, "controller", result, task.lease_token)
    assert done.status == "succeeded"
    assert done.output["_submission"]["candidate_id"] == receipt["candidate_id"]
    assert done.output["_submission"]["producer"] == "producer"
    assert len(done.output["_submission"]["execution_ids"]) == 3
    assert scheduler.pending_tasks("run", "xpu")
    assert not validate_result(done, scheduler.store.run("run"), done.output, "controller")
    assert finish_validation(scheduler, "run", receipt["validation_id"], result) == receipt
    with pytest.raises(ValueError, match="conflicts"):
        finish_validation(scheduler, "run", receipt["validation_id"], None, status="FAIL")


def test_passing_measurements_do_not_release_an_unterminated_process(managed):
    scheduler, task, _, _ = managed
    receipt, result = validated(managed, last_active=True)
    assert receipt["state"] == "succeeded"
    errors = validate_result(task, scheduler.store.run("run"), result, "controller")
    assert any("confirmed successful termination" in error for error in errors)
    with pytest.raises(ValueError, match="active|uncertain"):
        scheduler.complete(task.task_id, "controller", result, task.lease_token)
    finish_execution(scheduler, "run", receipt["execution_ids"][-1], state="SUCCEEDED",
                     returncode=0, termination_confirmed=True,
                     worker="controller", lease_token=task.lease_token)
    assert scheduler.complete(task.task_id, "controller", result, task.lease_token).status == "succeeded"


def test_reconcile_and_delivery_revalidate_frozen_candidate(managed, monkeypatch):
    scheduler, task, root, _ = managed
    receipt, result = validated(managed)
    with monkeypatch.context() as interrupted:
        interrupted.setattr(scheduler, "_ensure_next", lambda *args: None)
        assert scheduler.complete(task.task_id, "controller", result, task.lease_token).status == "succeeded"
    (root / "unlisted.py").write_text("changed after completion")
    reconciliation = scheduler.reconcile("run")
    assert not reconciliation["created"]
    assert "frozen candidate tree changed" in str(reconciliation["blocked"])
    bridge = GraphSchedulerBridge(Path(scheduler.store.path), "run", "model",
                                  Path(scheduler.store.run("run").metadata["artifact_root"]),
                                  {}, execute=False)
    try:
        delivery = bridge.delivery_status()
    finally:
        bridge.close()
    assert delivery["state"] == "OPERATORS_BLOCKED"
    assert "frozen candidate tree changed" in str(delivery["errors"])
    binding = delivery["task_results"][0]["managed_validation"]
    assert binding["candidate_id"] == receipt["candidate_id"]
    assert binding["managed_validation_sha256"] == receipt["receipt_sha256"]
    assert binding["evidence_mode"] == "simulation"


def test_downstream_claim_cannot_use_a_changed_frozen_predecessor(managed):
    scheduler, task, root, _ = managed
    _, result = validated(managed)
    assert scheduler.complete(task.task_id, "controller", result, task.lease_token).status == "succeeded"
    assert scheduler.pending_tasks("run", "xpu")
    (root / "entry.py").write_text("modified after completion")
    assert scheduler.claim_ready("xpu-worker", stage="xpu") == []


@pytest.mark.parametrize("damage", ["candidate", "reference", "result", "evidence", "id", "execution"])
def test_completed_receipt_revalidates_every_bound_input(managed, damage):
    scheduler, task, root, contract = managed
    receipt, result = validated(managed)
    if damage == "candidate":
        (root / "entry.py").write_text("changed candidate")
    elif damage == "reference":
        Path(contract["reference"]["entry"]).write_text("changed reference")
    elif damage == "result":
        result["unvalidated_field"] = True
    elif damage == "evidence":
        Path(result["evidence"]["focused_tests"]).write_text("changed evidence")
    elif damage == "id":
        result["managed_validation_id"] = "uploaded-untrusted-id"
    else:
        result["execution_ids"] = receipt["execution_ids"][:1]
    assert validate_result(task, scheduler.store.run("run"), result, "controller")
    assert scheduler.complete(task.task_id, "controller", result, task.lease_token).status == "failed"
    assert not scheduler.pending_tasks("run", "xpu")


def test_finish_rejects_self_asserted_pass_without_executor_records(managed):
    scheduler, task, root, _ = managed
    receipt, _ = begin_validation(scheduler, **request(managed))
    result = simulation_result(task, root.parent, worker_id="producer")
    result["execution_ids"] = receipt["execution_ids"]
    result["candidate_id"] = receipt["candidate_id"]
    with pytest.raises(ValueError, match="executor record"):
        finish_validation(scheduler, "run", receipt["validation_id"], result)
    assert scheduler.has_active_execution(task.task_id)


def test_failed_validation_releases_only_receipt_not_active_executor(managed):
    scheduler, task, root, _ = managed
    receipt, _ = begin_validation(scheduler, **request(managed))
    identity = receipt["execution_ids"][0]
    begin_execution(scheduler, "run", identity, {
        "argv": ["unit-test-simulation"], "cwd": str(root), "output_dir": str(root.parent),
    }, task_id=task.task_id, worker="controller", lease_token=task.lease_token)
    failed = finish_validation(scheduler, "run", receipt["validation_id"], None, status="FAIL", error="blocked")
    assert failed["state"] == "failed"
    assert scheduler.has_active_execution(task.task_id)
    with pytest.raises(ValueError, match="active|uncertain"):
        scheduler.fail(task.task_id, "controller", "blocked", task.lease_token)
    finish_execution(scheduler, "run", identity, state="FAILED", returncode=1,
                     termination_confirmed=True, worker="controller", lease_token=task.lease_token)
    assert scheduler.fail(task.task_id, "controller", "blocked", task.lease_token).status == "failed"


def test_validation_finish_requires_unexpired_original_lease(managed, monkeypatch):
    scheduler, task, _, _ = managed
    receipt, _ = begin_validation(scheduler, **request(managed))
    monkeypatch.setattr("engine.scheduler.time.time", lambda: task.lease_expires + 1)
    with pytest.raises(ValueError, match="expired"):
        finish_validation(scheduler, "run", receipt["validation_id"], None, status="FAIL")


def absent_controller(receipt):
    return {"controller_absent": True, "controller_process": receipt["recipe"]["controller_process"],
            "explanation": "unit-test trusted observer confirms original controller absent"}


@pytest.mark.parametrize("after_partial_execution", [False, True])
def test_reconcile_orphan_validation_allows_recovery_without_expired_lease_authority(managed, monkeypatch,
                                                                                 after_partial_execution):
    scheduler, task, root, _ = managed
    args = request(managed)
    receipt, _ = begin_validation(scheduler, **args)
    if after_partial_execution:
        identity = receipt["execution_ids"][0]
        begin_execution(scheduler, "run", identity, {
            "argv": ["unit-test-simulation"], "cwd": str(root), "output_dir": str(root.parent),
        }, task_id=task.task_id, worker="controller", lease_token=task.lease_token)
        finish_execution(scheduler, "run", identity, state="SUCCEEDED", returncode=0,
                         termination_confirmed=True, worker="controller", lease_token=task.lease_token)
    monkeypatch.setattr("engine.scheduler.time.time", lambda: task.lease_expires + 1)
    assert scheduler.recover() == 0
    failed = reconcile_managed_validation(scheduler, "run", receipt["validation_id"], observer=absent_controller)
    assert failed["state"] == "failed" and failed["result"] is None
    assert failed["status"] == "FAIL" and failed["receipt_sha256"]
    assert begin_validation(scheduler, **args) == (failed, False)
    assert reconcile_managed_validation(scheduler, "run", receipt["validation_id"],
                                        observer=lambda _: pytest.fail("terminal must not reobserve")) == failed
    assert scheduler.recover() == 1
    reclaimed = scheduler.claim_ready("new-controller")[0]
    assert reclaimed.attempt == task.attempt + 1


@pytest.mark.parametrize("observed", [
    {"controller_absent": False}, {"controller_absent": "yes"},
    {"controller_absent": True, "controller_process": {"pid": 222}},
])
def test_reconcile_never_treats_uncertain_or_different_controller_as_dead(managed, observed):
    scheduler, task, _, _ = managed
    receipt, _ = begin_validation(scheduler, **request(managed))
    assert reconcile_managed_validation(scheduler, "run", receipt["validation_id"],
                                        observer=lambda _: observed) == receipt
    assert scheduler.has_active_execution(task.task_id)
    assert scheduler.recover(force=True) == 0


def test_reconcile_cannot_release_an_unterminated_child(managed):
    scheduler, task, root, _ = managed
    receipt, _ = begin_validation(scheduler, **request(managed))
    begin_execution(scheduler, "run", receipt["execution_ids"][0], {
        "argv": ["unit-test-simulation"], "cwd": str(root), "output_dir": str(root.parent),
    }, task_id=task.task_id, worker="controller", lease_token=task.lease_token)
    assert reconcile_managed_validation(scheduler, "run", receipt["validation_id"],
                                        observer=absent_controller) == receipt
    assert scheduler.recover(force=True) == 0


def test_reconcile_rejects_receipt_changed_during_observation(managed):
    scheduler, _, _, _ = managed
    receipt, _ = begin_validation(scheduler, **request(managed))

    def competing_finish(original):
        finish_validation(scheduler, "run", receipt["validation_id"], None, status="FAIL", error="finished meanwhile")
        return absent_controller(original)

    with pytest.raises(ValueError, match="changed during"):
        reconcile_managed_validation(scheduler, "run", receipt["validation_id"], observer=competing_finish)


def test_environment_requires_stable_resource_uid_only_for_managed_protocol(tmp_path):
    scheduler = TaskScheduler(tmp_path / "state.db")
    proof = environment_proof(tmp_path / "proof")
    scheduler.create_run(run_id="managed", model_id="model", metadata={"worker_protocol": "managed-v2"})
    for identity in (None, {}, {"cluster": "a", "namespace": "b", "pod_uid": ""}):
        with pytest.raises(ValueError, match="resource|pod_uid"):
            scheduler.bind_environment("managed", {**proof, "resource_identity": identity})
    resource = {"cluster": "fixture", "namespace": "fixture", "pod_uid": "fixture-uid"}
    run = scheduler.bind_environment("managed", {**proof, "resource_identity": resource})
    assert run.environment["environment_proof"]["resource_identity"] == resource
    scheduler.create_run(run_id="legacy", model_id="model")
    assert scheduler.bind_environment("legacy", proof).status == "ENVIRONMENT_READY"
    assert managed_result_binding(scheduler.store.run("legacy"), {"managed_validation_id": []}) == {}
    scheduler.store.close()


def test_legacy_graph_snapshot_does_not_gain_empty_managed_fields(tmp_path):
    scheduler = TaskScheduler(tmp_path / "state.db")
    root = tmp_path / "run"
    scheduler.create_run(run_id="legacy", model_id="model", backend="device", metadata={
        "evidence_mode": "simulation", "artifact_root": str(root),
    })
    scheduler.discover_operator("legacy", legacy_spec())
    task = scheduler.claim_ready("legacy-worker")[0]
    result = simulation_result(task, root, worker_id="legacy-worker")
    assert scheduler.complete(task.task_id, "legacy-worker", result, task.lease_token).status == "succeeded"
    bridge = GraphSchedulerBridge(Path(scheduler.store.path), "legacy", "model", root, {}, execute=False)
    try:
        delivery = bridge.delivery_status()
    finally:
        bridge.close()
        scheduler.store.close()
    for item in delivery["task_results"]:
        assert set(item) == {"task_id", "operator_key", "stage", "status", "attempt",
                            "evidence_sha256", "submission"}
