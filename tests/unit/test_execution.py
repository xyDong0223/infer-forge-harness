"""Managed process occupancy is durable and independent of worker lease expiry."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Barrier

import pytest

from engine import IOSpec, OperatorSpec, TaskScheduler
from engine.execution import (
    ExecutionConflict, begin_execution, canonical_resource, finish_execution,
    get_execution, has_active_task_execution, heartbeat, is_active,
    list_executions, mark_started, reconcile_execution, resource_key,
)
from engine.scheduler import EventStore


POD = {"cluster": "cluster-a", "namespace": "adaptations", "pod_uid": "prepared-pod-uid"}


@pytest.fixture
def scheduler(tmp_path):
    instance = TaskScheduler(tmp_path / "state.sqlite")
    for name in ("one", "two"):
        instance.create_run(
            run_id=name, model_id="demo", backend="kunlun",
            environment={"fingerprint": f"different-{name}"},
            metadata={"evidence_mode": "simulation", "artifact_root": str(tmp_path / name)},
        )
    yield instance
    instance.store.close()


def payload(tmp_path, **extra):
    return {"argv": ["python3", "-c", "print('test')"],
            "cwd": str(tmp_path), "output_dir": str(tmp_path / "one" / "output"), **extra}


def claimed_task(scheduler, name="one"):
    spec = OperatorSpec(
        operator_id="identity", model_id="demo", model_revision="model-v1",
        plugin_revision="plugin-v1", backend="kunlun",
        inputs=[IOSpec("x", "float32", [1], "contiguous")],
        outputs=[IOSpec("y", "float32", [1], "contiguous")],
        semantics={"operation": "identity"},
    )
    scheduler.discover_operator(name, spec)
    task = scheduler.claim_ready("producer", run_id=name, lease_seconds=60)[0]
    return task, {"task_id": task.task_id, "worker": "producer", "lease_token": task.lease_token}


def owner(credentials):
    return {key: value for key, value in credentials.items() if key != "task_id"}


def snapshot(scheduler):
    return list(scheduler.store.db.iterdump())


def test_resource_identity_uses_pod_uid_not_environment_fingerprint(scheduler, tmp_path):
    first, new = begin_execution(
        scheduler, "one", "first", payload(tmp_path, environment_fingerprint="old"), resource=POD,
    )
    assert new and first["state"] == "STARTING"
    assert first["resource_key"] == resource_key({
        "pod_uid": " prepared-pod-uid ", "namespace": "adaptations", "cluster": "cluster-a",
    })
    before = snapshot(scheduler)
    with pytest.raises(ExecutionConflict, match="unresolved execution"):
        begin_execution(scheduler, "two", "second",
                        payload(tmp_path, environment_fingerprint="new"), resource=POD)
    assert snapshot(scheduler) == before
    assert len(list_executions(scheduler.store, active_only=True)) == 1


@pytest.mark.parametrize("resource", [
    {}, {"cluster": "cluster-a", "namespace": "adaptations"},
    {**POD, "pod_uid": ""}, {**POD, "namespace": " "},
    {**POD, "cluster": 1}, {**POD, "fingerprint": "does-not-identify-the-pod"},
])
def test_incomplete_or_ambiguous_resources_are_rejected(scheduler, tmp_path, resource):
    before = snapshot(scheduler)
    with pytest.raises(ValueError, match="resource"):
        begin_execution(scheduler, "one", "exec", payload(tmp_path), resource=resource)
    assert snapshot(scheduler) == before


def test_concurrent_runs_cannot_reserve_the_same_pod(scheduler, tmp_path):
    clients = [TaskScheduler(tmp_path / "state.sqlite") for _ in range(2)]
    barrier = Barrier(2)

    def reserve(index):
        try:
            barrier.wait(timeout=10)
            record, new = begin_execution(
                clients[index], ("one", "two")[index], f"exec-{index}", payload(tmp_path), resource=POD,
            )
            return record["execution_id"] if new else None
        except ExecutionConflict:
            return None
        finally:
            clients[index].store.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(reserve, range(2)))
    assert len([result for result in results if result]) == 1
    assert len(list_executions(scheduler.store, active_only=True)) == 1
    assert len([event for event in scheduler.store.events()
                if event.event_type == "execution_reserved"]) == 1


def test_same_execution_request_is_idempotent_and_conflicting_payload_rejected(scheduler, tmp_path):
    first, new = begin_execution(scheduler, "one", "exec", payload(tmp_path), resource=POD)
    before = snapshot(scheduler)
    repeated, new_again = begin_execution(scheduler, "one", "exec", payload(tmp_path), resource=POD)
    assert first == repeated and new and not new_again
    assert snapshot(scheduler) == before
    with pytest.raises(ExecutionConflict, match="different request"):
        begin_execution(scheduler, "one", "exec", payload(tmp_path, argv=["different"]), resource=POD)
    assert snapshot(scheduler) == before


def test_unconfirmed_execution_blocks_same_task_without_a_pod(scheduler, tmp_path):
    task, credentials = claimed_task(scheduler)
    record, _ = begin_execution(scheduler, "one", "first", payload(tmp_path), **credentials)
    assert record["task_attempt"] == task.attempt
    assert has_active_task_execution(scheduler.store, task.task_id)
    with pytest.raises(ExecutionConflict):
        begin_execution(scheduler, "one", "second", payload(tmp_path), **credentials)
    finish_execution(scheduler, "one", "first", state="FAILED", returncode=1,
                     termination_confirmed=True, **owner(credentials))
    assert not has_active_task_execution(scheduler.store, task.task_id)
    _, new = begin_execution(scheduler, "one", "second", payload(tmp_path), **credentials)
    assert new


def test_current_lease_and_matching_run_are_required_before_reservation(scheduler, tmp_path, monkeypatch):
    task, credentials = claimed_task(scheduler)
    before = snapshot(scheduler)
    with pytest.raises(ValueError, match="lease token"):
        begin_execution(scheduler, "one", "wrong", payload(tmp_path),
                        **{**credentials, "lease_token": "wrong"})
    with pytest.raises(ValueError, match="does not belong"):
        begin_execution(scheduler, "two", "wrong-run", payload(tmp_path), **credentials)
    assert snapshot(scheduler) == before
    monkeypatch.setattr("engine.execution.time.time", lambda: task.lease_expires + 1)
    with pytest.raises(ValueError, match="unexpired"):
        begin_execution(scheduler, "one", "expired", payload(tmp_path), **credentials)
    assert snapshot(scheduler) == before


def test_lease_expiry_never_releases_starting_or_unknown_execution(scheduler, tmp_path, monkeypatch):
    task, credentials = claimed_task(scheduler)
    begin_execution(scheduler, "one", "exec", payload(tmp_path), resource=POD, **credentials)
    monkeypatch.setattr("engine.execution.time.time", lambda: task.lease_expires + 1000)
    # A fresh scheduler process observes the same occupancy; it does not infer
    # termination from missing heartbeat, missing pid, or an expired task lease.
    restarted = TaskScheduler(tmp_path / "state.sqlite")
    try:
        assert get_execution(restarted.store, "one", "exec")["state"] == "STARTING"
        assert has_active_task_execution(restarted.store, task.task_id)
        with pytest.raises(ExecutionConflict):
            begin_execution(restarted, "two", "next", payload(tmp_path), resource=POD)
        unknown = finish_execution(restarted, "one", "exec", state="UNKNOWN",
                                   error="lost contact", **owner(credentials))
        assert is_active(unknown)
        with pytest.raises(ExecutionConflict):
            begin_execution(restarted, "two", "next", payload(tmp_path), resource=POD)
        with pytest.raises(ValueError, match="unexpired"):
            heartbeat(restarted, "one", "exec", **owner(credentials))
    finally:
        restarted.store.close()


def test_original_owner_can_confirm_exit_after_expiry_but_cannot_start_new_work(
    scheduler, tmp_path, monkeypatch,
):
    task, credentials = claimed_task(scheduler)
    initial, _ = begin_execution(scheduler, "one", "exec", payload(tmp_path), resource=POD, **credentials)
    monkeypatch.setattr("engine.execution.time.time", lambda: task.lease_expires + 1)
    repeated, new = begin_execution(scheduler, "one", "exec", payload(tmp_path), resource=POD, **credentials)
    assert repeated == initial and not new
    with pytest.raises(ValueError, match="unexpired"):
        mark_started(scheduler, "one", "exec", pid=123, process_identity="start-tick-1",
                     host_id="local-host", **owner(credentials))
    with pytest.raises(ValueError, match="original worker"):
        finish_execution(scheduler, "one", "exec", state="FAILED", termination_confirmed=True,
                         **{**owner(credentials), "lease_token": "different"})
    finished = finish_execution(scheduler, "one", "exec", state="FAILED", returncode=-15,
                                termination_confirmed=True, **owner(credentials))
    assert not is_active(finished)
    assert not has_active_task_execution(scheduler.store, task.task_id)
    _, new = begin_execution(scheduler, "two", "next", payload(tmp_path), resource=POD)
    assert new


def test_process_identity_and_heartbeat_are_persisted_without_token(scheduler, tmp_path):
    task, credentials = claimed_task(scheduler)
    begin_execution(scheduler, "one", "exec", payload(tmp_path), resource=POD, **credentials)
    started = mark_started(
        scheduler, "one", "exec", pid=123, process_identity={"pid": 123, "start": "tick-1"},
        host_id="host-a", remote_handle={"job_id": "remote-job-1"}, **owner(credentials),
    )
    assert started["state"] == "RUNNING" and started["pid"] == 123
    assert started["remote_handle"] == {"job_id": "remote-job-1"}
    before = snapshot(scheduler)
    assert mark_started(
        scheduler, "one", "exec", pid=123, process_identity={"pid": 123, "start": "tick-1"},
        host_id="host-a", remote_handle={"job_id": "remote-job-1"}, **owner(credentials),
    ) == started
    assert snapshot(scheduler) == before
    with pytest.raises(ExecutionConflict, match="STARTING"):
        mark_started(scheduler, "one", "exec", pid=456, process_identity="other-process",
                     host_id="host-a", **owner(credentials))
    beat = heartbeat(scheduler, "one", "exec", **owner(credentials))
    assert beat["revision"] == started["revision"] + 1 and beat["heartbeat_at"] is not None
    serialized = json.dumps(scheduler.store.run("one").metadata)
    assert task.lease_token not in serialized
    assert task.lease_token not in json.dumps([event.to_dict() for event in scheduler.store.events()])
    assert "lease_token_sha256" in serialized


@pytest.mark.parametrize("state,returncode,confirmed", [
    ("SUCCEEDED", 0, False), ("FAILED", 1, False), ("UNKNOWN", None, True),
    ("SUCCEEDED", 1, True), ("SUCCEEDED", None, True), ("RUNNING", None, False),
])
def test_unconfirmed_or_inconsistent_finish_never_releases(
    scheduler, tmp_path, state, returncode, confirmed,
):
    begin_execution(scheduler, "one", "exec", payload(tmp_path), resource=POD)
    before = snapshot(scheduler)
    with pytest.raises(ValueError):
        finish_execution(scheduler, "one", "exec", state=state, returncode=returncode,
                         termination_confirmed=confirmed)
    assert snapshot(scheduler) == before
    assert len(list_executions(scheduler.store, active_only=True)) == 1


def test_confirmed_finish_releases_resource_and_has_idempotent_receipt(scheduler, tmp_path):
    begin_execution(scheduler, "one", "exec", payload(tmp_path), resource=POD)
    result = finish_execution(scheduler, "one", "exec", state="SUCCEEDED", returncode=0,
                              log_path=str(tmp_path / "one" / "output" / "run.log"),
                              termination_confirmed=True)
    assert not is_active(result)
    before = snapshot(scheduler)
    repeated = finish_execution(scheduler, "one", "exec", state="SUCCEEDED", returncode=0,
                                log_path=str(tmp_path / "one" / "output" / "run.log"),
                                termination_confirmed=True)
    assert repeated == result and snapshot(scheduler) == before
    with pytest.raises(ExecutionConflict, match="cannot be replaced"):
        finish_execution(scheduler, "one", "exec", state="FAILED", returncode=1,
                         termination_confirmed=True)
    _, new = begin_execution(scheduler, "two", "next", payload(tmp_path), resource=POD)
    assert new
    previous, new = begin_execution(scheduler, "one", "exec", payload(tmp_path), resource=POD)
    assert previous == result and not new


def test_reconciliation_requires_actual_terminal_observation(scheduler, tmp_path):
    begin_execution(scheduler, "one", "exec", payload(tmp_path), resource=POD)
    with pytest.raises(ValueError, match="observer"):
        reconcile_execution(scheduler, "one", "exec", observer=True)
    with pytest.raises(ValueError, match="identify"):
        reconcile_execution(scheduler, "one", "exec", observer=lambda _: {
            "execution_id": "different", "terminal": True, "termination_confirmed": True,
            "state": "SUCCEEDED", "returncode": 0,
        })
    unknown = reconcile_execution(scheduler, "one", "exec", observer=lambda _: {
        "execution_id": "exec", "terminal": False, "termination_confirmed": False,
        "error": "old host unavailable",
    })
    assert unknown["state"] == "UNKNOWN" and is_active(unknown)
    with pytest.raises(ExecutionConflict):
        begin_execution(scheduler, "two", "next", payload(tmp_path), resource=POD)
    finished = reconcile_execution(scheduler, "one", "exec", observer=lambda record: {
        "execution_id": record["execution_id"], "terminal": True,
        "termination_confirmed": True, "state": "FAILED", "returncode": -15,
        "error": "observer confirmed owned process termination",
    })
    assert not is_active(finished)
    _, new = begin_execution(scheduler, "two", "next", payload(tmp_path), resource=POD)
    assert new


def test_stale_observation_cannot_finish_concurrently_changed_execution(scheduler, tmp_path):
    begin_execution(scheduler, "one", "exec", payload(tmp_path), resource=POD)

    def observer(record):
        heartbeat(scheduler, "one", "exec")
        return {"execution_id": record["execution_id"], "terminal": True,
                "termination_confirmed": True, "state": "SUCCEEDED", "returncode": 0}

    with pytest.raises(ExecutionConflict, match="changed during"):
        reconcile_execution(scheduler, "one", "exec", observer=observer)
    assert is_active(get_execution(scheduler.store, "one", "exec"))


def test_readonly_observation_does_not_renew_or_reconcile(scheduler, tmp_path):
    task, credentials = claimed_task(scheduler)
    begin_execution(scheduler, "one", "exec", payload(tmp_path), resource=POD, **credentials)
    before = snapshot(scheduler)
    store = EventStore(tmp_path / "state.sqlite", readonly=True)
    try:
        store.db.execute("BEGIN")
        assert get_execution(store, "one", "exec")["state"] == "STARTING"
        assert get_execution(store, "one", "missing") is None
        assert len(list_executions(store, "one", active_only=True)) == 1
        assert has_active_task_execution(store, task.task_id)
    finally:
        store.close()
    assert snapshot(scheduler) == before


def test_runtime_output_in_checkout_and_token_bearing_payload_are_rejected(scheduler, tmp_path):
    task, credentials = claimed_task(scheduler)
    before = snapshot(scheduler)
    with pytest.raises(ValueError, match="source repository"):
        begin_execution(scheduler, "one", "bad-path",
                        payload(tmp_path, output_dir=str(Path(__file__).resolve().parents[2] / "runtime")))
    with pytest.raises(ValueError, match="lease token"):
        begin_execution(scheduler, "one", "secret",
                        payload(tmp_path, argv=["command", "--token", task.lease_token]), **credentials)
    assert snapshot(scheduler) == before
