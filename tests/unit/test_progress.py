"""Explanations follow real scheduler transitions without causing transitions."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from core.paths import REPO_ROOT
from engine import IOSpec, OperatorSpec, TaskScheduler
from engine.progress import graph_progress, run_progress
from engine.scheduler import EventStore
from tests.scheduler_helpers import simulation_result


@pytest.fixture
def scheduler(tmp_path):
    scheduler = TaskScheduler(tmp_path / "state with spaces.sqlite")
    scheduler.create_run(run_id="r", model_id="m", metadata={"evidence_mode": "simulation"})
    scheduler.discover_operator("r", OperatorSpec(
        operator_id="op", model_id="m", model_revision="1", plugin_revision="1", backend="xpu",
        inputs=[IOSpec("x", "float32", [1], "contiguous")],
        outputs=[IOSpec("y", "float32", [1], "contiguous")], semantics={"op": "identity"},
    ))
    yield scheduler
    scheduler.store.close()


def view(scheduler, **kwargs):
    return run_progress(scheduler.store, "r", **kwargs)["progress"]


def complete(scheduler, task, tmp_path, **kwargs):
    result = simulation_result(task, tmp_path, **kwargs)
    scheduler.complete(task.task_id, kwargs.get("worker_id", "worker"), result, task.lease_token)


def test_unclaimed_running_and_expired_are_distinct_read_only_states(scheduler):
    pending = view(scheduler)
    assert pending["reason_code"] == "WORKER_UNCLAIMED"
    assert pending["next_action"]["command"][3] == scheduler.store.path
    assert "database-wide" in pending["next_action"]["instruction"]
    claimed = scheduler.claim_ready("worker", stage="torch", lease_seconds=100)[0]
    before = [event.to_dict() for event in scheduler.store.events("r")]
    assert view(scheduler)["reason_code"] == "WORKER_RUNNING"
    assert view(scheduler)["next_action"]["owner"] == "worker"
    expired = view(scheduler, now=claimed.lease_expires + 1)
    assert expired["reason_code"] == "LEASE_EXPIRED"
    assert expired["next_action"]["action"] == "RECLAIM_TASK"
    assert scheduler.store.get_task(claimed.task_id).status == "running"
    assert [event.to_dict() for event in scheduler.store.events("r")] == before


@pytest.mark.parametrize("action", ["RETRY", "BLOCKED", "DISPATCH_XPU_FIX", "REDISCOVER_OPERATOR"])
def test_diagnosis_explains_application_or_external_work(scheduler, tmp_path, action):
    source = scheduler.claim_ready("worker", stage="torch")[0]
    scheduler.fail(source.task_id, "worker", "original kernel failure", source.lease_token)
    assert view(scheduler)["reason_code"] == "DIAGNOSIS_PENDING"
    diagnosis = scheduler.claim_ready("worker", stage="diagnosis")[0]
    complete(scheduler, diagnosis, tmp_path, next_action=action)
    progress = view(scheduler)
    if action in {"RETRY", "BLOCKED"}:
        assert progress["reason_code"] == "DIAGNOSIS_NOT_APPLIED"
        command = progress["next_action"]["command"]
        # Execute the actual suggested command, not a hand-built equivalent.
        applied = subprocess.run(command, capture_output=True, text=True, check=False)
        assert applied.returncode == (0 if action == "RETRY" else 6), applied.stdout + applied.stderr
        if action == "RETRY":
            assert view(scheduler)["reason_code"] == "WORKER_UNCLAIMED"
            retry = scheduler.claim_ready("worker-2", stage="torch")[0]
            assert retry.attempt == source.attempt + 1
            scheduler.fail(retry.task_id, "worker-2", "second failure", retry.lease_token)
            assert view(scheduler)["reason_code"] == "DIAGNOSIS_PENDING"
        else:
            assert view(scheduler)["reason_code"] == "DIAGNOSIS_BLOCKED"
            assert view(scheduler)["next_action"]["command"] is None
    else:
        assert progress["reason_code"] == "EXTERNAL_REPAIR_REQUIRED"
        assert progress["next_action"]["action"] == action
        assert progress["next_action"]["command"] is None


def test_saved_graph_wait_is_recomputed_after_worker_completion_and_restart(scheduler, tmp_path):
    scheduler.record_graph_transition("r", "graph_progress", {
        "status": "WAITING", "reason_code": "WAITING_FOR_OPERATORS",
        "node": "mat-026", "observed_at": time.time(), "resume_command": ["graph", "--resume"],
    })
    for stage in ("torch", "xpu", "integration"):
        task = scheduler.claim_ready("worker", stage=stage)[0]
        complete(scheduler, task, tmp_path)
    store = EventStore(scheduler.store.path, readonly=True)
    try:
        progress = run_progress(store, "r")["progress"]
        assert progress["state"] == "READY"
        assert progress["reason_code"] == "RESUME_GRAPH"
        assert progress["next_action"]["command"] == ["graph", "--resume"]
        assert progress["details"]["evidence_revalidated"] is False
    finally:
        store.close()


def test_graph_contract_block_is_not_hidden_by_runnable_queue(scheduler):
    scheduler.record_graph_transition("r", "graph_progress", {
        "status": "BLOCKED", "reason_code": "DISPATCH_BLOCKED", "node": "mat-024",
        "message": "entries[0]: shape is required", "observed_at": time.time(),
        "artifacts": ["/external/dispatch_status.json"],
    })
    progress = view(scheduler)
    assert progress["reason_code"] == "DISPATCH_BLOCKED"
    assert "shape" in progress["summary"]
    assert progress["next_action"]["action"] == "COMPLETE_OPERATOR_CONTRACT"


def test_cli_text_and_json_explain_same_queue_without_writes(scheduler):
    before = Path(scheduler.store.path).read_bytes()
    before_events = [event.to_dict() for event in scheduler.store.events("r")]
    command = [sys.executable, str(REPO_ROOT / "cli/adaptation.py"), "--state", scheduler.store.path,
               "status", "--run-id", "r"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    human = subprocess.run(command + ["--format", "text"], capture_output=True, text=True, check=True)
    assert data["progress"]["reason_code"] in human.stdout
    assert data["progress"]["location"] in human.stdout
    assert "Command:" in human.stdout
    assert Path(scheduler.store.path).read_bytes() == before
    assert [event.to_dict() for event in scheduler.store.events("r")] == before_events


def test_partial_and_graph_only_completion_are_not_functional_delivery():
    for code in ("UNTIL_NODE_REACHED", "GRAPH_ONLY_COMPLETE", "PLAN_COMPLETE"):
        assert graph_progress({"status": "DELIVERED", "reason_code": code})["state"] != "COMPLETED"
    for status in ("FUNCTIONAL_READY", "SIMULATION_PASS"):
        assert graph_progress({"status": status, "reason_code": "DELIVERY_RECORDED"})["state"] == "COMPLETED"


def test_new_work_invalidates_historical_delivery_explanation(scheduler):
    scheduler.record_graph_transition("r", "graph_progress", {
        "status": "SIMULATION_PASS", "reason_code": "DELIVERY_RECORDED", "observed_at": 0,
    })
    assert view(scheduler)["reason_code"] == "WORKER_UNCLAIMED"


def test_evidence_rejection_is_not_hidden_by_succeeded_task_rows(scheduler, tmp_path):
    for stage in ("torch", "xpu", "integration"):
        complete(scheduler, scheduler.claim_ready("worker", stage=stage)[0], tmp_path)
    scheduler.record_graph_transition("r", "graph_progress", {
        "status": "BLOCKED", "reason_code": "OPERATORS_BLOCKED", "observed_at": time.time(),
        "message": "torch evidence hash mismatch", "node": "mat-026",
    })
    progress = view(scheduler)
    assert progress["state"] == "BLOCKED"
    assert "hash mismatch" in progress["summary"]


def test_latest_environment_attempt_takes_precedence_over_old_failure(scheduler):
    scheduler.record_graph_transition("r", "graph_progress", {
        "status": "RUNNING", "node": "kdp-001a-environment-proof",
        "reason_code": "NODE_STARTED", "observed_at": time.time(),
    })
    scheduler.record_environment_failure("r", {}, "Pod unavailable")
    assert view(scheduler)["reason_code"] == "ENVIRONMENT_FAILED"
    scheduler.record_graph_transition("r", "graph_progress", {
        "status": "BLOCKED", "reason_code": "INVALID_INPUT", "observed_at": time.time(),
        "message": "environment proof rejected",
    })
    assert view(scheduler)["reason_code"] == "ENVIRONMENT_FAILED"
    scheduler.record_graph_transition("r", "graph_progress", {
        "status": "RUNNING", "node": "kdp-001a-environment-proof",
        "reason_code": "NODE_STARTED", "observed_at": time.time(),
    })
    assert view(scheduler)["reason_code"] == "NODE_STARTED"
    assert view(scheduler)["state"] == "RUNNING"
