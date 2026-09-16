"""End-to-end tests for the persistent operator adaptation scheduler."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.contracts import IOSpec, OperatorSpec  # noqa: E402
from engine.scheduler import EventStore, TaskScheduler  # noqa: E402
from tests.scheduler_helpers import simulation_result  # noqa: E402


def make_spec(operator_id: str = "missing_op") -> OperatorSpec:
    tensor = IOSpec(name="x", dtype="float32", shape=[2, 4], layout="contiguous")
    return OperatorSpec(
        operator_id=operator_id,
        model_id="demo-model",
        model_revision="rev-1",
        plugin_revision="kunlun-1",
        backend="kunlun-p800",
        inputs=[tensor],
        outputs=[IOSpec(name="y", dtype="float32", shape=[2, 4], layout="contiguous")],
        semantics={"reference": f"torch.ops.demo.{operator_id}"},
        evidence={"trace": "failure.json"},
    )


def new_scheduler(tmp_path: Path) -> tuple[TaskScheduler, Path]:
    db = tmp_path / "events.sqlite3"
    return TaskScheduler(EventStore(db)), db


def test_fake_agents_complete_full_operator_pipeline_and_main_flow_continues(tmp_path: Path):
    scheduler, _ = new_scheduler(tmp_path)
    run = scheduler.create_run(
        run_id="run-1", model_id="demo-model", backend="kunlun-p800",
        metadata={"evidence_mode": "simulation"},
    )
    spec = make_spec()

    torch_task = scheduler.discover_operator(run.run_id, spec)
    # Discovery is idempotent: replaying the same observation does not dispatch twice.
    assert scheduler.discover_operator(run.run_id, spec).task_id == torch_task.task_id

    # The main flow can continue discovering another gap while torch work is running.
    second = scheduler.discover_operator(run.run_id, make_spec("another_missing_op"))
    assert second.stage == "torch"
    claimed = scheduler.claim_ready("torch-agent", stage="torch", limit=1)
    assert len(claimed) == 1
    first_claim = claimed[0]
    assert first_claim.operator_key == torch_task.operator_key
    scheduler.complete(
        first_claim.task_id, worker_id="torch-agent", lease_token=first_claim.lease_token,
        result=simulation_result(first_claim, tmp_path / "artifacts", worker_id="torch-agent"),
    )
    xpu = next(task for task in scheduler.store.tasks(run.run_id) if task.operator_key == spec.operator_key and task.stage == "xpu")
    assert xpu.status == "pending"

    xpu_claim = scheduler.claim_ready("xpu-agent", stage="xpu")[0]
    scheduler.complete(
        xpu_claim.task_id, worker_id="xpu-agent", lease_token=xpu_claim.lease_token,
        result=simulation_result(xpu_claim, tmp_path / "artifacts", worker_id="xpu-agent"),
    )
    integration = next(task for task in scheduler.store.tasks(run.run_id) if task.operator_key == spec.operator_key and task.stage == "integration")
    assert integration.status == "pending"

    integration_claim = scheduler.claim_ready("integration-agent", stage="integration")[0]
    scheduler.complete(
        integration_claim.task_id,
        worker_id="integration-agent",
        lease_token=integration_claim.lease_token,
        result=simulation_result(integration_claim, tmp_path / "artifacts", worker_id="integration-agent"),
    )

    final = scheduler.store.get_task(integration_claim.task_id)
    assert final is not None and final.status == "succeeded"
    # The unrelated operator remains independently schedulable; no global barrier exists.
    assert scheduler.claim_ready("torch-agent", stage="torch", limit=1)[0].operator_key == second.operator_key

    events = scheduler.store.events(run.run_id)
    stages = [e.payload.get("stage") for e in events if e.event_type == "task_created" and e.payload.get("operator_key") == spec.operator_key]
    assert stages == ["torch", "xpu", "integration"]


def test_restart_recovers_expired_lease_and_does_not_duplicate_tasks(tmp_path: Path):
    scheduler, db = new_scheduler(tmp_path)
    run = scheduler.create_run(
        run_id="run-restart", model_id="demo-model", backend="kunlun-p800",
        metadata={"evidence_mode": "simulation"},
    )
    spec = make_spec()
    task = scheduler.discover_operator(run.run_id, spec)
    now = time.time()
    with patch("engine.scheduler.time.time", return_value=now):
        claimed = scheduler.claim_ready("crashed-worker", stage="torch", lease_seconds=1)[0]
    assert claimed.task_id == task.task_id
    scheduler.store.close()

    # A fresh process sees the same durable task and recovers the expired lease.
    restarted = TaskScheduler(EventStore(db))
    with patch("engine.scheduler.time.time", return_value=now + 2):
        assert restarted.recover() == 1
        recovered = restarted.claim_ready("replacement-worker", stage="torch")[0]
    assert recovered.task_id == task.task_id
    assert recovered.attempt == 2
    assert recovered.lease_token != claimed.lease_token

    restarted.complete(
        recovered.task_id, worker_id="replacement-worker", lease_token=recovered.lease_token,
        result=simulation_result(recovered, tmp_path / "artifacts", worker_id="replacement-worker"),
    )
    xpu = restarted.claim_ready("xpu-agent", stage="xpu")[0]
    restarted.complete(
        xpu.task_id, worker_id="xpu-agent", lease_token=xpu.lease_token,
        result=simulation_result(xpu, tmp_path / "artifacts", worker_id="xpu-agent"),
    )
    integration = restarted.claim_ready("integration-agent", stage="integration")[0]
    restarted.complete(
        integration.task_id, worker_id="integration-agent", lease_token=integration.lease_token,
        result=simulation_result(integration, tmp_path / "artifacts", worker_id="integration-agent"),
    )

    tasks = restarted.store.tasks(run.run_id)
    assert [(item.stage, item.status) for item in tasks] == [
        ("torch", "succeeded"),
        ("xpu", "succeeded"),
        ("integration", "succeeded"),
    ]
    events = restarted.store.events(run.run_id)
    created = [e for e in events if e.event_type == "task_created"]
    assert [(e.payload["operator_key"], e.payload["stage"]) for e in created] == [
        (spec.operator_key, "torch"),
        (spec.operator_key, "xpu"),
        (spec.operator_key, "integration"),
    ]
