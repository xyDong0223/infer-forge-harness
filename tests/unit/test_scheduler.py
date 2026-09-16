import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest

from engine import AdaptationRun, EventStore, IOSpec, OperatorSpec, TaskScheduler
from engine.contracts import BugReport, DiagnosticTask
from tests.scheduler_helpers import simulation_result


def _spec(operator_id="add"):
    return OperatorSpec(
        operator_id=operator_id, model_id="m", model_revision="1", plugin_revision="p", backend="xpu",
        inputs=[IOSpec("x", "float32", [1], "contiguous")],
        outputs=[IOSpec("y", "float32", [1], "contiguous")],
        semantics={"reference": f"torch.ops.demo.{operator_id}"},
    )


def test_scheduler_chain_and_idempotency(tmp_path: Path):
    scheduler = TaskScheduler(tmp_path / "state.db")
    scheduler.create_run(AdaptationRun(run_id="r", model_id="m", metadata={"evidence_mode": "simulation"}))
    first = scheduler.discover_operator("r", _spec())
    assert scheduler.discover_operator("r", _spec()).task_id == first.task_id
    for stage in ("torch", "xpu", "integration"):
        claimed = scheduler.claim_ready("worker", stage=stage)
        assert len(claimed) == 1
        task = claimed[0]
        scheduler.complete(
            task.task_id, worker_id="worker", lease_token=task.lease_token,
            result=simulation_result(task, tmp_path / "artifacts"),
        )
    assert [t.status for t in scheduler.store.tasks("r")] == ["succeeded"] * 3
    assert [t.stage for t in scheduler.store.tasks("r")] == ["torch", "xpu", "integration"]


def test_scheduler_recovers_expired_lease_after_restart(tmp_path: Path):
    path = tmp_path / "state.db"
    s1 = TaskScheduler(path)
    s1.create_run(run_id="r", model_id="m", metadata={"evidence_mode": "simulation"})
    s1.discover_operator("r", _spec())
    now = time.time()
    with patch("engine.scheduler.time.time", return_value=now):
        claimed = s1.claim_ready("dead-worker", lease_seconds=1)[0]
    s1.store.close()
    s2 = TaskScheduler(path)
    with patch("engine.scheduler.time.time", return_value=now + 2):
        assert s2.recover() == 1
        recovered = s2.claim_ready("new-worker")[0]
    assert recovered.task_id == claimed.task_id
    assert recovered.attempt == claimed.attempt + 1
    assert recovered.lease_token != claimed.lease_token


def test_events_are_durable(tmp_path: Path):
    path = tmp_path / "state.db"
    s = TaskScheduler(path)
    s.create_run(run_id="r", model_id="m", metadata={"evidence_mode": "simulation"})
    s.discover_operator("r", _spec())
    assert [e.event_type for e in EventStore(path).events("r")] == ["run_created", "operator_discovered", "task_created"]


def test_rejected_result_marks_task_failed_without_advancing_pipeline(tmp_path: Path):
    scheduler = TaskScheduler(tmp_path / "state.db")
    scheduler.create_run(run_id="r", model_id="m", metadata={"evidence_mode": "simulation"})
    task = scheduler.discover_operator("r", _spec())
    claimed = scheduler.claim_ready("worker")[0]
    failed = scheduler.complete(
        claimed.task_id, worker_id="worker", lease_token=claimed.lease_token,
        result={"status": "failed"},
    )
    assert failed.status == "failed"
    assert scheduler.store.pending_tasks("r", stage="xpu") == []
    assert scheduler.task_status(task.task_id).status == "failed"
    assert [e.event_type for e in scheduler.store.events("r")][-1] == "task_failed"


def test_pending_task_query_and_concurrent_claim_are_idempotent(tmp_path: Path):
    scheduler = TaskScheduler(tmp_path / "state.db")
    scheduler.create_run(run_id="r", model_id="m", metadata={"evidence_mode": "simulation"})
    scheduler.discover_operator("r", _spec())
    assert len(scheduler.pending_tasks(run_id="r", stage="torch")) == 1

    def claim(worker):
        return scheduler.claim_ready(worker, stage="torch")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, ["a", "b"]))
    claimed = [task for batch in results for task in batch]
    assert len(claimed) == 1
    assert scheduler.pending_tasks(run_id="r", stage="torch") == []


def test_failure_dispatches_idempotent_diagnosis_with_structured_evidence(tmp_path: Path):
    scheduler = TaskScheduler(tmp_path / "state.db")
    scheduler.create_run(run_id="r", model_id="m", metadata={"evidence_mode": "simulation"})
    task = scheduler.discover_operator("r", _spec())
    claimed = scheduler.claim_ready("worker")[0]
    scheduler.fail(claimed.task_id, worker_id="worker", lease_token=claimed.lease_token, error={
        "error_type": "RuntimeError", "message": "missing kernel",
        "traceback": "trace", "context": {"shape": [1, 4]},
    })
    diagnosis = scheduler.diagnosis_for(task.task_id)
    assert isinstance(diagnosis, DiagnosticTask)
    assert diagnosis.status == "pending"
    assert diagnosis.input["bug_report"]["message"] == "missing kernel"
    with pytest.raises(ValueError, match="current unexpired lease token"):
        scheduler.fail(
            claimed.task_id, worker_id="worker", lease_token=claimed.lease_token,
            error="duplicate failure",
        )
    assert len(scheduler.pending_tasks("r", stage="diagnosis")) == 1
    assert len([t for t in scheduler.store.tasks("r") if t.stage == "diagnosis"]) == 1


def test_diagnosis_returns_repair_conclusion_to_main_agent(tmp_path: Path):
    scheduler = TaskScheduler(tmp_path / "state.db")
    scheduler.create_run(run_id="r", model_id="m", metadata={"evidence_mode": "simulation"})
    source = scheduler.discover_operator("r", _spec())
    claimed = scheduler.claim_ready("worker")[0]
    scheduler.fail(
        source.task_id, worker_id="worker", lease_token=claimed.lease_token,
        error="bad dispatch",
    )
    diag = scheduler.claim_ready("diagnoser", stage="diagnosis")[0]
    assert isinstance(diag, DiagnosticTask)
    scheduler.complete(
        diag.task_id, worker_id="diagnoser", lease_token=diag.lease_token,
        result=simulation_result(
            diag, tmp_path / "artifacts", worker_id="diagnoser",
            diagnosis="registration mismatch",
            repair_conclusion={"action": "update registry", "confidence": 0.9},
            next_action="DISPATCH_XPU_FIX", confidence=0.9,
        ),
    )
    completed = [e for e in scheduler.store.events("r") if e.event_type == "diagnostic_completed"][-1]
    assert completed.payload["source_task_id"] == source.task_id
    assert completed.payload["repair_conclusion"]["action"] == "update registry"
    assert scheduler.diagnosis_for(source.task_id).status == "succeeded"


def test_completed_retry_diagnosis_requeues_source_with_a_new_attempt(tmp_path: Path):
    scheduler = TaskScheduler(tmp_path / "state.db")
    scheduler.create_run(run_id="r", model_id="m", metadata={"evidence_mode": "simulation"})
    source = scheduler.discover_operator("r", _spec())
    first = scheduler.claim_ready("worker")[0]
    scheduler.fail(
        source.task_id, worker_id="worker", lease_token=first.lease_token,
        error="transient failure",
    )
    diagnosis = scheduler.claim_ready("diagnoser", stage="diagnosis")[0]
    scheduler.complete(
        diagnosis.task_id, worker_id="diagnoser", lease_token=diagnosis.lease_token,
        result=simulation_result(
            diagnosis, tmp_path / "artifacts", worker_id="diagnoser",
            next_action="RETRY",
        ),
    )
    pending = scheduler.apply_diagnosis(diagnosis.task_id)
    assert pending.status == "pending"
    assert pending.attempt == first.attempt
    assert pending.input["recovery"]["diagnosis_task_id"] == diagnosis.task_id
    retried = scheduler.claim_ready("worker-2", stage="torch")[0]
    assert retried.task_id == source.task_id
    assert retried.attempt == first.attempt + 1
    assert scheduler.apply_diagnosis(diagnosis.task_id).task_id == source.task_id
    assert len([
        event for event in scheduler.store.events("r")
        if event.event_type == "diagnosis_applied"
    ]) == 1

    scheduler.fail(
        retried.task_id, "worker-2", "failed again", retried.lease_token
    )
    reopened = scheduler.diagnosis_for(source.task_id)
    assert reopened.status == "pending"
    assert reopened.input["source_attempt"] == retried.attempt
    second_diagnosis = scheduler.claim_ready("diagnoser-2", stage="diagnosis")[0]
    assert second_diagnosis.attempt == diagnosis.attempt + 1
    scheduler.complete(
        second_diagnosis.task_id,
        "diagnoser-2",
        simulation_result(
            second_diagnosis, tmp_path / "artifacts-2", worker_id="diagnoser-2",
            next_action="RETRY",
        ),
        second_diagnosis.lease_token,
    )
    assert scheduler.apply_diagnosis(second_diagnosis.task_id).status == "pending"
    assert len([
        event for event in scheduler.store.events("r")
        if event.event_type == "diagnosis_applied"
    ]) == 2


def test_blocked_diagnosis_is_consumed_without_requeue(tmp_path: Path):
    scheduler = TaskScheduler(tmp_path / "state.db")
    scheduler.create_run(run_id="r", model_id="m", metadata={"evidence_mode": "simulation"})
    source = scheduler.discover_operator("r", _spec())
    claimed = scheduler.claim_ready("worker")[0]
    scheduler.fail(source.task_id, "worker", "permanent", claimed.lease_token)
    diagnosis = scheduler.claim_ready("diagnoser", stage="diagnosis")[0]
    scheduler.complete(
        diagnosis.task_id, "diagnoser",
        simulation_result(
            diagnosis, tmp_path / "artifacts", worker_id="diagnoser",
            next_action="BLOCKED",
        ),
        diagnosis.lease_token,
    )
    assert scheduler.apply_diagnosis(diagnosis.task_id).status == "failed"
    assert scheduler.pending_tasks("r", stage="torch") == []


def test_external_repair_diagnosis_is_not_guessed_by_scheduler(tmp_path: Path):
    scheduler = TaskScheduler(tmp_path / "state.db")
    scheduler.create_run(run_id="r", model_id="m", metadata={"evidence_mode": "simulation"})
    source = scheduler.discover_operator("r", _spec())
    claimed = scheduler.claim_ready("worker")[0]
    scheduler.fail(source.task_id, "worker", "bad implementation", claimed.lease_token)
    diagnosis = scheduler.claim_ready("diagnoser", stage="diagnosis")[0]
    scheduler.complete(
        diagnosis.task_id, "diagnoser",
        simulation_result(
            diagnosis, tmp_path / "artifacts", worker_id="diagnoser",
            next_action="DISPATCH_XPU_FIX",
        ),
        diagnosis.lease_token,
    )
    with pytest.raises(ValueError, match="requires an external repair"):
        scheduler.apply_diagnosis(diagnosis.task_id)
    assert scheduler.task_status(source.task_id).status == "failed"


def test_bug_report_normalizes_scalar_errors():
    report = BugReport.from_error("task", "timeout")
    assert report.source_task_id == "task"
    assert report.message == "timeout"


def test_each_failed_stage_gets_its_own_diagnosis_task(tmp_path: Path):
    scheduler = TaskScheduler(tmp_path / "state.db")
    scheduler.create_run(run_id="r", model_id="m", metadata={"evidence_mode": "simulation"})
    failed_sources = []
    for failed_stage in ("torch", "xpu", "integration"):
        scheduler.discover_operator("r", _spec(f"fails_at_{failed_stage}"))
        for stage in ("torch", "xpu", "integration"):
            task = scheduler.claim_ready(stage, stage=stage)[0]
            if stage == failed_stage:
                scheduler.fail(
                    task.task_id, worker_id=stage, lease_token=task.lease_token,
                    error=f"simulated {stage} failure",
                )
                failed_sources.append(task)
                break
            scheduler.complete(
                task.task_id, worker_id=stage, lease_token=task.lease_token,
                result=simulation_result(task, tmp_path / "artifacts", worker_id=stage),
            )
    diagnoses = scheduler.claim_ready("diagnoser", stage="diagnosis", limit=3)
    assert len(diagnoses) == 3
    assert {task.source_task_id for task in diagnoses} == {task.task_id for task in failed_sources}
    assert {task.input["failed_stage"] for task in diagnoses} == {"torch", "xpu", "integration"}
    for diagnosis in diagnoses:
        scheduler.complete(
            diagnosis.task_id, worker_id="diagnoser", lease_token=diagnosis.lease_token,
            result=simulation_result(diagnosis, tmp_path / "artifacts", worker_id="diagnoser"),
        )
    assert all(scheduler.diagnosis_for(task.task_id).status == "succeeded" for task in failed_sources)
    assert all(scheduler.task_status(task.task_id).status == "failed" for task in failed_sources)


def test_environment_required_run_cannot_discover_before_binding(tmp_path: Path):
    scheduler = TaskScheduler(tmp_path / "state.db")
    scheduler.create_run(
        AdaptationRun(
            run_id="r",
            model_id="m",
            status="WAITING_FOR_ENVIRONMENT",
            metadata={"environment_required": True},
        )
    )
    with pytest.raises(ValueError, match="environment proof"):
        scheduler.discover_operator("r", _spec())
