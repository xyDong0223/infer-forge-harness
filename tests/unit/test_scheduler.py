import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from orchestration import AdaptationRun, EventStore, IOSpec, OperatorSpec, TaskScheduler
from orchestration.contracts import BugReport, DiagnosticTask


def _spec():
    return OperatorSpec(
        operator_id="add", model_id="m", model_revision="1", plugin_revision="p", backend="xpu",
        inputs=[IOSpec("x", "float32", [1], "contiguous")],
        outputs=[IOSpec("y", "float32", [1], "contiguous")], semantics={"reference": "torch"},
    )


def test_scheduler_chain_and_idempotency():
    with tempfile.TemporaryDirectory() as d:
        scheduler = TaskScheduler(Path(d) / "state.db")
        scheduler.create_run(AdaptationRun(run_id="r", model_id="m"))
        first = scheduler.discover_operator("r", _spec())
        assert scheduler.discover_operator("r", _spec()).task_id == first.task_id
        for stage in ("torch", "xpu", "integration"):
            claimed = scheduler.claim_ready("worker", stage=stage)
            assert len(claimed) == 1
            scheduler.complete(claimed[0].task_id, worker_id="worker", result={"stage": stage})
        assert [t.status for t in scheduler.store.tasks("r")] == ["succeeded"] * 3
        assert [t.stage for t in scheduler.store.tasks("r")] == ["torch", "xpu", "integration"]


def test_scheduler_recovers_expired_lease_after_restart():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "state.db"
        s1 = TaskScheduler(path)
        s1.create_run(run_id="r", model_id="m")
        s1.discover_operator("r", _spec())
        claimed = s1.claim_ready("dead-worker", lease_seconds=0.01)[0]
        time.sleep(0.03)
        s2 = TaskScheduler(path)
        assert s2.recover() == 1
        assert s2.claim_ready("new-worker")[0].task_id == claimed.task_id


def test_events_are_durable():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "state.db"
        s = TaskScheduler(path)
        s.create_run(run_id="r", model_id="m")
        s.discover_operator("r", _spec())
        assert [e.event_type for e in EventStore(path).events("r")] == ["run_created", "operator_discovered", "task_created"]


def test_rejected_result_marks_task_failed_without_advancing_pipeline():
    with tempfile.TemporaryDirectory() as d:
        scheduler = TaskScheduler(Path(d) / "state.db")
        scheduler.create_run(run_id="r", model_id="m")
        task = scheduler.discover_operator("r", _spec())
        claimed = scheduler.claim_ready("worker")[0]
        failed = scheduler.complete(claimed.task_id, worker_id="worker", result={"status": "failed"})
        assert failed.status == "failed"
        assert scheduler.store.pending_tasks("r", stage="xpu") == []
        assert scheduler.task_status(task.task_id).status == "failed"
        assert [e.event_type for e in scheduler.store.events("r")][-1] == "task_failed"


def test_pending_task_query_and_concurrent_claim_are_idempotent():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "state.db"
        scheduler = TaskScheduler(path)
        scheduler.create_run(run_id="r", model_id="m")
        scheduler.discover_operator("r", _spec())
        assert len(scheduler.pending_tasks(run_id="r", stage="torch")) == 1

        def claim(worker):
            return scheduler.claim_ready(worker, stage="torch")

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(claim, ["a", "b"]))
        claimed = [task for batch in results for task in batch]
        assert len(claimed) == 1
        assert scheduler.pending_tasks(run_id="r", stage="torch") == []


def test_failure_dispatches_idempotent_diagnosis_with_structured_evidence():
    with tempfile.TemporaryDirectory() as d:
        scheduler = TaskScheduler(Path(d) / "state.db")
        scheduler.create_run(run_id="r", model_id="m")
        task = scheduler.discover_operator("r", _spec())
        claimed = scheduler.claim_ready("worker")[0]
        scheduler.fail(claimed.task_id, worker_id="worker", error={
            "error_type": "RuntimeError", "message": "missing kernel",
            "traceback": "trace", "context": {"shape": [1, 4]},
        })
        diagnosis = scheduler.diagnosis_for(task.task_id)
        assert isinstance(diagnosis, DiagnosticTask)
        assert diagnosis.status == "pending"
        assert diagnosis.input["bug_report"]["message"] == "missing kernel"
        assert len(scheduler.pending_tasks("r", stage="diagnosis")) == 1
        assert len([t for t in scheduler.store.tasks("r") if t.stage == "diagnosis"]) == 1


def test_diagnosis_returns_repair_conclusion_to_main_agent():
    with tempfile.TemporaryDirectory() as d:
        scheduler = TaskScheduler(Path(d) / "state.db")
        scheduler.create_run(run_id="r", model_id="m")
        source = scheduler.discover_operator("r", _spec())
        scheduler.claim_ready("worker")[0]
        scheduler.fail(source.task_id, worker_id="worker", error="bad dispatch")
        diag = scheduler.claim_ready("diagnoser", stage="diagnosis")[0]
        assert isinstance(diag, DiagnosticTask)
        scheduler.complete(diag.task_id, worker_id="diagnoser", result={
            "status": "needs_patch", "diagnosis": "registration mismatch",
            "repair_conclusion": {"action": "update registry", "confidence": 0.9},
        })
        completed = [e for e in scheduler.store.events("r") if e.event_type == "diagnostic_completed"][-1]
        assert completed.payload["source_task_id"] == source.task_id
        assert completed.payload["repair_conclusion"]["action"] == "update registry"
        assert scheduler.diagnosis_for(source.task_id).status == "succeeded"


def test_bug_report_normalizes_scalar_errors():
    report = BugReport.from_error("task", "timeout")
    assert report.source_task_id == "task"
    assert report.message == "timeout"


def test_each_failed_stage_gets_its_own_diagnosis_task():
    with tempfile.TemporaryDirectory() as d:
        scheduler = TaskScheduler(Path(d) / "state.db")
        scheduler.create_run(run_id="r", model_id="m")
        source = scheduler.discover_operator("r", _spec())
        torch = scheduler.claim_ready("torch")[0]
        scheduler.complete(torch.task_id, worker_id="torch", result={"ok": True})
        xpu = scheduler.claim_ready("xpu", stage="xpu")[0]
        scheduler.fail(xpu.task_id, worker_id="xpu", error="compile error")
        # The failed torch and xpu stages can both be diagnosed independently
        # when the earlier failure is reported after a retry/replay.
        scheduler.store.db.execute("UPDATE tasks SET status='running' WHERE task_id=?", (source.task_id,))
        scheduler.store.db.commit()
        scheduler.fail(source.task_id, error="runtime error")
        diagnoses = [t for t in scheduler.store.tasks("r") if t.stage == "diagnosis"]
        assert len(diagnoses) == 2


def test_environment_required_run_cannot_discover_before_binding():
    with tempfile.TemporaryDirectory() as d:
        scheduler = TaskScheduler(Path(d) / "state.db")
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
