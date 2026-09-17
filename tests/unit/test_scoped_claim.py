"""Run/task claims isolate recovery as well as selection in the durable scheduler."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from engine import IOSpec, OperatorSpec, TaskScheduler


@pytest.fixture
def scheduler(tmp_path):
    instance = TaskScheduler(tmp_path / "state.db")
    yield instance
    instance.store.close()


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("engine.scheduler.time.time", lambda: now[0])
    return now


def create_run(scheduler, tmp_path, run_id):
    return scheduler.create_run(
        run_id=run_id, model_id="model", backend="xpu",
        metadata={"evidence_mode": "simulation", "artifact_root": str(tmp_path / run_id)},
    )


def discover(scheduler, run_id, operator_id="identity"):
    return scheduler.discover_operator(run_id, OperatorSpec(
        operator_id=operator_id, model_id="model", model_revision="1",
        plugin_revision="1", backend="xpu",
        inputs=[IOSpec("x", "float32", [1], "contiguous")],
        outputs=[IOSpec("y", "float32", [1], "contiguous")],
        semantics={"operation": operator_id},
    ))


def task_row(scheduler, task_id):
    return dict(scheduler.store.db.execute(
        "SELECT * FROM tasks WHERE task_id=?", (task_id,),
    ).fetchone())


def persistent_snapshot(scheduler, tmp_path):
    """Read-only comparison catches hidden lease, event, and workspace writes."""
    return {
        "database": list(scheduler.store.db.iterdump()),
        "artifacts": {
            str(path.relative_to(tmp_path)): path.read_bytes()
            for run in scheduler.store.db.execute("SELECT run_id FROM runs")
            for path in (tmp_path / run["run_id"]).rglob("*") if path.is_file()
        },
    }


def test_run_scope_does_not_claim_or_recover_another_run(scheduler, tmp_path, clock):
    for run_id in ("other", "selected"):
        create_run(scheduler, tmp_path, run_id)
        discover(scheduler, run_id)
    other = scheduler.claim_ready("other-worker", lease_seconds=1, run_id="other")[0]
    other_pending = discover(scheduler, "other", "add")
    before = task_row(scheduler, other.task_id)
    other_events = scheduler.store.events("other")
    clock[0] += 2

    selected = scheduler.claim_ready("worker", run_id="selected", limit=10)

    assert [task.run_id for task in selected] == ["selected"]
    assert task_row(scheduler, other.task_id) == before
    assert scheduler.task_status(other_pending.task_id).status == "pending"
    assert scheduler.store.events("other") == other_events


def test_task_scope_recovers_only_target_with_fresh_attempt(scheduler, tmp_path, clock):
    create_run(scheduler, tmp_path, "run")
    sibling = discover(scheduler, "run", "add")
    target = discover(scheduler, "run", "identity")
    first_sibling, first = scheduler.claim_ready("worker", lease_seconds=1, run_id="run", limit=2)
    assert first_sibling.task_id == sibling.task_id
    assert first.task_id == target.task_id
    prior_output = Path(first.input["workspace"]["output"])
    (prior_output / "interrupted.txt").write_text("preserve previous evidence")
    sibling_before = task_row(scheduler, sibling.task_id)
    clock[0] += 2

    retried = scheduler.claim_ready("new-worker", run_id="run", task_id=target.task_id, limit=10)

    assert len(retried) == 1
    second = retried[0]
    assert second.task_id == target.task_id
    assert second.attempt == first.attempt + 1
    assert second.lease_token != first.lease_token
    assert second.input["workspace"]["root"] != first.input["workspace"]["root"]
    assert (prior_output / "interrupted.txt").read_text() == "preserve previous evidence"
    assert task_row(scheduler, sibling.task_id) == sibling_before


@pytest.mark.parametrize("scoped", [False, True])
def test_stage_filter_also_limits_expired_lease_recovery(scheduler, tmp_path, clock, scoped):
    create_run(scheduler, tmp_path, "run")
    discover(scheduler, "run", "broken")
    source = scheduler.claim_ready("worker", run_id="run")[0]
    scheduler.fail(source.task_id, "worker", "test failure", source.lease_token)
    diagnosis = scheduler.claim_ready("diagnoser", stage="diagnosis", lease_seconds=1, run_id="run")[0]
    discover(scheduler, "run", "identity")
    first = scheduler.claim_ready("worker", stage="torch", lease_seconds=1, run_id="run")[0]
    diagnosis_before = task_row(scheduler, diagnosis.task_id)
    clock[0] += 2

    selected = scheduler.claim_ready("worker-2", stage="torch", **({"run_id": "run"} if scoped else {}))

    assert [task.task_id for task in selected] == [first.task_id]
    assert selected[0].attempt == 2
    assert task_row(scheduler, diagnosis.task_id) == diagnosis_before


def test_directed_claim_skips_older_pending_work(scheduler, tmp_path):
    create_run(scheduler, tmp_path, "run")
    earlier = discover(scheduler, "run", "add")
    target = discover(scheduler, "run", "identity")

    selected = scheduler.claim_ready("worker", run_id="run", task_id=target.task_id, stage="torch")

    assert [task.task_id for task in selected] == [target.task_id]
    assert scheduler.task_status(earlier.task_id).status == "pending"
    assert scheduler.task_status(earlier.task_id).attempt == 0


@pytest.mark.parametrize(("case", "error", "message"), [
    ("missing_run", ValueError, "task_id requires run_id"),
    ("unknown_run", KeyError, "unknown run"),
    ("unknown_task", KeyError, "unknown task"),
    ("wrong_run", ValueError, "does not belong"),
    ("wrong_stage", ValueError, "does not match requested stage"),
    ("invalid_stage", ValueError, "invalid stage"),
])
def test_invalid_scope_rejected_before_any_state_change(scheduler, tmp_path, clock, case, error, message):
    for run_id in ("run", "other"):
        create_run(scheduler, tmp_path, run_id)
        discover(scheduler, run_id)
    expired = scheduler.claim_ready("worker", lease_seconds=1, run_id="run")[0]
    kwargs = {
        "missing_run": {"task_id": expired.task_id},
        "unknown_run": {"run_id": "missing"},
        "unknown_task": {"run_id": "run", "task_id": "missing"},
        "wrong_run": {"run_id": "other", "task_id": expired.task_id},
        "wrong_stage": {"run_id": "run", "task_id": expired.task_id, "stage": "xpu"},
        "invalid_stage": {"run_id": "run", "stage": "unknown"},
    }[case]
    clock[0] += 2
    before = persistent_snapshot(scheduler, tmp_path)

    with pytest.raises(error, match=message):
        scheduler.claim_ready("new-worker", **kwargs)

    assert persistent_snapshot(scheduler, tmp_path) == before


def test_valid_busy_failed_or_dependency_blocked_target_returns_empty(scheduler, tmp_path):
    create_run(scheduler, tmp_path, "run")
    target = discover(scheduler, "run")
    first = scheduler.claim_ready("worker", run_id="run", task_id=target.task_id)[0]
    before = persistent_snapshot(scheduler, tmp_path)
    assert scheduler.claim_ready("other-worker", run_id="run", task_id=target.task_id) == []
    assert persistent_snapshot(scheduler, tmp_path) == before
    scheduler.fail(first.task_id, "worker", "local failure", first.lease_token)
    assert scheduler.claim_ready("worker", run_id="run", task_id=target.task_id) == []

    create_run(scheduler, tmp_path, "blocked")
    blocked = discover(scheduler, "blocked")
    scheduler.record_graph_transition("blocked", "graph_environment_required", {})
    before = persistent_snapshot(scheduler, tmp_path)
    assert scheduler.claim_ready("worker", run_id="blocked", task_id=blocked.task_id) == []
    assert persistent_snapshot(scheduler, tmp_path) == before


def test_concurrent_scoped_claims_use_independent_connections(scheduler, tmp_path):
    tasks = {}
    for run_id in ("one", "two", "untouched"):
        create_run(scheduler, tmp_path, run_id)
        tasks[run_id] = discover(scheduler, run_id)
    clients = [TaskScheduler(tmp_path / "state.db") for _ in range(4)]
    barrier = Barrier(len(clients))

    def claim(index):
        run_id = "one" if index < 2 else "two"
        try:
            barrier.wait(timeout=10)
            return clients[index].claim_ready(
                f"worker-{index}", run_id=run_id, task_id=tasks[run_id].task_id,
            )
        finally:
            clients[index].store.close()

    with ThreadPoolExecutor(max_workers=4) as pool:
        batches = list(pool.map(claim, range(4)))

    claimed = [task for batch in batches for task in batch]
    assert sorted(task.run_id for task in claimed) == ["one", "two"]
    assert all(task.attempt == 1 for task in claimed)
    assert scheduler.task_status(tasks["untouched"].task_id).status == "pending"
    assert len([event for event in scheduler.store.events() if event.event_type == "task_claimed"]) == 2


def test_unscoped_positional_api_keeps_global_queue(scheduler, tmp_path):
    for run_id in ("one", "two"):
        create_run(scheduler, tmp_path, run_id)
        discover(scheduler, run_id)

    claimed = scheduler.claim_ready("external-worker", None, 300, 10)

    assert {task.run_id for task in claimed} == {"one", "two"}
