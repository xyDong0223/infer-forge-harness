"""Real local subprocesses exercise the managed execution/lease boundary."""

from pathlib import Path
import sys

import pytest

from core.storage import RunPaths
from engine import IOSpec, OperatorSpec, TaskScheduler
from engine.execution import begin_execution, list_executions, mark_started
from runners import managed_execution as runner


POD = {"cluster": "local-test-cluster", "namespace": "test", "pod_uid": "unit-test-pod"}


@pytest.fixture
def scheduler(tmp_path):
    instance = TaskScheduler(tmp_path / "state.sqlite")
    for name in ("one", "two"):
        instance.create_run(
            run_id=name, model_id="demo", backend="kunlun", status="ENVIRONMENT_READY",
            environment={"environment_proof": {"resource_identity": POD, "fingerprint": "test-only"}},
            metadata={"evidence_mode": "simulation", "artifact_root": str(tmp_path / name)},
        )
    yield instance
    instance.store.close()


def graph_output(tmp_path, run_id="one"):
    return RunPaths(tmp_path / run_id, run_id).allocate_attempt("graph-local-probe").output


def claim(scheduler, run_id="one", *, lease_seconds=60):
    scheduler.discover_operator(run_id, OperatorSpec(
        operator_id="identity", model_id="demo", model_revision="v1",
        plugin_revision="v1", backend="kunlun",
        inputs=[IOSpec("x", "float32", [1], "contiguous")],
        outputs=[IOSpec("y", "float32", [1], "contiguous")],
        semantics={"operation": "identity"},
    ))
    task = scheduler.claim_ready("producer", run_id=run_id, lease_seconds=lease_seconds)[0]
    return task, {"task_id": task.task_id, "worker": "producer", "lease_token": task.lease_token}


def execute(scheduler, tmp_path, output, source="print('measured output')", **kwargs):
    return runner.execute_managed(
        scheduler, "one", "exec", [sys.executable, "-c", source],
        cwd=tmp_path, output_dir=output, **kwargs,
    )


def test_real_success_is_persisted_and_replay_does_not_spawn_twice(scheduler, tmp_path):
    root = graph_output(tmp_path)
    source = "from pathlib import Path; p=Path('counter'); p.write_text('once'); print('result')"
    result = execute(scheduler, tmp_path, root, source, resource=POD)
    assert result["state"] == "SUCCEEDED" and result["returncode"] == 0
    assert result["termination_confirmed"] is True
    assert result["pid"] > 0 and result["process_identity"] and result["host_id"]
    assert (root / "console.log").read_text().strip() == "result"
    before = list(scheduler.store.db.iterdump())
    (tmp_path / "counter").write_text("must not change")
    replay = execute(scheduler, tmp_path, root, source, resource=POD)
    assert replay == result
    assert (tmp_path / "counter").read_text() == "must not change"
    assert list(scheduler.store.db.iterdump()) == before


def test_real_failed_process_has_exit_code_and_logs(scheduler, tmp_path):
    root = graph_output(tmp_path)
    result = execute(scheduler, tmp_path, root, "import sys; print('original failure'); sys.exit(7)")
    assert result["state"] == "FAILED" and result["returncode"] == 7
    assert result["termination_confirmed"] is True
    assert "original failure" in (root / "console.log").read_text()
    assert not list_executions(scheduler.store, active_only=True)


def test_real_pod_resource_blocks_before_local_transport_spawn(scheduler, tmp_path, monkeypatch):
    root = graph_output(tmp_path)
    original_run = scheduler.store.run
    def real_run(run_id):
        run = original_run(run_id)
        run.metadata.update(evidence_mode="real", worker_protocol="managed-v2")
        return run
    monkeypatch.setattr(scheduler.store, "run", real_run)
    def forbidden(*args, **kwargs):
        pytest.fail("a local transport cannot prove remote Pod termination")
    monkeypatch.setattr(runner.subprocess, "Popen", forbidden)
    with pytest.raises(ValueError, match="trusted remote process"):
        execute(scheduler, tmp_path, root, resource=POD)
    assert not (root / "console.log").exists()
    assert not original_run("one").metadata.get("execution_records")


def test_timeout_stops_known_owned_child_and_releases_resource(scheduler, tmp_path):
    root = graph_output(tmp_path)
    result = execute(scheduler, tmp_path, root, "import time; time.sleep(60)",
                     timeout=0.2, resource=POD)
    assert result["state"] == "FAILED" and result["termination_confirmed"] is True
    assert "TimeoutError" in result["error"] and result["returncode"] < 0
    assert runner.process_identity(result["pid"]) is None
    assert not runner._group_alive(result["pid"])


def test_silent_task_renews_initially_and_while_running(scheduler, tmp_path):
    task, credentials = claim(scheduler, lease_seconds=0.2)
    root = Path(task.input["workspace"]["output"]) / "silent"
    original_expiry = task.lease_expires
    result = execute(scheduler, tmp_path, root, "import time; time.sleep(1.3)", **credentials)
    assert result["state"] == "SUCCEEDED"
    assert (root / "console.log").read_text() == ""
    events = scheduler.store.events("one")
    renewals = [event for event in events if event.event_type == "lease_renewed"]
    assert len(renewals) >= 2
    assert renewals[0].payload["lease_expires"] > original_expiry
    names = [event.event_type for event in events]
    assert names.index("lease_renewed") < names.index("execution_started")
    assert names.count("execution_heartbeat") >= 2


def test_interrupt_cancels_only_known_owned_process(scheduler, tmp_path, monkeypatch):
    root = graph_output(tmp_path)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("test cancellation")

    monkeypatch.setattr(runner, "heartbeat", interrupted)
    result = execute(scheduler, tmp_path, root, "import time; time.sleep(60)", resource=POD)
    assert result["state"] == "FAILED" and result["termination_confirmed"] is True
    assert "KeyboardInterrupt" in result["error"]
    assert runner.process_identity(result["pid"]) is None


def test_ambiguous_spawn_interrupt_stays_unknown(scheduler, tmp_path, monkeypatch):
    root = graph_output(tmp_path)
    monkeypatch.setattr(runner, "host_identity", lambda: "local-host")

    def interrupt_before_handle(*args, **kwargs):
        raise KeyboardInterrupt("spawn interrupted before handle returned")

    monkeypatch.setattr(runner.subprocess, "Popen", interrupt_before_handle)
    result = execute(scheduler, tmp_path, root, resource=POD)
    assert result["state"] == "UNKNOWN" and result["termination_confirmed"] is False
    assert result["pid"] is None
    assert len(list_executions(scheduler.store, active_only=True)) == 1


def test_lease_lost_before_spawn_finishes_reservation_without_starting_process(
    scheduler, tmp_path, monkeypatch,
):
    task, credentials = claim(scheduler)
    root = Path(task.input["workspace"]["output"]) / "renew-failure"

    def reject_renewal(*args, **kwargs):
        raise ValueError("lease lost before spawn")

    def forbidden_spawn(*args, **kwargs):
        pytest.fail("process must not start after renewal failed")

    monkeypatch.setattr(scheduler, "renew_lease", reject_renewal)
    monkeypatch.setattr(runner.subprocess, "Popen", forbidden_spawn)
    result = execute(scheduler, tmp_path, root, **credentials)
    assert result["state"] == "FAILED" and result["termination_confirmed"] is True
    assert result["pid"] is None and result["returncode"] is None


@pytest.mark.parametrize("case", ["outside_run", "other_run", "input_dir", "older_attempt"])
def test_output_scope_rejected_before_reservation_or_process(scheduler, tmp_path, case):
    task, credentials = claim(scheduler)
    root = {
        "outside_run": tmp_path / "unmanaged",
        "other_run": graph_output(tmp_path, "two"),
        "input_dir": Path(task.input["workspace"]["input"]),
        "older_attempt": RunPaths(tmp_path / "one", "one").allocate_attempt(task.task_id).output,
    }[case]
    before = list(scheduler.store.db.iterdump())
    with pytest.raises(ValueError, match="execution output"):
        execute(scheduler, tmp_path, root, **credentials)
    assert list(scheduler.store.db.iterdump()) == before
    assert not list_executions(scheduler.store)
    assert not (root / "console.log").exists()


def test_wrong_run_task_or_pod_identity_does_not_launch(scheduler, tmp_path):
    root = graph_output(tmp_path)
    before = list(scheduler.store.db.iterdump())
    with pytest.raises(ValueError, match="environment proof"):
        execute(scheduler, tmp_path, root, resource={**POD, "pod_uid": "another-pod"})
    assert list(scheduler.store.db.iterdump()) == before
    task, credentials = claim(scheduler, "two")
    with pytest.raises(ValueError, match="task does not belong"):
        execute(scheduler, tmp_path, root, **credentials)
    assert not list_executions(scheduler.store)


def test_resource_cannot_be_used_after_environment_becomes_unready(scheduler, tmp_path):
    root = graph_output(tmp_path)
    scheduler.record_environment_failure("one", {}, "unit fixture readiness lost")
    before = list(scheduler.store.db.iterdump())
    with pytest.raises(ValueError, match="currently ready"):
        execute(scheduler, tmp_path, root, resource=POD)
    assert list(scheduler.store.db.iterdump()) == before


def test_controller_lease_is_not_forwarded_to_candidate_environment(scheduler, tmp_path):
    task, credentials = claim(scheduler)
    root = Path(task.input["workspace"]["output"]) / "token-check"
    with pytest.raises(ValueError, match="lease token"):
        execute(scheduler, tmp_path, root, env_overrides={"TOKEN": task.lease_token}, **credentials)
    assert not list_executions(scheduler.store)


def test_remote_mode_is_rejected_without_reserving_or_spawning(scheduler, tmp_path):
    root = graph_output(tmp_path)
    before = list(scheduler.store.db.iterdump())
    with pytest.raises(ValueError, match="trusted remote"):
        execute(scheduler, tmp_path, root, remote=True, resource=POD)
    assert list(scheduler.store.db.iterdump()) == before
    assert not (root / "console.log").exists()


@pytest.mark.parametrize("host,present,group,remote,released", [
    ("same-host", False, False, False, True),
    ("other-host", False, False, False, False),
    ("same-host", True, False, False, False),
    ("same-host", False, True, False, False),
    ("same-host", False, False, True, False),
])
def test_reconcile_requires_same_host_and_absent_process_group(
    scheduler, tmp_path, monkeypatch, host, present, group, remote, released,
):
    root = graph_output(tmp_path)
    begin_execution(scheduler, "one", "exec", {
        "argv": ["test-observed-process"], "cwd": str(tmp_path), "output_dir": str(root),
        "remote": remote,
    }, resource=POD)
    mark_started(scheduler, "one", "exec", pid=12345, process_identity="original-birth",
                 host_id=host, remote_handle={"job": "remote"} if remote else None)
    monkeypatch.setattr(runner, "host_identity", lambda: "same-host")
    monkeypatch.setattr(runner, "process_identity", lambda _: "current-birth" if present else None)
    monkeypatch.setattr(runner, "_group_alive", lambda _: group)
    restarted = TaskScheduler(tmp_path / "state.sqlite")
    try:
        result = runner.reconcile_local_execution(restarted, "one", "exec")
    finally:
        restarted.store.close()
    assert result["termination_confirmed"] is released
    assert result["state"] == ("FAILED" if released else "UNKNOWN")
    assert bool(list_executions(scheduler.store, active_only=True)) is not released


def test_unidentified_starting_window_cannot_be_reconciled_from_pid_absence(
    scheduler, tmp_path, monkeypatch,
):
    root = graph_output(tmp_path)
    begin_execution(scheduler, "one", "exec", {
        "argv": ["test-process"], "cwd": str(tmp_path), "output_dir": str(root),
    }, resource=POD)
    monkeypatch.setattr(runner, "host_identity", lambda: "host")
    result = runner.reconcile_local_execution(scheduler, "one", "exec")
    assert result["state"] == "UNKNOWN" and result["termination_confirmed"] is False


def test_pid_reuse_is_never_signalled(monkeypatch):
    class Process:
        pid = 123

        def poll(self):
            return None

    def forbidden_signal(*args, **kwargs):
        pytest.fail("must not signal a process with a different birth identity")

    monkeypatch.setattr(runner, "process_identity", lambda _: "new-process")
    monkeypatch.setattr(runner.os, "killpg", forbidden_signal)
    assert runner._terminate_owned(Process(), "old-process") is False
