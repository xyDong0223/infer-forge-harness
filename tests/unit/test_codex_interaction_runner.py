"""The foreground controller reconstructs inputs and never repeats accepted work."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from core.paths import REPO_ROOT
from core.storage import ArtifactStore, RunPaths
from engine.brain import DecisionRequest, FailureEvidence
from engine import IOSpec, OperatorSpec
from engine.interaction import canonical_digest, create_graph_handoff
from engine.run_control import control_run
from engine.scheduler import TaskScheduler
from runners.codex_interaction import advance_run, graph_arguments, submit_graph_decision


@pytest.fixture
def scheduler(tmp_path):
    scheduler = TaskScheduler(tmp_path / "state.sqlite")
    run_root = tmp_path / "run"
    scheduler.create_run(run_id="r", model_id="model", metadata={
        "evidence_mode": "simulation", "artifact_root": str(run_root),
    })
    workflow = REPO_ROOT / "workflows/model_adaptation.yaml"
    scheduler.record_graph_transition("r", "graph_execution_context", {
        "schema_version": 1, "subject": "model", "run_id": "r",
        "workflow": str(workflow), "workflow_sha256": hashlib.sha256(workflow.read_bytes()).hexdigest(),
        "artifact_root": str(run_root), "scheduler_state": scheduler.store.path,
        "journal": str(run_root / "journal.jsonl"), "loop_state": str(run_root / "task_memory.json"),
        "env": ["hardware=P800"], "set": ["port=8000", "port=8001"],
        "auto_recover": False, "recovery_budget": 2, "watch_interval": 0,
    })
    yield scheduler
    scheduler.store.close()


def handoff(scheduler):
    run = scheduler.store.run("r")
    attempt = RunPaths(run.metadata["artifact_root"], "r").allocate_attempt("service")
    status = ArtifactStore(attempt.output).write_json("status.json", {"state": "FAILED"})
    source = {
        "node": "service", "task_type": "service_proof", "attempt_id": attempt.identity["attempt_id"],
        "artifacts": str(attempt.output),
        "source_files": {str(status): hashlib.sha256(status.read_bytes()).hexdigest()},
        "execution_context_sha256": canonical_digest(run.metadata["graph_execution_context"]),
        "environment_sha256": canonical_digest(run.environment),
    }
    item = create_graph_handoff(scheduler, "r", source, DecisionRequest(
        model="model", backend="xpu", failure=FailureEvidence("service", "FAILED", "timeout"),
        attempts_remaining=2, available_actions=["RETRY", "BLOCKED"], context={},
    ))
    decision = {"next_action": "RETRY", "diagnosis": "observed timeout", "confidence": 0.8,
                "evidence_refs": [str(status)]}
    return item, decision


def submit(scheduler, item, decision):
    return submit_graph_decision(scheduler, "r", item["handoff_id"], "decision-1",
                                 item["source_version"], decision)


def test_reconstructs_typed_inputs_and_explicit_settings(scheduler):
    args = graph_arguments(scheduler, "r", ["port=8002"])
    assert args.set == ["port=8000", "port=8001", "port=8002"]
    assert args.resume and args.execute and args.interaction_mode == "codex"
    assert not args.auto_recover and args.from_node is None
    assert args.workflow == REPO_ROOT / "workflows/model_adaptation.yaml"
    assert args.watch_interval == 0


@pytest.mark.parametrize("change", [
    {"subject": "foreign"}, {"run_id": "foreign"}, {"schema_version": 2},
    {"workflow_sha256": "changed"}, {"auto_recover": True}, {"decide_command": "external-agent"},
    {"set": [False]}, {"env": {}}, {"watch_interval": float("inf")}, {"recovery_budget": True},
])
def test_invalid_saved_configuration_is_rejected_before_execution(scheduler, change):
    saved = scheduler.store.run("r").metadata["graph_execution_context"]
    scheduler.record_graph_transition("r", "graph_execution_context", {**saved, **change})
    before = list(scheduler.store.db.iterdump())
    with pytest.raises(ValueError):
        advance_run(scheduler, "r")
    assert list(scheduler.store.db.iterdump()) == before


def test_pending_handoff_advance_does_not_allocate_or_wait(scheduler, monkeypatch):
    from runners import graph_runner
    item, _ = handoff(scheduler)
    before = list(scheduler.store.db.iterdump())
    monkeypatch.setattr(graph_runner, "run", lambda args: pytest.fail("must not execute"))
    result = advance_run(scheduler, "r")
    assert result["exit_code"] == 4 and result["handoff"] == item
    assert result["progress"]["reason_code"] == "GRAPH_DECISION_REQUIRED"
    assert list(scheduler.store.db.iterdump()) == before
    with pytest.raises(ValueError, match="bypassed"):
        advance_run(scheduler, "r", settings=["port=8003"])


def test_worker_boundary_does_not_silently_discard_explicit_settings(scheduler):
    task = scheduler.discover_operator("r", OperatorSpec(
        operator_id="identity", model_id="model", model_revision="1", plugin_revision="1",
        backend="xpu", inputs=[IOSpec("x", "float32", [1], "contiguous")],
        outputs=[IOSpec("y", "float32", [1], "contiguous")], semantics={"op": "identity"},
    ))
    scheduler.record_graph_transition("r", "graph_progress", {
        "reason_code": "GRAPH_RECOVERY_SUCCEEDED", "status": "READY", "node": "service",
    })
    before = list(scheduler.store.db.iterdump())
    with pytest.raises(ValueError, match="pending worker"):
        advance_run(scheduler, "r", settings=["port=8003"])
    result = advance_run(scheduler, "r")
    assert result["exit_code"] == 3
    assert result["progress"]["reason_code"] == "WORKER_UNCLAIMED"
    assert result["progress"]["location"] == task.task_id
    assert list(scheduler.store.db.iterdump()) == before


def test_submission_is_durable_before_execution_and_replay_never_reruns(scheduler, monkeypatch, tmp_path):
    from runners import graph_runner
    item, decision = handoff(scheduler)
    calls = []

    def execute(args, current, candidate, owner):
        saved = owner.store.run("r").metadata["graph_decisions"]["decision-1"]
        assert saved["execution_status"] == "IN_PROGRESS_OR_UNKNOWN"
        calls.append(current["handoff_id"])
        return {"status": "RECOVERED", "node": "service", "state": "READY",
                "final_artifacts": str(tmp_path / "unit-only-candidate")}

    monkeypatch.setattr(graph_runner, "execute_graph_decision", execute)
    first = submit(scheduler, item, decision)
    assert first["receipt"]["execution_status"] == "FINISHED"
    assert first["progress"]["reason_code"] == "GRAPH_RECOVERY_SUCCEEDED"
    assert calls == [item["handoff_id"]]
    before = list(scheduler.store.db.iterdump())
    other = TaskScheduler(scheduler.store.path)
    try:
        replay = submit(other, item, decision)
    finally:
        other.store.close()
    assert replay["replayed"] and replay["receipt"] == first["receipt"]
    assert calls == [item["handoff_id"]]
    assert list(scheduler.store.db.iterdump()) == before


def test_unexpected_execution_failure_stays_uncertain_across_restart(scheduler, monkeypatch):
    from runners import graph_runner
    item, decision = handoff(scheduler)

    def crash(*args):
        raise RuntimeError("external result lost")

    monkeypatch.setattr(graph_runner, "execute_graph_decision", crash)
    first = submit(scheduler, item, decision)
    assert first["exit_code"] == 2
    assert first["receipt"]["state"] == "executing"
    assert "external result lost" in first["execution_error"]
    monkeypatch.setattr(graph_runner, "execute_graph_decision", lambda *args: pytest.fail("must not rerun"))
    replay = submit(scheduler, item, decision)
    assert replay["replayed"] and replay["exit_code"] == 2
    assert advance_run(scheduler, "r")["progress"]["reason_code"] == "GRAPH_EXECUTION_UNCERTAIN"


def test_explicit_block_never_invokes_runtime(scheduler, monkeypatch):
    from runners import graph_runner
    item, decision = handoff(scheduler)
    decision["next_action"] = "BLOCKED"
    monkeypatch.setattr(graph_runner, "execute_graph_decision", lambda *args: pytest.fail("must not run"))
    result = submit(scheduler, item, decision)
    assert result["exit_code"] == 2
    assert result["receipt"]["remaining_budget"] == 2
    assert result["handoff"]["state"] == "blocked"


def test_run_control_is_reentrant_but_rejects_another_thread(scheduler):
    def concurrent():
        with pytest.raises(ValueError, match="controller"):
            with control_run(scheduler.store.path, "r"):
                pytest.fail("two controllers acquired one run")

    with control_run(scheduler.store.path, "r"):
        with control_run(scheduler.store.path, "r"):
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(concurrent).result()
    with control_run(scheduler.store.path, "r"):
        pass  # Normal exit released the OS lock.


@pytest.mark.parametrize("command", ["advance", "submit-decision"])
def test_control_cli_does_not_create_missing_database(tmp_path, command):
    state = tmp_path / "missing" / "state.sqlite"
    extra = [] if command == "advance" else [
        "--handoff-id", "h", "--decision-id", "d", "--expected-version", "v",
        "--decision", str(tmp_path / "decision.json"),
    ]
    result = subprocess.run([sys.executable, str(REPO_ROOT / "cli/adaptation.py"),
                             "--state", str(state), command, "--run-id", "r", *extra],
                            text=True, capture_output=True, check=False)
    assert result.returncode == 2
    assert json.loads(result.stdout)["progress"]["reason_code"] == "COMMAND_REJECTED"
    assert not state.parent.exists()
