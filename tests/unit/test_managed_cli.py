"""Production CLI boundaries for the explicitly versioned managed rollout."""
import json
from pathlib import Path
import sys

import pytest

from cli.adaptation import main
from engine import IOSpec, OperatorSpec, TaskScheduler
from engine.execution import begin_execution
from engine.progress import run_progress
from runners.managed_boundary import require_supported_runtime


def call(capsys, state, *args):
    status = main(["--state", str(state), *args])
    return status, json.loads(capsys.readouterr().out)


def test_create_protocol_cannot_relabel_existing_run(tmp_path, capsys):
    state = tmp_path / "state.db"
    common = ("create-run", "--run-id", "one", "--model", "model", "--backend", "xpu")
    code, first = call(capsys, state, *common)
    assert code == 0
    code, rejected = call(capsys, state, *common, "--worker-protocol", "managed-v2")
    assert code == 2 and "retroactively" in rejected["error"]
    code, same = call(capsys, state, *common)
    assert same["run"] == first["run"]


def test_managed_real_environment_blocks_before_runtime(tmp_path, capsys, monkeypatch):
    state = tmp_path / "state.db"
    code, _ = call(capsys, state, "create-run", "--run-id", "r", "--model", "m",
                   "--backend", "xpu", "--worker-protocol", "managed-v2")
    assert code == 0
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported managed real runtime must not start")
    monkeypatch.setattr("cli.adaptation.subprocess.run", forbidden)
    code, result = call(capsys, state, "environment", "--run-id", "r", "--user-id", "explicit-owner")
    assert code == 2 and "trusted remote process" in result["error"]


@pytest.mark.parametrize("command,extra", [
    ("execution-status", []), ("reconcile-execution", ["--execution-id", "e"]),
    ("reconcile-validation", ["--validation-id", "v"]),
])
def test_missing_database_is_not_created(tmp_path, capsys, command, extra):
    state = tmp_path / "missing.db"
    code, _ = call(capsys, state, command, "--run-id", "missing", *extra)
    assert code == 2 and not state.exists()


@pytest.mark.parametrize("resource", [{}, {"cluster": "", "namespace": "n", "pod_uid": "p"}])
def test_explicit_invalid_resource_never_becomes_unlocked_cpu_execution(tmp_path, capsys, resource):
    state = tmp_path / "state.db"
    scheduler = TaskScheduler(state)
    try:
        scheduler.create_run(run_id="r", model_id="m", metadata={"evidence_mode": "simulation",
                                                                "artifact_root": str(tmp_path / "run")})
        scheduler.discover_operator("r", OperatorSpec(
            operator_id="op", model_id="m", model_revision="1", plugin_revision="1", backend="xpu",
            inputs=[IOSpec("x", "float64", [1], "contiguous")],
            outputs=[IOSpec("y", "float64", [1], "contiguous")], semantics={"op": "identity"}))
        task = scheduler.claim_ready("worker", run_id="r")[0]
        resource_path = tmp_path / "resource.json"
        resource_path.write_text(json.dumps(resource))
        output = Path(task.input["workspace"]["logs"]) / "execution"
        code, result = call(capsys, state, "execute-worker", "--task-id", task.task_id,
                            "--worker", "worker", "--lease-token", task.lease_token,
                            "--execution-id", "e", "--cwd", str(tmp_path), "--output-dir", str(output),
                            "--resource", str(resource_path), "--", sys.executable, "-c", "print('forbidden')")
        assert code == 2 and "resource" in result["error"]
        assert not output.exists()
        assert not scheduler.store.run("r").metadata.get("execution_records")
    finally:
        scheduler.store.close()


def test_progress_does_not_recommend_reclaim_with_unresolved_execution(tmp_path):
    scheduler = TaskScheduler(tmp_path / "state.db")
    try:
        scheduler.create_run(run_id="r", model_id="m", metadata={"evidence_mode": "simulation"})
        begin_execution(scheduler, "r", "unknown", {
            "argv": ["not-run"], "cwd": str(tmp_path), "output_dir": str(tmp_path / "external")})
        before = [event.to_dict() for event in scheduler.store.events("r")]
        view = run_progress(scheduler.store, "r")
        assert view["progress"]["reason_code"] == "EXECUTION_UNCERTAIN"
        assert view["progress"]["next_action"]["action"] == "INSPECT_EXECUTION"
        assert before == [event.to_dict() for event in scheduler.store.events("r")]
    finally:
        scheduler.store.close()


def test_runtime_rollout_does_not_change_legacy_or_simulation_policy(tmp_path):
    scheduler = TaskScheduler(tmp_path / "state.db")
    try:
        for mode in ("real", "simulation"):
            for protocol in (None, "managed-v2"):
                run = scheduler.create_run(run_id=f"{mode}-{protocol}", model_id="m",
                                           metadata={"evidence_mode": mode, "worker_protocol": protocol})
                if mode == "real" and protocol == "managed-v2":
                    with pytest.raises(ValueError, match="BLOCKED"):
                        require_supported_runtime(run)
                else:
                    require_supported_runtime(run)
    finally:
        scheduler.store.close()
