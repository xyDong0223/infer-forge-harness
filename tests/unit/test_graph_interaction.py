"""Graph yields durable decisions without synchronous decider waits."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import cli.workflow.graph as graph_cli
from engine.interaction import accept_graph_decision, current_graph_handoff
from engine.scheduler import TaskScheduler
from runners import graph_runner
from tests.unit.test_graph_scheduler_bridge import proof


@pytest.fixture
def interactive(tmp_path, monkeypatch):
    state = tmp_path / "state.sqlite"
    root = tmp_path / "run"
    scheduler = TaskScheduler(state)
    scheduler.create_run(run_id="r", model_id="demo", metadata={
        "evidence_mode": "simulation", "artifact_root": str(root),
    })
    input_file = tmp_path / "upstream.json"
    input_file.write_text('{"observed": "source input"}')
    monkeypatch.setattr(graph_runner, "load_workflow", lambda _: [
        {"id": "environment", "task": "fixture", "on_success": "DELIVERED",
         "on_failure": "environment"},
    ])
    monkeypatch.setattr(graph_runner, "node_task_type", lambda _: "environment_proof")
    monkeypatch.setattr(graph_runner.skill_registry, "resolve_for_context",
                        lambda *_: {"id": "fixture", "verification": [], "tools": []})
    monkeypatch.setitem(graph_runner.NODES, "environment_proof", {
        "produces": "EnvironmentProof", "state_file": "status.json",
        "command": ["python3", "cli/deployment/proof.py", "--artifact-dir", "{artifacts}",
                    "--fixture-input", str(input_file)],
    })
    calls = []
    outcomes = ["READINESS_TIMEOUT"]

    def execute(command, *, log_path, **kwargs):
        calls.append(command)
        outcome = outcomes.pop(0) if outcomes else "READINESS_TIMEOUT"
        if isinstance(outcome, Exception):
            raise outcome
        out = Path(command[command.index("--artifact-dir") + 1])
        proof(out, state=outcome, user_id=command[command.index("--user-id") + 1])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("observed fixture runtime output")
        return SimpleNamespace(returncode=0 if outcome == "ENVIRONMENT_READY" else 1,
                               crash_log=None)

    monkeypatch.setattr(graph_runner.evidence, "run_logged", execute)
    argv = ["graph", "--subject", "demo", "--run-id", "r", "--scheduler-state", str(state),
            "--artifact-root", str(root), "--interaction-mode", "codex", "--execute",
            "--set", "port=8000", "--set", "user_id=resource-owner", "--watch-interval", "0",
            "--set", "proof_health_interval=1",
            "--json", "--recovery-budget", "3"]
    yield SimpleNamespace(argv=argv, scheduler=scheduler, outcomes=outcomes, calls=calls,
                          input_file=input_file)
    scheduler.store.close()


def invoke(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", argv)
    return graph_cli.main()


def replay_args(snapshot):
    values = {**snapshot, "execute": True, "resume": True, "json": True,
              "from_node": None, "until_node": None, "target": None,
              "operator_report": None, "shim_registry": None}
    values["set"] = list(snapshot["set"])
    for key in ("workflow", "artifact_root", "scheduler_state", "journal", "loop_state"):
        values[key] = Path(values[key])
    return SimpleNamespace(**values)


@pytest.mark.parametrize("extra", [["--auto-recover"], ["--decide-command", "no-execution"]])
def test_codex_rejects_competing_decider(interactive, monkeypatch, extra):
    assert invoke(monkeypatch, interactive.argv + extra) == 2
    assert not interactive.calls
    assert current_graph_handoff(interactive.scheduler, "r") is None


def test_codex_requires_scheduler(interactive, monkeypatch):
    argv = list(interactive.argv)
    index = argv.index("--scheduler-state")
    del argv[index:index + 2]
    assert invoke(monkeypatch, argv) == 2
    assert not interactive.calls


def test_failure_yields_and_headless_cannot_bypass_handoff(interactive, monkeypatch, capsys):
    assert invoke(monkeypatch, interactive.argv) == 4
    handoff = current_graph_handoff(interactive.scheduler, "r")
    assert handoff["state"] == "pending"
    assert handoff["remaining_budget"] == 3
    assert isinstance(handoff["request"]["context"]["_producer_task_types"], dict)
    assert handoff["request"]["available_actions"] == ["RETRY", "RETRY_WITH_PARAMS", "BLOCKED"]
    files = handoff["source"]["source_files"]
    assert any(path.endswith("task_memory_snapshot.json") for path in files)
    assert not any(path.endswith("task_memory.json") for path in files)
    assert any(path.endswith("node_console.log") for path in files)
    assert any(path.endswith("commands.json") for path in files)
    assert any(path.endswith("skill.json") for path in files)
    assert str(interactive.input_file) in files
    run = interactive.scheduler.store.run("r")
    assert run.environment["failed_environment_proof"]["pod"] == "test-pod"
    assert run.metadata["graph_execution_context"]["interaction_mode"] == "codex"
    summaries = [json.loads(line) for line in capsys.readouterr().out.splitlines()
                 if line.startswith('{"status":')]
    assert "--interaction-mode" in summaries[-1]["resume_command"]
    assert invoke(monkeypatch, interactive.argv + ["--interaction-mode", "headless"]) == 4
    assert len(interactive.calls) == 1
    assert current_graph_handoff(interactive.scheduler, "r") == handoff


def test_rebuilding_memory_cache_does_not_invalidate_pending_decision(interactive, monkeypatch):
    from engine.state import task_memory

    assert invoke(monkeypatch, interactive.argv) == 4
    handoff = current_graph_handoff(interactive.scheduler, "r")
    snapshot = next(Path(path) for path in handoff["source"]["source_files"]
                    if path.endswith("task_memory_snapshot.json"))
    manifest = json.loads((snapshot.parent.parent / "manifest.json").read_text())
    assert any(entry["path"] == "input/task_memory_snapshot.json" for entry in manifest["artifacts"])
    context = interactive.scheduler.store.run("r").metadata["graph_execution_context"]
    cache, journal = Path(context["loop_state"]), Path(context["journal"])
    original = cache.read_bytes()
    journal_before = journal.read_bytes()
    cache.unlink()
    task_memory.rebuild(cache, Path(context["workflow"]).stem, "demo", journal=journal, run_id="r")
    assert json.loads(cache.read_text()) == json.loads(original)
    assert journal.read_bytes() == journal_before
    _, execute = accept_graph_decision(
        interactive.scheduler, "r", handoff["handoff_id"], "rebuild-decision",
        handoff["source_version"], {"next_action": "RETRY", "diagnosis": "cache was rebuilt",
                                  "evidence_refs": [str(snapshot)]},
    )
    assert execute


def test_pending_legacy_memory_binding_returns_before_projection_migration(interactive, monkeypatch):
    import engine.interaction as interaction

    original_create = interaction.create_graph_handoff

    def legacy_handoff(scheduler, run_id, source, request):
        context = scheduler.store.run(run_id).metadata["graph_execution_context"]
        cache = Path(context["loop_state"])
        source["source_files"] = {
            path: digest for path, digest in source["source_files"].items()
            if not path.endswith("task_memory_snapshot.json")
        }
        source["source_files"][str(cache)] = graph_runner.file_digest(cache)
        return original_create(scheduler, run_id, source, request)

    monkeypatch.setattr(interaction, "create_graph_handoff", legacy_handoff)
    assert invoke(monkeypatch, interactive.argv) == 4
    handoff = current_graph_handoff(interactive.scheduler, "r")
    cache = Path(next(path for path in handoff["source"]["source_files"]
                      if path.endswith("task_memory.json")))
    before = cache.read_bytes()
    monkeypatch.setattr(graph_runner, "load_task_memory",
                        lambda _: pytest.fail("pending legacy handoff must not migrate memory"))
    assert invoke(monkeypatch, interactive.argv) == 4
    assert cache.read_bytes() == before
    assert current_graph_handoff(interactive.scheduler, "r") == handoff


def test_missing_user_input_is_not_retry_handoff(interactive, monkeypatch):
    interactive.outcomes[:] = ["INPUT_REQUIRED"]
    assert invoke(monkeypatch, interactive.argv) == 2
    assert current_graph_handoff(interactive.scheduler, "r") is None


def test_process_start_failure_has_original_error_evidence(interactive, monkeypatch):
    interactive.outcomes[:] = [OSError("fixture executor unavailable")]
    assert invoke(monkeypatch, interactive.argv) == 4
    handoff = current_graph_handoff(interactive.scheduler, "r")
    assert handoff["request"]["failure"]["reason"] == "fixture executor unavailable"
    assert any(path.endswith("execution_error.json") for path in handoff["source"]["source_files"])


def test_replaced_upstream_command_input_rejects_old_decision(interactive, monkeypatch):
    assert invoke(monkeypatch, interactive.argv) == 4
    handoff = current_graph_handoff(interactive.scheduler, "r")
    interactive.input_file.write_text('{"observed": "different source"}')
    with pytest.raises(ValueError, match="source evidence changed"):
        accept_graph_decision(
            interactive.scheduler, "r", handoff["handoff_id"], "decision-one",
            handoff["source_version"], {
                "next_action": "RETRY", "diagnosis": "readiness may be transient",
                "evidence_refs": [str(interactive.input_file)],
            },
        )
    assert len(interactive.calls) == 1


def test_task_definition_change_rejects_pending_decision(interactive, monkeypatch, tmp_path):
    task_file = tmp_path / "task.yaml"
    task_file.write_text("versioned Task execution definition")
    monkeypatch.setitem(graph_runner.NODES, "environment_proof", {
        **graph_runner.NODES["environment_proof"], "task_path": str(task_file),
        "task_sha256": graph_runner.file_digest(task_file),
    })
    assert invoke(monkeypatch, interactive.argv) == 4
    handoff = current_graph_handoff(interactive.scheduler, "r")
    assert str(task_file) in handoff["source"]["source_files"]
    assert any(path.endswith("task_execution.json") for path in handoff["source"]["source_files"])
    task_file.write_text("a changed execution definition")
    with pytest.raises(ValueError, match="source evidence changed"):
        accept_graph_decision(
            interactive.scheduler, "r", handoff["handoff_id"], "task-change-decision",
            handoff["source_version"], {"next_action": "RETRY", "diagnosis": "inspect Task change",
                                      "evidence_refs": [str(task_file)]},
        )
    assert current_graph_handoff(interactive.scheduler, "r") == handoff
    assert len(interactive.calls) == 1


def test_execution_helper_rejects_undeclared_retry_effects_before_runtime(interactive, monkeypatch):
    assert invoke(monkeypatch, interactive.argv) == 4
    scheduler = interactive.scheduler
    handoff = current_graph_handoff(scheduler, "r")
    args = replay_args(scheduler.store.run("r").metadata["graph_execution_context"])
    with pytest.raises(ValueError):
        graph_runner.execute_graph_decision(args, handoff, {
            "next_action": "RETRY_WITH_PARAMS", "diagnosis": "change another declared scalar",
            "params": {"port": 9000},
        }, scheduler)
    assert len(interactive.calls) == 1
    assert scheduler.store.run("r").metadata["graph_execution_context"]["resolved_context"]["port"] == "8000"


@pytest.mark.parametrize("recovered", [False, True])
def test_submitted_retry_is_one_shot_and_persists_parameters(
    interactive, monkeypatch, recovered,
):
    assert invoke(monkeypatch, interactive.argv) == 4
    scheduler = interactive.scheduler
    handoff = current_graph_handoff(scheduler, "r")
    decision = {"next_action": "RETRY_WITH_PARAMS", "diagnosis": "adjust measured health polling interval",
                "params": {"proof_health_interval": 0.5}, "confidence": 0.8,
                "evidence_refs": [next(iter(handoff["source"]["source_files"]))]}
    receipt, execute = accept_graph_decision(
        scheduler, "r", handoff["handoff_id"], "decision-one", handoff["source_version"], decision,
    )
    assert execute
    interactive.outcomes[:] = ["ENVIRONMENT_READY" if recovered else "READINESS_TIMEOUT"]
    args = replay_args(scheduler.store.run("r").metadata["graph_execution_context"])
    outcome = graph_runner.execute_graph_decision(args, handoff, decision, scheduler)
    assert len(interactive.calls) == 2
    assert outcome["status"] == ("RECOVERED" if recovered else "REWORK")
    assert outcome["final_artifacts"] != handoff["source"]["artifacts"]
    assert Path(outcome["final_artifacts"]).is_dir()
    assert outcome["retry_context"]["pod"] == "test-pod"
    assert outcome["retry_context"]["port"] == "8000"
    assert outcome["retry_context"]["proof_health_interval"] == "0.5"
    command = interactive.calls[-1]
    assert command[command.index("--health-interval-seconds") + 1] == "0.5"
    saved = scheduler.store.run("r").metadata["graph_execution_context"]
    assert saved["set"][-1] == "proof_health_interval=0.5"
    assert saved["resolved_context"]["proof_health_interval"] == "0.5"
    assert receipt["remaining_budget"] == 2
    if recovered:
        assert scheduler.store.run("r").status == "ENVIRONMENT_READY"
