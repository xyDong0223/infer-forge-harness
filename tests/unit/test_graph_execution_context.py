"""Restart configuration is durable before execution and never written by plans."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import pytest
import yaml

import cli.workflow.graph as graph_cli
from core.paths import REPO_ROOT
from engine.scheduler import EventStore, TaskScheduler
from runners import graph_runner


ENVIRONMENT_NODE = "kdp-001a-environment-proof"
WORKFLOW = REPO_ROOT / "workflows" / "model_adaptation.yaml"


def read_run(state):
    store = EventStore(state, readonly=True)
    try:
        return store.run("r"), store.events("r")
    finally:
        store.close()


@pytest.fixture
def invocation(tmp_path, monkeypatch):
    state = tmp_path / "state.sqlite"
    scheduler = TaskScheduler(state)
    scheduler.create_run(
        run_id="r", model_id="demo", model_revision="model-v1",
        plugin_revision="plugin-v1", backend="kunlun",
        metadata={"evidence_mode": "simulation", "artifact_root": str(tmp_path / "run")},
    )
    scheduler.store.close()
    target = {
        "target": {
            "model": "demo", "hardware": "kunlun/p800",
            "runtime": {"engine": "vllm", "backend": "kunlun", "plugin": "vllm-kunlun"},
        },
    }
    (tmp_path / "target.yaml").write_text(yaml.safe_dump(target))
    (tmp_path / "operators.json").write_text('{"entries": []}')
    (tmp_path / "shims.json").write_text('{"entries": []}')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HARNESS_TEST_SECRET", "not-a-graph-input")
    argv = [
        "graph", "--subject", "demo", "--run-id", "r",
        "--scheduler-state", "state.sqlite", "--artifact-root", "run",
        "--journal", "coordination/journal.jsonl", "--loop-state", "coordination/memory.json",
        "--workflow", os.path.relpath(WORKFLOW, tmp_path),
        "--target", "target.yaml", "--operator-report", "operators.json",
        "--shim-registry", "shims.json",
        "--env", "tag=first", "--env", "tag=last",
        "--set", "user_id=test-owner", "--set", "model_path=/mounted/demo",
        "--set", "port=8010", "--set", "port=8020",
        "--watch-interval", "0", "--auto-recover", "--brain", "rule",
        "--recovery-budget", "4", "--decide-command", "local-decider {request} {response}",
        "--decide-timeout", "1.5", "--until-node", ENVIRONMENT_NODE, "--json",
    ]
    observed = []

    def unavailable_external_executor(*args, **kwargs):
        # No report is fabricated: the first runtime invocation fails. The
        # scheduler and unchanged production graph run through their real path.
        run, _ = read_run(state)
        observed.append(run.metadata.get("graph_execution_context"))
        raise OSError("simulated unavailable external executor")

    monkeypatch.setattr(graph_runner.evidence, "run_logged", unavailable_external_executor)
    return argv, state, observed


def invoke(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", argv)
    return graph_cli.main()


def test_validated_context_is_saved_before_first_runtime_invocation(
    invocation, tmp_path, monkeypatch, capsys,
):
    argv, state, observed = invocation
    assert invoke(monkeypatch, argv + ["--execute", "--from-node", ENVIRONMENT_NODE]) == 2
    run, events = read_run(state)
    snapshot = run.metadata["graph_execution_context"]
    assert observed == [snapshot]
    assert snapshot["schema_version"] == 1
    assert snapshot["workflow"] == str(WORKFLOW)
    assert snapshot["workflow_sha256"] == hashlib.sha256(WORKFLOW.read_bytes()).hexdigest()
    assert snapshot["subject"] == "demo"
    assert snapshot["run_id"] == "r"
    for key, path in {
        "artifact_root": "run", "scheduler_state": "state.sqlite",
        "journal": "coordination/journal.jsonl", "loop_state": "coordination/memory.json",
        "target": "target.yaml", "operator_report": "operators.json", "shim_registry": "shims.json",
    }.items():
        assert snapshot[key] == str(tmp_path / path)
    assert snapshot["env"] == ["tag=first", "tag=last"]
    assert snapshot["set"] == [
        "user_id=test-owner", "model_path=/mounted/demo", "port=8010", "port=8020",
    ]
    assert snapshot["resolved_environment"]["tag"] == "last"
    assert snapshot["resolved_environment"]["hardware"] == "kunlun/p800"
    assert snapshot["resolved_environment"]["evidence_mode"] == "simulation"
    assert snapshot["resolved_context"]["port"] == "8020"
    assert snapshot["auto_recover"] is True
    assert snapshot["brain"] == "rule"
    assert snapshot["recovery_budget"] == 4
    assert snapshot["decide_command"] == "local-decider {request} {response}"
    assert snapshot["decide_timeout"] == 1.5
    assert snapshot["watch_interval"] == 0
    assert not set(snapshot).intersection({"from_node", "until_node", "execute", "resume"})
    assert not set(snapshot["resolved_context"]).intersection({"artifacts", "attempt"})
    assert "HARNESS_TEST_SECRET" not in json.dumps(snapshot)
    assert "not-a-graph-input" not in json.dumps(snapshot)
    assert [event.event_type for event in events].count("graph_execution_context") == 1

    summaries = [json.loads(line) for line in capsys.readouterr().out.splitlines()
                 if line.startswith('{"status":')]
    replay = summaries[-1]["resume_command"]
    assert replay[replay.index("--watch-interval") + 1] == "0.0"
    assert "--from-node" not in replay
    assert "--until-node" not in replay


def test_restart_keeps_inputs_and_uses_a_fresh_attempt(invocation, tmp_path, monkeypatch):
    argv, state, observed = invocation
    assert invoke(monkeypatch, argv + ["--execute"]) == 2
    original, _ = read_run(state)
    assert invoke(monkeypatch, argv + ["--execute", "--resume"]) == 2
    restarted, events = read_run(state)
    assert restarted.metadata["graph_execution_context"] == original.metadata["graph_execution_context"]
    assert len(observed) == 2
    assert observed[0] == observed[1]
    assert len(list((tmp_path / "run" / "tasks" / ENVIRONMENT_NODE / "attempts").iterdir())) == 2
    assert [event.event_type for event in events].count("graph_execution_context") == 1


@pytest.mark.parametrize("after_execute", [False, True])
def test_plan_preserves_scheduler_and_saved_execution_context(invocation, monkeypatch, after_execute):
    argv, state, observed = invocation
    if after_execute:
        assert invoke(monkeypatch, argv + ["--execute"]) == 2
    before, events_before = read_run(state)
    assert invoke(monkeypatch, argv + ["--set", "port=9000"]) == 0
    after, events_after = read_run(state)
    assert after.to_dict() == before.to_dict()
    assert events_after == events_before
    assert len(observed) == int(after_execute)


@pytest.mark.parametrize("rejected", [
    ["--subject", "another-model"],
    ["--run-id", "unknown-run"],
    ["--artifact-root", "another-run"],
    ["--env", "backend=cuda"],
    ["--env", "model_revision=wrong-revision"],
    ["--set", "run_id=another-run"],
    ["--set", "malformed"],
    ["--workflow", "missing.yaml"],
    ["--from-node", "missing-node"],
    ["--until-node", "missing-node"],
])
def test_rejected_restart_does_not_replace_last_valid_context(invocation, monkeypatch, rejected):
    argv, state, observed = invocation
    assert invoke(monkeypatch, argv + ["--execute"]) == 2
    before, events_before = read_run(state)
    assert invoke(monkeypatch, argv + ["--execute", *rejected]) == 2
    after, events_after = read_run(state)
    assert after.metadata["graph_execution_context"] == before.metadata["graph_execution_context"]
    assert len(observed) == 1
    assert [event for event in events_after if event.event_type == "graph_execution_context"] == [
        event for event in events_before if event.event_type == "graph_execution_context"
    ]


@pytest.mark.parametrize("after_execute", [False, True])
def test_unresolved_node_input_does_not_publish_restart_context(
    invocation, monkeypatch, capsys, after_execute,
):
    argv, state, observed = invocation
    if after_execute:
        assert invoke(monkeypatch, argv + ["--execute"]) == 2
    before, events_before = read_run(state)
    invalid = list(argv)
    model_path_index = invalid.index("model_path=/mounted/demo")
    del invalid[model_path_index - 1:model_path_index + 1]
    intake = "mat-001-model-intake"
    assert invoke(monkeypatch, invalid + [
        "--execute", "--from-node", intake, "--until-node", intake,
    ]) == 2
    after, events_after = read_run(state)
    assert after.metadata.get("graph_execution_context") == before.metadata.get("graph_execution_context")
    assert len(observed) == int(after_execute)
    assert [event for event in events_after if event.event_type == "graph_execution_context"] == [
        event for event in events_before if event.event_type == "graph_execution_context"
    ]
    summaries = [json.loads(line) for line in capsys.readouterr().out.splitlines()
                 if line.startswith('{"status":')]
    assert summaries[-1]["reason_code"] == "INPUT_UNRESOLVED"
    assert "model_path" in summaries[-1]["message"]
