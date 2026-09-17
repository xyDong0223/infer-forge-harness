"""Graph execution, provenance and state gates consume the Task definition."""

import json

import pytest

from core.storage import RunPaths
from core.task_execution import load_execution_catalog
from engine.graph_bridge import DELIVERY_FACTS, FACT_STATUS
from engine.state import journal
from runners import graph_runner as graph


def test_graph_and_journal_derive_execution_metadata():
    catalog = load_execution_catalog()
    assert set(graph.NODES) == set(catalog)
    for task_type, task in catalog.items():
        expected = task.to_node_spec()
        assert {key: graph.NODES[task_type][key] for key in expected} == expected
        assert journal.KINDS[task_type] == task.produces
    assert FACT_STATUS == {task.produces: task.status_file for task in catalog.values()
                           if task.produces in DELIVERY_FACTS}


def test_other_tasks_success_state_is_not_intake_success(tmp_path):
    spec = graph.NODES["model_intake"]
    status = tmp_path / spec["state_file"]
    status.write_text(json.dumps({"state": "ENVIRONMENT_READY"}))
    assert "ENVIRONMENT_READY" in graph.SUCCESS_STATES
    assert not graph.node_passed(tmp_path, spec, 0)
    facts = tmp_path / "journal.jsonl"
    graph.record_fact(facts, spec, "demo", tmp_path, {"hardware": "cpu"})
    assert graph.reusable_fact(spec, "demo", facts, {"hardware": "cpu"}) is None


def test_task_change_invalidates_new_fact_but_legacy_fact_remains_readable(tmp_path):
    spec = graph.NODES["model_intake"]
    (tmp_path / spec["state_file"]).write_text(json.dumps({"state": "INTAKE_READY"}))
    facts, environment = tmp_path / "journal.jsonl", {"hardware": "cpu"}
    graph.record_fact(facts, spec, "demo", tmp_path, environment)
    assert graph.reusable_fact(spec, "demo", facts, environment)
    changed = {**spec, "task_sha256": "f" * 64}
    assert graph.reusable_fact(changed, "demo", facts, environment) is None
    journal.record(facts, spec["produces"], "demo", "INTAKE_READY", tmp_path, environment)
    assert graph.reusable_fact(changed, "demo", facts, environment)


def test_execution_snapshot_binds_task_and_refuses_changed_source(tmp_path):
    spec = graph.NODES["model_intake"]
    paths, planned = RunPaths(tmp_path / "run", "r"), tmp_path / "planned"
    attempt, _ = graph.allocate_commands(paths, "intake", planned, [(planned, ["fixture"])],
                                         task_spec=spec)
    packet = json.loads((attempt.input / "task_execution.json").read_text())
    assert packet["task_contract"] == graph.task_binding(spec)
    assert packet["descriptor"]["command"] == spec["command"]
    with pytest.raises(ValueError, match="changed before execution"):
        graph.allocate_commands(paths, "intake", planned, [(planned, ["fixture"])],
                                task_spec={**spec, "task_sha256": "f" * 64})


def test_workflow_cannot_substitute_same_type_different_contract(tmp_path):
    from core.paths import REPO_ROOT

    source = graph.NODES["model_intake"]["task_path"]
    duplicate = tmp_path / "task.yaml"
    duplicate.write_bytes((REPO_ROOT / source).read_bytes())
    with pytest.raises(ValueError, match="differs from the loaded execution contract"):
        graph.node_task_type({"task": str(duplicate)})
