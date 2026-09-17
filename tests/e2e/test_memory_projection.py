"""Production Graph/CLI rebuild disposable memory without re-running evidence."""

import json

import pytest

from tests.e2e.test_model_adaptation import Scenario, scenario  # noqa: F401
from tests.e2e.test_codex_interaction import failed_toy, decision_file, submit, toy_failure


pytestmark = pytest.mark.local_e2e


def memory_cli(case: Scenario, *arguments, expected=0):
    return case.process("cli/state/task_memory.py", [
        "--path", str(case.run_root / "task_memory.json"),
        "--journal", str(case.run_root / "journal.jsonl"),
        "--run-id", case.run_id, "--task-id", "model_adaptation",
        "--subject", case.fixture["subject"], *arguments,
    ], expected=expected)


def test_memory_projection_rebuild_rejection_and_graph_restart(scenario):
    scenario.graph("--until-node", "mat-001-model-intake")
    path = scenario.run_root / "task_memory.json"
    expected = json.loads(path.read_text())
    assert expected["completed_loop_blocks"]
    journal = scenario.run_root / "journal.jsonl"
    authoritative = journal.read_bytes()
    path.unlink()  # Fault injection targets only the expendable view.
    before = scenario.persisted_snapshot()
    assert json.loads(memory_cli(scenario, "--show").stdout) == expected
    assert scenario.persisted_snapshot() == before
    memory_cli(scenario, "--rebuild")
    assert json.loads(path.read_text()) == expected
    assert journal.read_bytes() == authoritative
    before = scenario.persisted_snapshot()
    rejected = memory_cli(scenario, "--run-id", "unrelated-run", "--rebuild", expected=2)
    assert "another run" in rejected.stderr
    assert scenario.persisted_snapshot() == before
    # Fresh Graph process consumes original validated facts, not the rebuilt view.
    scenario.graph("--resume", "--until-node", "mat-001-model-intake")
    resumed = json.loads(path.read_text())
    assert resumed["completed_loop_blocks"][:len(expected["completed_loop_blocks"])] == expected["completed_loop_blocks"]
    assert any(record["routing"].get("mode") == "reuse_journal_fact"
               for record in resumed["completed_loop_blocks"][len(expected["completed_loop_blocks"]):])


def test_memory_projection_preserves_failure_claims_and_pending_decision(scenario):
    handoff = failed_toy(scenario)
    path = scenario.run_root / "task_memory.json"
    expected = json.loads(path.read_text())
    assert any(claim.get("observed_issue") == "command_failure"
               for claim in expected["claims"])
    assert str(path) not in handoff["source"]["source_files"]
    assert any(name.endswith("task_memory_snapshot.json")
               for name in handoff["source"]["source_files"])
    authoritative = (scenario.run_root / "journal.jsonl").read_bytes()
    path.write_text("interrupted expendable cache")
    before = scenario.persisted_snapshot()
    assert json.loads(memory_cli(scenario, "--show").stdout) == expected
    assert scenario.persisted_snapshot() == before
    memory_cli(scenario, "--rebuild")
    assert json.loads(path.read_text()) == expected
    assert (scenario.run_root / "journal.jsonl").read_bytes() == authoritative
    assert scenario.context()["handoff"] == handoff
    request = decision_file(scenario, handoff)
    toy_failure(scenario, False)
    recovered = submit(scenario, handoff, request)
    assert recovered["receipt"]["outcome"]["status"] == "RECOVERED"
