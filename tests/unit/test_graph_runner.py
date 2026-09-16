"""The graph executor: input resolution, edges, and refusing to guess."""

from __future__ import annotations

import sys
import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners.graph_runner import (  # noqa: E402
    MANUAL,
    NODES,
    Unresolved,
    _failure_reason,
    load_workflow,
    node_task_type,
    resolve,
    reusable_fact,
    bind_proven_environment,
    fact_environment,
    record_fact,
)
from engine.state.journal import query, record
from runners import graph_runner  # noqa: E402
import cli.workflow.graph as _graph_runner_cli
from core.storage import RunPaths, WritePolicyError  # noqa: E402

WORKFLOW = ROOT / "workflows" / "model_adaptation.yaml"
ENVIRONMENT = {"hardware": "P800", "stack_commit": "3ced109a"}


def graph_fixture(tmp_path, monkeypatch, outcomes):
    workflow = [{"id": "intake", "task": "fixture", "on_success": "DELIVERED",
                 "on_failure": "REWORK"}]
    monkeypatch.setattr(graph_runner, "load_workflow", lambda _: workflow)
    monkeypatch.setattr(graph_runner, "node_task_type", lambda _: "model_intake")
    monkeypatch.setattr(graph_runner.skill_registry, "resolve_for_context",
                        lambda *_: {"id": "fixture", "verification": [], "tools": []})
    calls = []

    def run(command, *, log_path, **kwargs):
        calls.append({"command": command, "kwargs": kwargs})
        out = Path(command[command.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        payload = outcomes.pop(0) if outcomes else {"state": "INTAKE_READY"}
        (out / "intake_status.json").write_text(json.dumps(payload))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("fixture console")
        return SimpleNamespace(returncode=0, crash_log=None)

    monkeypatch.setattr(graph_runner.evidence, "run_logged", run)
    argv = ["graph", "--subject", "demo", "--artifact-root", str(tmp_path / "run"),
            "--env", "hardware=P800", "--set", "model_path=fixture",
            "--watch-interval", "0", "--json"]
    return argv, calls


def test_graph_two_invocations_preserve_reports_and_resume_journal_paths(tmp_path, monkeypatch):
    argv, calls = graph_fixture(tmp_path, monkeypatch, [])
    monkeypatch.setattr(sys, "argv", argv + ["--execute"])
    assert _graph_runner_cli.main() == 0
    root = tmp_path / "run"
    first = next(root.glob("tasks/intake/attempts/*/output/intake_status.json"))
    old = first.read_bytes()
    assert _graph_runner_cli.main() == 0
    reports = sorted(root.glob("tasks/intake/attempts/*/output/intake_status.json"))
    assert len(reports) == 2
    assert first.read_bytes() == old
    assert all((p.parent.parent / "manifest.json").is_file() for p in reports)
    monkeypatch.setattr(sys, "argv", argv + ["--execute", "--resume"])
    assert _graph_runner_cli.main() == 0
    assert len(calls) == 2
    memory = json.loads((root / "task_memory.json").read_text())
    assert str(reports[-1].parent) in json.dumps(memory["completed_loop_blocks"][-1])
    assert len(list(root.glob("tasks/intake/attempts/*"))) == 2


def test_graph_snapshots_selected_skill_in_attempt_and_task_memory(tmp_path, monkeypatch):
    argv, calls = graph_fixture(tmp_path, monkeypatch, [])
    monkeypatch.setattr(sys, "argv", argv + ["--execute"])
    assert _graph_runner_cli.main() == 0
    root = tmp_path / "run"
    packet_path = next(root.glob("tasks/intake/attempts/*/input/skill.json"))
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    assert packet["id"] == "fixture"
    assert packet["task_type"] == "model_intake"
    memory = json.loads((root / "task_memory.json").read_text(encoding="utf-8"))
    assert memory["completed_loop_blocks"][0]["routing"]["skill"] == "fixture"
    assert len(calls) == 1
    assert calls[0]["kwargs"]["env_overrides"]["INFER_FORGE_SKILL_CONTRACT"] == str(
        packet_path
    )


def test_graph_plan_and_plan_resume_do_not_write(tmp_path, monkeypatch):
    argv, calls = graph_fixture(tmp_path, monkeypatch, [])
    monkeypatch.setattr(sys, "argv", argv)
    assert _graph_runner_cli.main() == 0
    assert not (tmp_path / "run").exists()
    monkeypatch.setattr(sys, "argv", argv + ["--execute"])
    assert _graph_runner_cli.main() == 0
    files = {p: p.read_bytes() for p in (tmp_path / "run").rglob("*") if p.is_file()}
    monkeypatch.setattr(sys, "argv", argv + ["--resume"])
    assert _graph_runner_cli.main() == 0
    assert files == {p: p.read_bytes() for p in (tmp_path / "run").rglob("*") if p.is_file()}
    assert len(calls) == 1


@pytest.mark.parametrize("extra", [
    ["--artifact-root", str(ROOT / "runtime-test")],
    ["--journal", str(ROOT / "journal-test.jsonl")],
    ["--loop-state", str(ROOT / "memory-test.json")],
    ["--set", "artifacts=/outside"],
    ["--set", "malformed"],
])
def test_graph_write_violation_blocks_before_commands(tmp_path, monkeypatch, extra):
    argv, calls = graph_fixture(tmp_path, monkeypatch, [])
    monkeypatch.setattr(sys, "argv", argv + ["--execute"] + extra)
    assert _graph_runner_cli.main() == 2
    assert calls == []
    assert not (tmp_path / "run").exists()


def test_graph_default_run_root_uses_explicit_identity(tmp_path, monkeypatch):
    argv, calls = graph_fixture(tmp_path, monkeypatch, [])
    index = argv.index("--artifact-root")
    del argv[index:index + 2]
    monkeypatch.setenv("INFER_FORGE_STATE_ROOT", str(tmp_path / "state"))
    monkeypatch.setattr(sys, "argv", argv + ["--run-id", "explicit", "--execute"])
    assert _graph_runner_cli.main() == 0
    root = tmp_path / "state" / "runs" / "explicit"
    assert json.loads((root / "run.json").read_text())["run_id"] == "explicit"
    assert len(calls) == 1


def test_recovery_preserves_original_and_propagates_successful_attempt(tmp_path, monkeypatch):
    argv, calls = graph_fixture(tmp_path, monkeypatch, [
        {"state": "FAILED", "reason": "readiness timeout"},
        {"state": "FAILED", "reason": "still not valid"},
        {"state": "INTAKE_READY"},
    ])
    monkeypatch.setattr(sys, "argv", argv + [
        "--execute", "--auto-recover", "--brain", "rule", "--recovery-budget", "2",
    ])
    assert _graph_runner_cli.main() == 0
    root = tmp_path / "run"
    reports = sorted(root.glob("tasks/intake/attempts/*/output/intake_status.json"))
    assert len(calls) == len(reports) == 3
    assert len(list(root.glob("tasks/intake/attempts/*/input/skill.json"))) == 3
    assert json.loads(reports[0].read_text())["reason"] == "readiness timeout"
    assert json.loads(reports[1].read_text())["state"] == "FAILED"
    hit = reusable_fact(NODES["model_intake"], "demo", root / "journal.jsonl",
                        {"hardware": "P800"})
    assert hit["artifacts"] == str(reports[-1].parent)
    memory = json.loads((root / "task_memory.json").read_text())
    assert str(reports[-1].parent) in json.dumps(memory["completed_loop_blocks"][-1])


@pytest.mark.parametrize("payload", [
    {"state": "UNKNOWN"}, {"state": "INTAKE_READY", "validator": {"passed": False}},
])
def test_recovery_exit_zero_without_valid_state_stays_blocked(tmp_path, monkeypatch, payload):
    argv, calls = graph_fixture(tmp_path, monkeypatch, [
        {"state": "FAILED", "reason": "readiness timeout"}, payload,
    ])
    monkeypatch.setattr(sys, "argv", argv + [
        "--execute", "--auto-recover", "--brain", "rule", "--recovery-budget", "1",
    ])
    assert _graph_runner_cli.main() == 0
    assert len(calls) == 2
    outcome = next((tmp_path / "run").rglob("recovery_outcome.json"))
    assert json.loads(outcome.read_text())["status"] == "BLOCKED"


def test_retry_cannot_redirect_output_to_source(tmp_path, monkeypatch):
    from engine.brain import Decision

    argv, calls = graph_fixture(tmp_path, monkeypatch, [
        {"state": "FAILED", "reason": "readiness timeout"},
    ])
    brain = SimpleNamespace(decide=lambda _: Decision(
        "RETRY_WITH_PARAMS", "redirect output", params={"artifacts": str(ROOT)},
    ))
    monkeypatch.setattr("engine.brain.brain_from_config", lambda *_: brain)
    monkeypatch.setattr(sys, "argv", argv + [
        "--execute", "--auto-recover", "--recovery-budget", "1",
    ])
    assert _graph_runner_cli.main() == 0
    assert len(calls) == 1
    assert len(list((tmp_path / "run").glob("tasks/intake/attempts/*"))) == 1


@pytest.mark.parametrize("name,content", [
    ("run.json", "[]"),
    ("run.json", '{"schema_version":1,"run_id":"other"}'),
    ("task_memory.json", "[]"),
    ("task_memory.json", "{invalid"),
])
def test_malformed_existing_state_blocks_before_node(tmp_path, monkeypatch, name, content):
    argv, calls = graph_fixture(tmp_path, monkeypatch, [])
    root = tmp_path / "run"
    root.mkdir()
    (root / name).write_text(content)
    monkeypatch.setattr(sys, "argv", argv + ["--execute", "--run-id", "expected"])
    assert _graph_runner_cli.main() == 2
    assert calls == []
    assert not (root / "tasks").exists()


def test_symlinked_journal_into_source_is_blocked(tmp_path, monkeypatch):
    argv, calls = graph_fixture(tmp_path, monkeypatch, [])
    link = tmp_path / "source"
    link.symlink_to(ROOT, target_is_directory=True)
    monkeypatch.setattr(sys, "argv", argv + [
        "--execute", "--journal", str(link / "runtime-journal.jsonl"),
    ])
    assert _graph_runner_cli.main() == 2
    assert calls == []


@pytest.mark.parametrize("item", ["../escape", "/escape", ".", "", "nested/path"])
def test_unsafe_fanout_is_rejected_before_execution(tmp_path, item):
    spec = {"command": ["child", "{artifacts}"],
            "fan_out": {"list": ["list"], "var": "item", "aggregate": ["aggregate"]}}
    with patch.object(graph_runner.subprocess, "run",
                      return_value=SimpleNamespace(returncode=0, stdout=json.dumps([item]))):
        with pytest.raises(WritePolicyError):
            graph_runner.fan_out_plan(spec, {"subject": "demo"},
                                     tmp_path / "journal", {}, tmp_path / "out")


def test_fanout_children_have_unique_outputs_and_logs(tmp_path):
    spec = {"command": ["child", "{item}", "{artifacts}"],
            "fan_out": {"list": ["list"], "var": "item",
                        "aggregate": ["aggregate", "{artifacts}"]}}
    planned = tmp_path / "run" / "{attempt-output}"
    with patch.object(graph_runner.subprocess, "run",
                      return_value=SimpleNamespace(returncode=0, stdout='["one","two"]')):
        commands = graph_runner.fan_out_plan(
            spec, {"subject": "demo"}, tmp_path / "journal", {}, planned,
        )
    attempt, commands = graph_runner.allocate_commands(
        RunPaths(tmp_path / "run"), "fanout", planned, commands,
    )
    assert len({target for target, _ in commands}) == 3
    assert len({graph_runner.command_logs(attempt, target) for target, _ in commands}) == 3


def environment_bundle(bundle: Path) -> None:
    bundle.mkdir(parents=True, exist_ok=True)
    names = [
        "environment_fingerprint.txt", "runtime_import.txt", "code_readiness.json",
        "device_readiness.json", "base_model_identity.json", "base_server_log.txt",
        "base_health_result.txt", "base_chat_result.json",
    ]
    for name in names:
        (bundle / name).write_text("fixture evidence", encoding="utf-8")
    (bundle / "status.json").write_text(json.dumps({
        "state": "ENVIRONMENT_READY", "pod": "prepared-pod", "artifacts": names,
        "checks": {
            "pod_ready": True, "runtime_importable": True, "code_ready": True,
            "device_ready": True, "base_model_loaded": True, "base_prefill": True,
            "base_decode": True, "base_health_check": 200,
            "base_chat_completion": "non_empty", "unexpected_fallback": False,
        },
    }), encoding="utf-8")


class WorkflowShapeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.nodes = load_workflow(WORKFLOW)
        self.ids = {node["id"] for node in self.nodes}

    def test_every_edge_points_somewhere_meaningful(self):
        terminal = {"NEEDS_HUMAN", "REWORK", "DELIVERED", "PLANNED"}
        for node in self.nodes:
            for edge in ("on_success", "on_failure"):
                target = node.get(edge)
                if target is None:
                    continue
                with self.subTest(node=node["id"], edge=edge):
                    self.assertIn(target, self.ids | terminal)

    def test_a_service_proof_failure_leads_to_triage_not_to_a_dead_end(self):
        service = next(node for node in self.nodes if node["id"] == "kdp-001b-service-proof")
        self.assertEqual(service["on_failure"], "mat-006-failure-triage")

    def test_a_binary_layer_verdict_ends_in_a_delivery(self):
        triage = next(node for node in self.nodes if node["id"] == "mat-006-failure-triage")
        self.assertEqual(triage["on_failure"], "mat-020-vendor-handoff")
        handoff = next(node for node in self.nodes if node["id"] == "mat-020-vendor-handoff")
        self.assertEqual(handoff["on_success"], "DELIVERED")

    def test_every_wired_node_is_either_executable_or_declared_manual(self):
        """A node the walk cannot run must say so, not fail obscurely mid-graph."""
        for node in self.nodes:
            task_type = node_task_type(node)
            if task_type is None:
                continue
            with self.subTest(node=node["id"], task_type=task_type):
                self.assertTrue(task_type in NODES or task_type in MANUAL)

    def test_multi_step_nodes_explain_themselves(self):
        for task_type, guidance in MANUAL.items():
            with self.subTest(task_type=task_type):
                self.assertIn("runs_with", guidance)

    def test_triage_and_placement_are_executable_not_manual(self):
        """The walk must not stop where a recovery decision needs them."""
        for task_type in ("failure_triage", "patch_placement",
                          "platform_kernel_correctness", "end_to_end_accuracy",
                          "long_context_sparse_correctness"):
            self.assertIn(task_type, NODES)
            self.assertNotIn(task_type, MANUAL)

    def test_every_task_type_in_the_workflow_has_an_executor(self):
        """With MANUAL empty, no wired node may any longer stop for a person."""
        self.assertEqual(MANUAL, {})
        for node in self.nodes:
            task_type = node_task_type(node)
            if task_type is None:
                continue
            with self.subTest(node=node["id"], task_type=task_type):
                self.assertIn(task_type, NODES)


class FailureReasonTest(unittest.TestCase):
    def test_the_node_s_own_reason_reaches_the_brain(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            (artifacts / "status.json").write_text(
                json.dumps({"state": "DEPLOYMENT_FAILED",
                            "reason": "Check 0 == ret failed"}), encoding="utf-8")
            reason = _failure_reason(artifacts, {"state_file": "status.json"},
                                     "DEPLOYMENT_FAILED")
            self.assertEqual(reason, "Check 0 == ret failed")

    def test_without_a_reason_the_state_itself_is_the_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            reason = _failure_reason(Path(tmp), {"state_file": "status.json"},
                                     "READINESS_TIMEOUT")
            self.assertIn("READINESS_TIMEOUT", reason)


class InputResolutionTest(unittest.TestCase):
    def context(self, artifacts: Path) -> dict:
        return {"subject": "Qwen3-8B", "artifacts": str(artifacts), "attempt": "graph",
                "environment_text": "hardware=P800"}

    def test_inputs_come_from_recorded_facts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.jsonl"
            bundle = Path(tmp) / "mat-001"
            environment_bundle(bundle)
            (bundle / "intake_status.json").write_text('{"state":"INTAKE_READY"}')
            record(path, "ModelRequest", "Qwen3-8B", "INTAKE_READY", bundle, ENVIRONMENT)
            record(path, "EnvironmentProof", "Qwen3-8B", "ENVIRONMENT_READY", bundle, ENVIRONMENT)
            command = resolve(NODES["model_scan"], self.context(Path(tmp)), path, ENVIRONMENT)
            self.assertIn(str(bundle / "model_request.yaml"), command)
            self.assertIn(str(bundle / "status.json"), command)

    def test_a_missing_fact_stops_the_walk_instead_of_guessing_a_path(self):
        """The failure mode this replaces: a stale path answering for another model."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.jsonl"
            with self.assertRaises(Unresolved):
                resolve(NODES["model_scan"], self.context(Path(tmp)), path, ENVIRONMENT)

    def test_a_fact_from_another_environment_does_not_satisfy_a_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.jsonl"
            record(path, "ModelRequest", "Qwen3-8B", "INTAKE_READY", Path(tmp),
                   {**ENVIRONMENT, "stack_commit": "deadbeef"})
            record(path, "EnvironmentProof", "Qwen3-8B", "ENVIRONMENT_READY", Path(tmp),
                   {**ENVIRONMENT, "stack_commit": "deadbeef"})
            with self.assertRaises(Unresolved):
                resolve(NODES["model_scan"], self.context(Path(tmp)), path, ENVIRONMENT)

    def test_the_latest_fact_is_the_one_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.jsonl"
            old, new = Path(tmp) / "old", Path(tmp) / "new"
            for bundle in (old, new):
                bundle.mkdir()
                (bundle / "scan_status.json").write_text('{"state":"SCAN_READY"}')
                (bundle / "match_status.json").write_text('{"state":"MATCH_READY"}')
                record(path, "ModelSupportCard", "Qwen3-8B", "SCAN_READY", bundle, ENVIRONMENT)
                record(path, "CapabilityMatch", "Qwen3-8B", "MATCH_READY", bundle, ENVIRONMENT)
            command = resolve(NODES["gap_classification"], self.context(Path(tmp)), path, ENVIRONMENT)
            self.assertIn(str(new / "model_support.json"), command)
            self.assertNotIn(str(old / "model_support.json"), command)

    def test_resume_only_reuses_a_successful_fact_with_an_existing_state_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            journal = root / "journal.jsonl"
            bundle = root / "scan"
            bundle.mkdir()
            (bundle / "scan_status.json").write_text(
                '{"state": "SCAN_READY"}', encoding="utf-8"
            )
            record(journal, "ModelSupportCard", "Qwen3-8B", "SCAN_READY", bundle, ENVIRONMENT)
            hit = reusable_fact(
                NODES["model_scan"], "Qwen3-8B", journal, ENVIRONMENT
            )
            self.assertEqual(hit["artifacts"], str(bundle))

    def test_resume_does_not_reuse_a_failed_or_incomplete_fact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            journal = root / "journal.jsonl"
            failed = root / "failed"
            record(journal, "ModelSupportCard", "Qwen3-8B", "SCAN_FAILED", failed, ENVIRONMENT)
            self.assertIsNone(
                reusable_fact(NODES["model_scan"], "Qwen3-8B", journal, ENVIRONMENT)
            )


class FactReliabilityTest(unittest.TestCase):
    def test_recovered_environment_can_retain_crash_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            journal = root / "journal.jsonl"
            environment_bundle(root)
            (root / "crash").mkdir()
            status_path = root / "status.json"
            status = json.loads(status_path.read_text())
            status["artifacts"].append("crash")
            status_path.write_text(json.dumps(status))
            record_fact(journal, NODES["environment_proof"], "demo", root, ENVIRONMENT)
            self.assertIsNotNone(reusable_fact(NODES["environment_proof"], "demo", journal, ENVIRONMENT))
            required = root / "device_readiness.json"
            required.unlink()
            required.mkdir()
            self.assertIsNone(reusable_fact(NODES["environment_proof"], "demo", journal, ENVIRONMENT))

    def test_empty_environment_never_reuses_or_resolves_facts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            journal = root / "journal.jsonl"
            (root / "scan_status.json").write_text('{"state":"SCAN_READY"}')
            for env in (ENVIRONMENT, {}):
                record(journal, "ModelSupportCard", "demo", "SCAN_READY", root, env)
            self.assertEqual(len(query(journal, "ModelSupportCard", environment=None)), 2)
            self.assertEqual(query(journal, "ModelSupportCard", environment={}), [])
            self.assertIsNone(reusable_fact(NODES["model_scan"], "demo", journal, {}))
            with self.assertRaisesRegex(Unresolved, "nonempty environment"):
                resolve(NODES["model_scan"], {"subject": "demo"}, journal, {})

    def test_resume_rejects_changed_failed_malformed_or_invalid_status(self):
        for content in ('{"state":"SCAN_FAILED"}', '{"state":"INTAKE_READY"}',
                        '{"state":"SCAN_READY","validator":{"passed":false}}',
                        '{"state":"SCAN_READY","validator":{"passed":true,"errors":["bad"]}}',
                        '[]', 'null', '{invalid', '{}'):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                journal = root / "journal.jsonl"
                record(journal, "ModelSupportCard", "demo", "SCAN_READY", root, ENVIRONMENT)
                (root / "scan_status.json").write_text(content)
                self.assertIsNone(reusable_fact(NODES["model_scan"], "demo", journal, ENVIRONMENT))

    def test_new_failure_invalidates_older_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            journal = root / "journal.jsonl"
            (root / "scan_status.json").write_text('{"state":"SCAN_READY"}')
            record(journal, "ModelSupportCard", "demo", "SCAN_READY", root, ENVIRONMENT)
            record(journal, "ModelSupportCard", "demo", "SCAN_FAILED", root / "failed", ENVIRONMENT)
            self.assertIsNone(reusable_fact(NODES["model_scan"], "demo", journal, ENVIRONMENT))

    def test_recorded_status_is_content_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            journal = root / "journal.jsonl"
            status = root / "scan_status.json"
            status.write_text('{"state":"SCAN_READY","revision":"old"}')
            record_fact(journal, NODES["model_scan"], "demo", root, ENVIRONMENT)
            status.write_text('{"state":"SCAN_READY","revision":"new"}')
            self.assertIsNone(reusable_fact(NODES["model_scan"], "demo", journal, ENVIRONMENT))

    def test_resume_rejects_evidence_from_a_different_skill_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            journal = root / "journal.jsonl"
            (root / "scan_status.json").write_text('{"state":"SCAN_READY"}')
            old = {
                "id": "model-scanner",
                "method": {"sha256": "old"},
            }
            new = {
                "id": "model-scanner",
                "method": {"sha256": "new"},
            }
            record_fact(
                journal, NODES["model_scan"], "demo", root, ENVIRONMENT, skill=old
            )
            self.assertIsNotNone(
                reusable_fact(
                    NODES["model_scan"], "demo", journal, ENVIRONMENT, skill=old
                )
            )
            self.assertIsNone(
                reusable_fact(
                    NODES["model_scan"], "demo", journal, ENVIRONMENT, skill=new
                )
            )

    def test_failed_command_cannot_reuse_a_leftover_success_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            journal = root / "journal.jsonl"
            (root / "scan_status.json").write_text('{"state":"SCAN_READY"}')
            record_fact(journal, NODES["model_scan"], "demo", root, ENVIRONMENT, returncode=1)
            self.assertIsNone(reusable_fact(NODES["model_scan"], "demo", journal, ENVIRONMENT))

    def test_proven_fingerprint_scopes_runtime_but_not_intake(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            journal = root / "journal.jsonl"
            environment_bundle(root)
            record_fact(journal, NODES["environment_proof"], "demo", root, ENVIRONMENT)
            env, context = dict(ENVIRONMENT), {"subject": "demo"}
            bind_proven_environment(context, env, journal)
            self.assertTrue(env["environment_fingerprint"])
            self.assertEqual(context["pod"], "prepared-pod")
            self.assertEqual(fact_environment("ModelRequest", env), ENVIRONMENT)
            (root / "scan_status.json").write_text('{"state":"SCAN_READY"}')
            record_fact(journal, NODES["model_scan"], "demo", root, env)
            changed = {**env, "environment_fingerprint": "other-proof"}
            self.assertIsNone(reusable_fact(NODES["model_scan"], "demo", journal, changed))
            (root / "environment_fingerprint.txt").write_text("changed stack")
            self.assertIsNone(reusable_fact(NODES["environment_proof"], "demo", journal, ENVIRONMENT))

    def test_invalid_environment_proof_cannot_supply_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            journal = root / "journal.jsonl"
            environment_bundle(root)
            (root / "runtime_import.txt").unlink()
            record(journal, "EnvironmentProof", "demo", "ENVIRONMENT_READY", root, ENVIRONMENT)
            spec = {"command": ["probe"], "needs": {"--env-status": "fact:EnvironmentProof:status.json"}}
            with self.assertRaises(Unresolved):
                resolve(spec, {"subject": "demo"}, journal, ENVIRONMENT)
            context, env = {"subject": "demo"}, dict(ENVIRONMENT)
            bind_proven_environment(context, env, journal)
            self.assertNotIn("environment_fingerprint", env)


if __name__ == "__main__":
    unittest.main()
