"""Failure-edge regression: the walk, exercised through actual failures.

Every failure-path bug of run glm52-int-w8a8-p800-001 — the mat-006 <->
mat-020 failure-edge cycle, crash logs truncated by the triage reproof, a
node failing without any durable record — shared one root cause: the failure
edges had never been executed by a test. The graph tests covered topology
(edges point somewhere meaningful) and input resolution; nothing ever
*walked* a failing node.

These tests run main() for real: real task contracts, real workflow files,
stub node commands that write state and exit with controlled codes — so the
behaviour under failure is pinned at the only level where it exists: the
walk. Covered: REWORK routing with crash evidence, the cycle guard's bounded
termination, exit-0-but-failed-state routing, failure recording in task
memory, and reruns that must not destroy the previous attempt's evidence.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners import graph_runner  # noqa: E402

TRIAGE_NODE = {
    "id": "mat-006-failure-triage",
    "task": "tasks/mat-006-failure-triage/task.yaml",
    "on_success": "DELIVERED",
    "on_failure": "mat-020-vendor-handoff",
}
HANDOFF_NODE = {
    "id": "mat-020-vendor-handoff",
    "task": "tasks/mat-020-vendor-handoff/task.yaml",
    "on_success": "DELIVERED",
    "on_failure": "mat-006-failure-triage",  # the cycle that bit the run
}


# A real script file, not a -c blob: the walker .format()s every command
# element, so braces inside a -c script would be read as format fields.
STUB_SCRIPT = """\
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({"state": sys.argv[2], "reason": "stub"}))
print(sys.argv[3])
sys.exit(int(sys.argv[4]))
"""


def stub_command(script: Path, state: str, exit_code: int, message: str,
                 state_file: str = "status.json") -> list[str]:
    """A node command that writes a state file, prints, exits as told."""
    return [sys.executable, str(script), f"{{artifacts}}/{state_file}",
            state, message, str(exit_code)]


class WalkHarness(unittest.TestCase):
    """Run graph_runner.main() against a synthetic workflow with stub nodes."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.stub_script = self.tmp / "node_stub.py"
        self.stub_script.write_text(STUB_SCRIPT, encoding="utf-8")
        self.workflow = self.tmp / "failure_workflow.yaml"
        self.saved_nodes = {
            key: graph_runner.NODES[key]
            for key in ("failure_triage", "vendor_handoff")
        }
        self.saved_argv = list(sys.argv)

    def tearDown(self) -> None:
        graph_runner.NODES.update(self.saved_nodes)
        sys.argv[:] = self.saved_argv

    def write_workflow(self, nodes: list[dict], start: str) -> None:
        self.workflow.write_text(textwrap.dedent(f"""
            spec:
              nodes:
            """).lstrip("\n") + "".join(
            f"    - id: {node['id']}\n"
            f"      task: {node['task']}\n"
            f"      on_success: {node['on_success']}\n"
            f"      on_failure: {node['on_failure']}\n"
            for node in nodes
        ), encoding="utf-8")
        self.start = start

    def stub(self, task_type: str, state: str, exit_code: int,
             message: str = "stub traceback: boom") -> None:
        # Drop the input requirements: this suite pins edge mechanics, not
        # input resolution (that is InputResolutionTest's job in
        # test_graph_runner), and the real specs demand journal facts the
        # synthetic workflow never records.
        spec = {key: value for key, value in self.saved_nodes[task_type].items()
                if key not in ("needs", "optional")}
        # Write the spec's OWN state file: the walk reads that name, and a
        # stub writing a different one would fail (correctly) the
        # state-mismatch routing.
        state_file = spec["state_file"]
        spec["command"] = stub_command(
            self.stub_script, state, exit_code, message, state_file)
        graph_runner.NODES[task_type] = spec

    def walk(self, from_node: str) -> list[dict]:
        """Execute the walk; return every JSON summary it emitted."""
        journal = self.tmp / "journal.jsonl"
        argv = [
            "graph_runner",
            "--subject", "failure-edge-subject",
            "--artifact-root", str(self.tmp / "artifacts"),
            "--journal", str(journal),
            "--env", "hardware=P800",
            "--set", "issue=runtime_state",
            "--from-node", from_node,
            "--execute", "--json", "--watch-interval", "0",
        ]
        captured = io.StringIO()
        with redirect_stdout(captured):
            sys.argv[:] = argv
            graph_runner.main()
        summaries = []
        for line in captured.getvalue().splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    summaries.append(json.loads(line))
                except ValueError:
                    continue
        return summaries


class FailureRoutingTest(WalkHarness):
    def test_a_failing_node_routes_rework_with_its_crash_evidence(self):
        self.write_workflow([TRIAGE_NODE, HANDOFF_NODE], "mat-006-failure-triage")
        self.stub("failure_triage", "TRIAGE_FAILED", 1)
        self.stub("vendor_handoff", "HANDOFF_READY", 0)

        summaries = self.walk("mat-006-failure-triage")

        rework = next(s for s in summaries if s["status"] == "REWORK")
        self.assertEqual(rework["node"], "mat-006-failure-triage")
        self.assertEqual(rework["next_task"], "mat-020-vendor-handoff")
        self.assertEqual(rework["reason_code"], "COMMAND_FAILED")
        # The crash snapshot is part of the failure's artifacts, not a
        # console memory: the REWORK summary points at the file.
        crash_paths = [a for a in rework["artifacts"] if "/crash/" in a]
        self.assertEqual(len(crash_paths), 1)
        crash = Path(crash_paths[0])
        self.assertTrue(crash.exists())
        self.assertIn("stub traceback: boom", crash.read_text(encoding="utf-8"))
        # The walk followed the failure edge and the successor completed
        # successfully (a terminal success edge emits no CONTINUE summary;
        # its durable proof is the successor's own state file).
        handoff = self.tmp / "artifacts" / "mat-020-vendor-handoff" / "handoff_status.json"
        self.assertEqual(
            json.loads(handoff.read_text(encoding="utf-8"))["state"],
            "HANDOFF_READY")
        self.assertFalse(any(s.get("status") == "REWORK"
                             and s.get("node") == "mat-020-vendor-handoff"
                             for s in summaries))

    def test_exit_zero_with_a_failed_state_still_takes_the_failure_edge(self):
        # The node's own state file is the contract: exit 0 plus TRIAGE_FAILED
        # must reach triage's successor as REWORK, never ride the success
        # edge — that would skip diagnosis for a failure that already
        # happened.
        self.write_workflow([TRIAGE_NODE, HANDOFF_NODE], "mat-006-failure-triage")
        self.stub("failure_triage", "TRIAGE_FAILED", 0)
        self.stub("vendor_handoff", "HANDOFF_READY", 0)

        summaries = self.walk("mat-006-failure-triage")

        rework = next(s for s in summaries if s["status"] == "REWORK")
        self.assertEqual(rework["reason_code"], "STATE_NOT_SUCCESS")
        self.assertEqual(rework["next_task"], "mat-020-vendor-handoff")
        self.assertEqual(rework["state"], "TRIAGE_FAILED")

    def test_the_failure_is_recorded_in_task_memory(self):
        self.write_workflow([TRIAGE_NODE, HANDOFF_NODE], "mat-006-failure-triage")
        self.stub("failure_triage", "TRIAGE_FAILED", 1)
        self.stub("vendor_handoff", "HANDOFF_READY", 0)

        self.walk("mat-006-failure-triage")

        memory = json.loads(
            (self.tmp / "artifacts" / "task_memory.json").read_text(encoding="utf-8"))
        # Observed issues are persisted as OBSERVED claims.
        claims = memory.get("claims") or []
        self.assertTrue(
            any(claim.get("status") == "OBSERVED"
                and claim.get("observed_issue") == "command_failure"
                and claim.get("source") == "mat-006-failure-triage"
                for claim in claims),
            msg=f"failure not recorded: {claims}",
        )


class FailureCycleTest(WalkHarness):
    def test_a_failure_edge_cycle_terminates_bounded_not_forever(self):
        # mat-006 fails -> mat-020 fails -> back to mat-006 ... The run
        # looped this pair dozens of times before the guard existed; the
        # walk must stop with NEEDS_HUMAN after a bounded number of visits.
        self.write_workflow([TRIAGE_NODE, HANDOFF_NODE], "mat-006-failure-triage")
        self.stub("failure_triage", "TRIAGE_FAILED", 1)
        self.stub("vendor_handoff", "HANDOFF_FAILED", 1)

        summaries = self.walk("mat-006-failure-triage")

        terminal = next(s for s in summaries if s["status"] == "NEEDS_HUMAN")
        self.assertEqual(terminal["reason_code"], "FAILURE_EDGE_CYCLE")
        self.assertIn("mat-006-failure-triage", terminal["message"])
        # Bounded: at most 3 visits per node before the guard fires, so at
        # most 6 executions total — and every one left its crash snapshot.
        crash_dir = self.tmp / "artifacts" / "mat-006-failure-triage" / "crash"
        triage_crashes = list(crash_dir.glob("mat-006-failure-triage.crash.log*"))
        self.assertGreaterEqual(len(triage_crashes), 3)
        self.assertLessEqual(len(triage_crashes), 3)
        handoff_crashes = list((self.tmp / "artifacts" / "mat-020-vendor-handoff"
                                / "crash").glob("*.crash.log*"))
        self.assertLessEqual(len(handoff_crashes), 3)


class FailureEvidenceSurvivalTest(WalkHarness):
    def test_a_rerun_never_destroys_the_previous_attempt_evidence(self):
        # The original failure-edge clobbering: the rerun overwrote the
        # first attempt's logs. The walk must accumulate, never replace.
        self.write_workflow([TRIAGE_NODE, HANDOFF_NODE], "mat-006-failure-triage")
        self.stub("failure_triage", "TRIAGE_FAILED", 1)
        self.stub("vendor_handoff", "HANDOFF_READY", 0)

        self.walk("mat-006-failure-triage")
        self.walk("mat-006-failure-triage")

        crash_dir = self.tmp / "artifacts" / "mat-006-failure-triage" / "crash"
        crashes = list(crash_dir.glob("mat-006-failure-triage.crash.log*"))
        self.assertEqual(len(crashes), 2)
        # Each snapshot still carries its own attempt's content.
        for crash in crashes:
            self.assertIn("stub traceback: boom", crash.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
