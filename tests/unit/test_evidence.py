"""Crash-first evidence: a dying process's log must outlive the reruns.

Run glm52-int-w8a8-p800-001 (2026-09-14) lost crash evidence twice — a node's
traceback that was never persisted, and a server log truncated 40 s after the
crash by the triage sent to explain it. These tests pin the invariants that
prevent both: unique paths are never replaced, and a failing child is
snapshotted before its log can be rewritten.
"""

from __future__ import annotations

import subprocess
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners import evidence  # noqa: E402
from runners.triage_executor import PodOps  # noqa: E402
from runners.patch_executor import PatchOps  # noqa: E402


class UniquePathTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def test_first_write_takes_the_plain_name(self):
        target = evidence.write_unique(self.tmp / "server_crash_log.txt", "boom")
        self.assertEqual(target.name, "server_crash_log.txt")
        self.assertEqual(target.read_text(encoding="utf-8"), "boom")

    def test_a_second_attempt_never_replaces_the_first(self):
        first = evidence.write_unique(self.tmp / "log.txt", "attempt 1 evidence")
        second = evidence.write_unique(self.tmp / "log.txt", "attempt 2 evidence")
        self.assertEqual(first.read_text(encoding="utf-8"), "attempt 1 evidence")
        self.assertEqual(second.read_text(encoding="utf-8"), "attempt 2 evidence")
        self.assertNotEqual(first, second)

    def test_the_suffix_sequence_is_exhausted_not_reused(self):
        names = {evidence.write_unique(self.tmp / "log.txt", str(n)).name for n in range(3)}
        self.assertEqual(names, {"log.txt", "log.txt.1", "log.txt.2"})


class RunLoggedTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def run_node(self, code: str) -> evidence.LoggedResult:
        return evidence.run_logged(
            [sys.executable, "-c", code],
            cwd=self.tmp,
            log_path=self.tmp / "node_console.log",
            crash_tag="mat-999-test-node",
        )

    def test_output_is_teed_to_the_console_log(self):
        result = self.run_node("import sys; print('hello'); print('world', file=sys.stderr)")
        self.assertEqual(result.returncode, 0)
        self.assertIn("hello", (self.tmp / "node_console.log").read_text(encoding="utf-8"))
        self.assertIn("world", (self.tmp / "node_console.log").read_text(encoding="utf-8"))

    def test_a_live_child_leaves_no_crash_snapshot(self):
        result = self.run_node("print('fine')")
        self.assertIsNone(result.crash_log)
        self.assertFalse((self.tmp / "crash").exists())

    def test_a_dead_child_is_snapshotted_before_anything_reruns(self):
        result = self.run_node("print('traceback: boom'); raise SystemExit(3)")
        self.assertEqual(result.returncode, 3)
        self.assertIsNotNone(result.crash_log)
        snapshot = Path(result.crash_log)
        self.assertIn("traceback: boom", snapshot.read_text(encoding="utf-8"))

    def test_the_second_death_is_a_new_snapshot_not_an_overwrite(self):
        first = Path(self.run_node("print('first death'); raise SystemExit(1)").crash_log)
        self.run_node("print('healthy rerun')")
        second = Path(self.run_node("print('second death'); raise SystemExit(1)").crash_log)
        self.assertIn("first death", first.read_text(encoding="utf-8"))
        self.assertIn("second death", second.read_text(encoding="utf-8"))
        self.assertNotEqual(first, second)
        # The scratch console log was rewritten by the rerun; the evidence was
        # not — that asymmetry is the entire point.
        self.assertIn("second death", (self.tmp / "node_console.log").read_text(encoding="utf-8"))


class RemoteArchiveSnippetTest(unittest.TestCase):
    def test_the_relaunch_archive_never_overwrites(self):
        snippet = evidence.archive_before_truncate("/workspace/server.log")
        self.assertIn("while [ -e '/workspace/server.log.prev-$n' ]", snippet)
        self.assertNotIn("cp -f", snippet)
        # The copy target shares the next-free index, not a fixed name.
        self.assertIn("cp '/workspace/server.log' '/workspace/server.log.prev-$n'", snippet)

    def test_the_crash_archive_uses_a_distinct_series(self):
        snippet = evidence.archive_crash_remote("/workspace/server.log")
        self.assertIn("server.log.crash-$n", snippet)
        self.assertNotIn("server.log.prev-$n", snippet)
        self.assertNotIn("cp -f", snippet)


class TriageRerunPathSeparationTest(unittest.TestCase):
    """The triage reproof must not write the failure attempt's log path."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def test_the_rerun_command_carries_its_own_server_log(self):
        captured: dict[str, list[str]] = {}

        def fake_run(command, cwd=None, text=None, capture_output=None):
            captured["command"] = list(command)
            output = self.tmp / "service_rerun" / "tasks" / "proof" / "attempts" / "000001" / "output"
            output.mkdir(parents=True)
            payload = {"state": "DEPLOYMENT_READY", "artifact_root": str(output)}
            (output / "status.json").write_text(json.dumps(payload), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

        with mock.patch.object(subprocess, "run", fake_run):
            status = PodOps(adapter=None).rerun_service(
                "pod-x", Path("/nonexistent/contract.yaml"), self.tmp / "service_rerun"
            )
        self.assertEqual(status["state"], "DEPLOYMENT_READY")
        self.assertNotEqual(Path(status["artifact_root"]), self.tmp / "service_rerun")
        command = captured["command"]
        flag = command.index("--server-log")
        self.assertTrue(command[flag + 1].startswith("/workspace/server.log.rerun-"))
        self.assertNotEqual(command[flag + 1], "/workspace/server.log")

    def test_patch_rerun_consumes_actual_output_and_rejects_failed_exit(self):
        output = self.tmp / "actual-output"
        output.mkdir()
        payload = {"state": "DEPLOYMENT_READY", "artifact_root": str(output)}
        (output / "status.json").write_text(json.dumps(payload), encoding="utf-8")
        for returncode, expected in ((0, "DEPLOYMENT_READY"), (6, "BLOCKED")):
            result = subprocess.CompletedProcess([], returncode, json.dumps(payload), "failed validator")
            with mock.patch.object(subprocess, "run", return_value=result):
                status = PatchOps(adapter=None).rerun_service(
                    "pod-x", Path("/nonexistent/contract.yaml"), self.tmp / "requested",
                )
            self.assertEqual(status["state"], expected)
            self.assertEqual(status["artifact_root"], str(output))

    def test_task_status_requires_persisted_matching_output(self):
        for stdout in ("", "{}", "[]"):
            with self.assertRaises(ValueError):
                evidence.task_status(subprocess.CompletedProcess([], 0, stdout, ""))
        output = self.tmp / "mismatch"
        output.mkdir()
        payload = {"state": "DEPLOYMENT_READY", "artifact_root": str(output)}
        (output / "status.json").write_text('{"state":"BLOCKED"}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differs"):
            evidence.task_status(subprocess.CompletedProcess([], 0, json.dumps(payload), ""))

    def test_the_rerun_log_derives_from_the_contract_instance(self):
        contract = self.tmp / "kdp_instance.yaml"
        contract.write_text(
            "execution:\n  server_log: /workspace/server.log\n", encoding="utf-8"
        )
        path = PodOps.rerun_server_log(contract)
        self.assertTrue(path.startswith("/workspace/server.log.rerun-"))

    def test_a_missing_contract_falls_back_to_the_default_path(self):
        path = PodOps.rerun_server_log(None)
        self.assertTrue(path.startswith("/workspace/server.log.rerun-"))


if __name__ == "__main__":
    unittest.main()
