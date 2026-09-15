"""The watch journals: "still running" must never be a silent state.

Run glm52-int-w8a8-p800-001 had four "any progress?" interruptions during
15-minute silent stretches — a 707 GiB load behind a health poll answering
503, a node whose console legitimately prints nothing until it finishes.
Nothing durable existed in between. These tests pin the two watches that
close that gap: the local heartbeat journal beats even when the console is
quiet, and the pod-side startup watch journals log growth and flags a stall
well before the deadline.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners import evidence  # noqa: E402
from runners import watch as watch_module  # noqa: E402
from runners.deployment_proof import ActionFailed, DeploymentProofRunner  # noqa: E402


def journal_entries(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


class LogWatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def test_beats_journal_while_a_quiet_child_runs(self):
        # A silent console is the normal shape of a long load: the journal
        # must beat anyway, proving the runner is alive.
        log = self.tmp / "console.log"
        watch = watch_module.LogWatch(
            "quiet-node", log, self.tmp / "watch_journal.jsonl",
            interval=0.05, echo=None,
        ).start()
        result = evidence.run_logged(
            [sys.executable, "-c", "import time; time.sleep(0.25)"],
            cwd=self.tmp, log_path=log, echo=lambda *_a, **_k: None, watch=watch,
        )
        summary = watch.stop(f"exit {result.returncode}")

        entries = journal_entries(self.tmp / "watch_journal.jsonl")
        beats = [e for e in entries if e["phase"] == "beat"]
        final = [e for e in entries if e["phase"] == "final"]
        self.assertGreaterEqual(len(beats), 2)
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0]["outcome"], "exit 0")
        self.assertEqual(final[0]["log_bytes"], 0)
        self.assertEqual(summary["beats"], len(beats))

    def test_output_lines_become_progress_entries(self):
        log = self.tmp / "console.log"
        watch = watch_module.LogWatch(
            "chatty-node", log, self.tmp / "watch_journal.jsonl",
            interval=60, echo=None,  # no beat lands in a fast run
        ).start()
        evidence.run_logged(
            [sys.executable, "-c", "print('one'); print('two')"],
            cwd=self.tmp, log_path=log, echo=lambda *_a, **_k: None, watch=watch,
        )
        summary = watch.stop("exit 0")

        entries = journal_entries(self.tmp / "watch_journal.jsonl")
        progress = [e for e in entries if e["phase"] == "progress"]
        self.assertEqual(len(progress), 2)
        self.assertEqual(summary["observations"], 2)
        self.assertEqual(summary["beats"], 0)

    def test_the_final_entry_carries_the_last_line_digest(self):
        log = self.tmp / "console.log"
        watch = watch_module.LogWatch(
            "node", log, self.tmp / "watch_journal.jsonl",
            interval=60, echo=None,
        ).start()
        evidence.run_logged(
            [sys.executable, "-c", "print('loading shard 281/282')"],
            cwd=self.tmp, log_path=log, echo=lambda *_a, **_k: None, watch=watch,
        )
        watch.stop("exit 0")

        final = next(e for e in
                     journal_entries(self.tmp / "watch_journal.jsonl")
                     if e["phase"] == "final")
        self.assertIn("shard 281/282", final["last_line"])
        self.assertGreater(final["log_bytes"], 0)

    def test_stop_is_idempotent(self):
        log = self.tmp / "console.log"
        watch = watch_module.LogWatch(
            "node", log, self.tmp / "watch_journal.jsonl", interval=60,
            echo=None,
        ).start()
        watch.stop("exit 0")
        watch.stop("exit 0")
        finals = [e for e in
                  journal_entries(self.tmp / "watch_journal.jsonl")
                  if e["phase"] == "final"]
        self.assertEqual(len(finals), 1)

    def test_echo_is_suppressed_when_none(self):
        log = self.tmp / "console.log"
        watch = watch_module.LogWatch(
            "node", log, self.tmp / "watch_journal.jsonl", interval=0.01,
            echo=None,
        ).start()
        evidence.run_logged(
            [sys.executable, "-c", "pass"], cwd=self.tmp, log_path=log,
            echo=lambda *_a, **_k: None, watch=watch,
        )
        watch.stop("exit 0")  # no print, no crash


class _FakePodAdapter:
    """Duck-typed adapter: scripted exec output, 503 health forever."""

    def __init__(self, stat_output: str) -> None:
        self.stat_output = stat_output
        self.execs = 0

    def exec(self, pod, script, timeout=None):  # noqa: ANN001
        self.execs += 1
        return subprocess.CompletedProcess(
            args=[], returncode=0, stdout=self.stat_output, stderr=""
        )

    def http_probe(self, pod, path, port):  # noqa: ANN001
        return 503, ""


def make_runner(adapter, execution: dict) -> DeploymentProofRunner:
    contract = {
        "metadata": {"name": "kdp-001b-service-proof"},
        "execution": {"server_log": "/workspace/server.log", **execution},
        "checks": {"health": {"path": "/health", "expected_status": 200}},
        "context": {"server": {"port": 8356}},
    }
    return DeploymentProofRunner(
        contract=contract, adapter=adapter, repo_root=ROOT,
        artifact_dir=Path(tempfile.mkdtemp()),
        attach_pod="pod-x", phase="service",
    )


class StartupWatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def test_a_growing_log_never_flags_a_stall(self):
        adapter = _FakePodAdapter("1048576\nINFO loading shard 281/282")
        runner = make_runner(adapter, {})
        for poll in range(4):
            runner._watch_server_log(poll, 503)

        entries = journal_entries(runner.artifact_dir / "startup_watch.jsonl")
        self.assertEqual(len(entries), 4)
        self.assertTrue(all(e["log_bytes"] == 1048576 for e in entries))
        self.assertFalse(runner.checks.get("startup_log_stall"))

    def test_a_frozen_log_flags_the_stall_once_and_records_it(self):
        adapter = _FakePodAdapter("4096\nERROR cannot allocate")
        runner = make_runner(adapter, {"watch_stall_polls": 2})
        for poll in range(4):
            runner._watch_server_log(poll, 503)

        self.assertTrue(runner.checks.get("startup_log_stall"))
        entries = journal_entries(runner.artifact_dir / "startup_watch.jsonl")
        stalled = [e for e in entries if e.get("stall")]
        # Polls 2 and 3 of 0..3: every poll at or past the threshold.
        self.assertEqual(len(stalled), 2)
        self.assertEqual(stalled[0]["stall_polls"], 2)
        self.assertEqual(stalled[1]["stall_polls"], 3)
        self.assertIn("frozen", runner.records[-1]["detail"])
        # One failure record, not one per poll.
        stall_records = [r for r in runner.records
                         if r["action"] == "watch_server_log"]
        self.assertEqual(len(stall_records), 1)

    def test_append_only_journal_survives_repeated_polls(self):
        adapter = _FakePodAdapter("10\nline")
        runner = make_runner(adapter, {})
        runner._watch_server_log(0, 503)
        runner._watch_server_log(1, 503)
        self.assertEqual(
            len((runner.artifact_dir / "startup_watch.jsonl")
                .read_text(encoding="utf-8").strip().splitlines()),
            2,
        )

    def test_the_timeout_reason_names_the_stall(self):
        # The deadline answer changes from "health never stabilised" to the
        # actionable "the log is frozen": slow and hung get told apart.
        adapter = _FakePodAdapter("4096\nINFO frozen here")
        runner = make_runner(
            adapter,
            {"watch_stall_polls": 2, "startup_timeout_seconds": 1,
             "health_interval_seconds": 0.02, "health_successes_required": 3},
        )
        with self.assertRaises(ActionFailed) as ctx:
            runner.poll_health()
        self.assertEqual(ctx.exception.state, "READINESS_TIMEOUT")
        self.assertIn("frozen at 4096 bytes", ctx.exception.reason)
        self.assertIn("startup_watch.jsonl", ctx.exception.reason)
        self.assertGreater(adapter.execs, 2)


if __name__ == "__main__":
    unittest.main()
