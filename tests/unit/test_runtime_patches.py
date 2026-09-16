"""The replayable-repair rule, enforced: patches apply themselves.

AGENTS.md now forbids repairing runtime state without a committed,
idempotent, replayable patch — because a fix that lives only in a pod dies
with the pod (run glm52-int-w8a8-p800-001: thirteen repairs evaporated on
the next reinstall). The deployment proof enforces it mechanically: after
any install or attach, the patch set is replayed and the drift precheck
verifies the result. These tests pin the three behaviours that make the
enforcement honest: the replay happens, its output is evidence, and a
non-matching patch set is recorded as SKIPPED (not fatal — a repair for
one version pair must not block cross-version adaptation) with the drift
precheck as the verdict on what the environment actually needs.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners.deployment_proof import ActionFailed, DeploymentProofRunner  # noqa: E402


class _RecordingAdapter:
    def __init__(self, exit_codes: list[int]) -> None:
        self.exit_codes = list(exit_codes)
        self.scripts: list[str] = []

    def exec(self, pod, script, timeout=None):  # noqa: ANN001
        self.scripts.append(script)
        code = self.exit_codes.pop(0) if self.exit_codes else 0
        return subprocess.CompletedProcess(
            args=[], returncode=code,
            stdout=f"PATCHED something (exit {code})\n", stderr=""
        )


class _RaisingAdapter:
    def exec(self, pod, script, timeout=None):  # noqa: ANN001
        raise subprocess.TimeoutExpired(script, timeout)


def make_runner(adapter) -> DeploymentProofRunner:
    contract = {
        "metadata": {"name": "kdp-001b-service-proof"},
        "execution": {"server_log": "/workspace/server.log"},
        "checks": {"health": {"path": "/health", "expected_status": 200}},
        "context": {"server": {"port": 8356}},
    }
    return DeploymentProofRunner(
        contract=contract, adapter=adapter, repo_root=ROOT,
        artifact_dir=Path(tempfile.mkdtemp()),
        attach_pod="pod-x", phase="service",
    )


class RuntimePatchReplayTest(unittest.TestCase):
    def test_the_repo_patch_set_is_replayed_into_the_pod(self):
        # The real repo root carries tools/patches/patch_vllm_kunlun_drift.py;
        # the replay must push it (base64) and run it in the pod's venv.
        adapter = _RecordingAdapter([0])
        runner = make_runner(adapter)

        runner.apply_runtime_patches()

        self.assertEqual(len(adapter.scripts), 1)
        script = adapter.scripts[0]
        self.assertIn("base64 -d", script)
        self.assertIn("patch_vllm_kunlun_drift.py", script)
        self.assertIn("VIRTUAL_ENV=/opt/vllm_kunlun", script)
        artifact = runner.artifact_dir / "runtime_patches.txt"
        self.assertIn("PATCHED something", artifact.read_text(encoding="utf-8"))
        self.assertTrue(any(r["action"] == "apply_runtime_patches" and r["ok"]
                            for r in runner.records))

    def test_a_non_matching_replay_is_skipped_not_fatal(self):
        # Exit code 2 = anchors no longer match: this install is a different
        # engine/plugin pair than the one the patch set repairs. The replay
        # records SKIPPED with full evidence and the run continues — the
        # drift precheck decides whether this environment actually needs
        # repair. Failing the whole run here is what blocked cross-version
        # adaptation.
        adapter = _RecordingAdapter([2])
        runner = make_runner(adapter)

        runner.apply_runtime_patches()  # must not raise

        artifact = runner.artifact_dir / "runtime_patches.txt"
        evidence = artifact.read_text(encoding="utf-8")
        self.assertIn("exit 2", evidence)
        self.assertIn("SKIPPED", evidence)
        self.assertIn("patch_vllm_kunlun_drift.py", evidence)
        record = next(r for r in runner.records
                      if r["action"] == "apply_runtime_patches")
        self.assertTrue(record["ok"])
        self.assertIn("skipped (non-matching pair)", record["detail"])
        self.assertIn("patch_vllm_kunlun_drift.py", record["detail"])

    def test_an_execution_failure_is_not_mislabeled_as_version_mismatch(self):
        adapter = _RecordingAdapter([1])
        runner = make_runner(adapter)

        with self.assertRaises(ActionFailed) as ctx:
            runner.apply_runtime_patches()

        self.assertEqual(ctx.exception.state, "INSTALL_FAILED")
        evidence = (runner.artifact_dir / "runtime_patches.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("FAILED", evidence)
        record = next(r for r in runner.records
                      if r["action"] == "apply_runtime_patches")
        self.assertFalse(record["ok"])

    def test_a_timeout_is_recorded_as_an_install_failure(self):
        runner = make_runner(_RaisingAdapter())

        with self.assertRaises(ActionFailed) as ctx:
            runner.apply_runtime_patches()

        self.assertEqual(ctx.exception.state, "INSTALL_FAILED")
        evidence = (runner.artifact_dir / "runtime_patches.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("TimeoutExpired", evidence)
        record = next(r for r in runner.records
                      if r["action"] == "apply_runtime_patches")
        self.assertFalse(record["ok"])


class PatchScriptSemanticsTest(unittest.TestCase):
    """The patch script's own contract: applied / skipped / failed are
    distinguishable, because the replay's exit code is built on it."""

    @classmethod
    def setUpClass(cls) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "patch_vllm_kunlun_drift",
            ROOT / "tools" / "patches" / "patch_vllm_kunlun_drift.py",
        )
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_true_applied_false_skipped_none_failed(self):
        tmp = Path(tempfile.mkdtemp()) / "f.py"
        tmp.write_text("old line\n", encoding="utf-8")
        self.assertTrue(self.module.patch(tmp, [("old line", "new line")]))
        self.assertFalse(self.module.patch(tmp, [("old line", "new line")]))
        self.assertIsNone(self.module.patch(tmp, [("absent anchor", "x")]))
        # A failed patch leaves the file untouched — no partial writes.
        self.assertEqual(tmp.read_text(encoding="utf-8"), "new line\n")

    def test_transaction_does_not_write_until_committed(self):
        first = Path(tempfile.mkdtemp()) / "first.py"
        second = first.with_name("second.py")
        first.write_text("old first\n", encoding="utf-8")
        second.write_text("unexpected\n", encoding="utf-8")
        transaction = self.module.PatchTransaction()

        self.assertTrue(self.module.patch(
            first, [("old first", "new first")], transaction=transaction
        ))
        self.assertIsNone(self.module.patch(
            second, [("old second", "new second")], transaction=transaction
        ))

        self.assertEqual(first.read_text(encoding="utf-8"), "old first\n")
        self.assertEqual(second.read_text(encoding="utf-8"), "unexpected\n")

    def test_missing_target_is_an_execution_failure_not_an_anchor_mismatch(self):
        missing = Path(tempfile.mkdtemp()) / "missing.py"

        with self.assertRaises(FileNotFoundError):
            self.module.patch(missing, [("old", "new")])

    def test_transaction_rolls_back_a_mid_commit_failure(self):
        root = Path(tempfile.mkdtemp())
        first = root / "first.py"
        second = root / "second.py"
        first.write_text("old first\n", encoding="utf-8")
        second.write_text("old second\n", encoding="utf-8")
        transaction = self.module.PatchTransaction()
        transaction.stage(first, "new first\n")
        transaction.stage(second, "new second\n")
        original_write = Path.write_text

        def fail_second(path, content, encoding=None):
            if path == second and content == "new second\n":
                raise OSError("disk full")
            return original_write(path, content, encoding=encoding)

        with mock.patch.object(Path, "write_text", new=fail_second):
            with self.assertRaises(OSError):
                transaction.commit()

        self.assertEqual(first.read_text(encoding="utf-8"), "old first\n")
        self.assertEqual(second.read_text(encoding="utf-8"), "old second\n")

    def test_main_does_not_write_when_a_late_anchor_mismatches(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp) / "site-packages"
            calls = 0

            def late_mismatch(path, replacements, *, transaction):
                nonlocal calls
                calls += 1
                if calls == 4:
                    return None
                transaction.stage(path, f"staged change {calls}\n")
                return True

            with mock.patch.object(self.module, "SITE", site), \
                    mock.patch.object(self.module, "patch", side_effect=late_mismatch):
                result = self.module.main()

            self.assertEqual(result, self.module.NOT_APPLICABLE)
            self.assertFalse(site.exists())

    def test_local_patch_packaging_failure_is_recorded(self):
        runner = make_runner(_RecordingAdapter([]))

        with mock.patch(
            "runners.deployment_proof.push_snippet",
            side_effect=OSError("patch file unreadable"),
        ):
            with self.assertRaises(ActionFailed) as ctx:
                runner.apply_runtime_patches()

        self.assertEqual(ctx.exception.state, "INSTALL_FAILED")
        evidence = (runner.artifact_dir / "runtime_patches.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("OSError: patch file unreadable", evidence)
        record = next(r for r in runner.records
                      if r["action"] == "apply_runtime_patches")
        self.assertFalse(record["ok"])


if __name__ == "__main__":
    unittest.main()
