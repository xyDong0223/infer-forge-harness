"""The replayable-repair rule, enforced: patches apply themselves.

AGENTS.md now forbids repairing runtime state without a committed,
idempotent, replayable patch — because a fix that lives only in a pod dies
with the pod (run glm52-int-w8a8-p800-001: thirteen repairs evaporated on
the next reinstall). The deployment proof enforces it mechanically: after
any install or attach, the patch set is replayed and the drift precheck
verifies the result. These tests pin the three behaviours that make the
enforcement honest: the replay happens, its output is evidence, and a
broken replay fails the run instead of passing silently.
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

    def test_a_broken_replay_fails_the_run_not_silently(self):
        # Exit code 1 = anchors no longer match: the engine/plugin pair
        # moved, and the environment is NOT repaired. Passing here would
        # schedule the exact incident the rule exists to prevent.
        adapter = _RecordingAdapter([1])
        runner = make_runner(adapter)

        with self.assertRaises(ActionFailed) as ctx:
            runner.apply_runtime_patches()

        self.assertEqual(ctx.exception.state, "RUNTIME_PATCH_FAILED")
        self.assertIn("patch_vllm_kunlun_drift.py", ctx.exception.reason)
        self.assertIn("NOT in the repaired state", ctx.exception.reason)
        # The failure's evidence was persisted before the gate decided.
        artifact = runner.artifact_dir / "runtime_patches.txt"
        self.assertIn("exit 1", artifact.read_text(encoding="utf-8"))


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


if __name__ == "__main__":
    unittest.main()
