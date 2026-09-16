"""Archived drift repair semantics; deployment never replays this script."""

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



if __name__ == "__main__":
    unittest.main()
