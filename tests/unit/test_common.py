"""Tests for the shared tool-CLI plumbing in tools/common.py."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.common import ToolFailed  # noqa: E402


class TestToolFailed(unittest.TestCase):
    def test_carries_state_and_reason(self) -> None:
        error = ToolFailed("NEEDS_HUMAN", "probe produced no JSON")
        self.assertEqual(error.state, "NEEDS_HUMAN")
        self.assertEqual(error.reason, "probe produced no JSON")
        self.assertEqual(str(error), "NEEDS_HUMAN: probe produced no JSON")

    def test_task_subclasses_inherit_the_contract(self) -> None:
        # The per-task names exist so tests and triage can catch one failure
        # kind; they must not re-implement the (state, reason) body.
        from tools.model_intake import IntakeFailed

        error = IntakeFailed("REJECTED", "model path missing")
        self.assertIsInstance(error, ToolFailed)
        self.assertEqual(error.state, "REJECTED")


if __name__ == "__main__":
    unittest.main()
