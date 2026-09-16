"""Tests for the shared tool-CLI plumbing in cli/common.py."""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.errors import ToolFailed


class TestToolFailed(unittest.TestCase):
    def test_carries_state_and_reason(self) -> None:
        error = ToolFailed("NEEDS_HUMAN", "probe produced no JSON")
        self.assertEqual(error.state, "NEEDS_HUMAN")
        self.assertEqual(error.reason, "probe produced no JSON")
        self.assertEqual(str(error), "NEEDS_HUMAN: probe produced no JSON")

    def test_task_subclasses_inherit_the_contract(self) -> None:
        # The per-task names exist so tests and triage can catch one failure
        # kind; they must not re-implement the (state, reason) body.
        from operations.intake.model_intake import IntakeFailed

        error = IntakeFailed("REJECTED", "model path missing")
        self.assertIsInstance(error, ToolFailed)
        self.assertEqual(error.state, "REJECTED")


class TestScriptedPathsResolve(unittest.TestCase):
    """Commands are referenced by string path, so a move that misses a
    reference fails at run time, not at import time. This is the guard that
    failed loud during the archive refactor (three live paths were nearly
    archived with their dead-era neighbors) and now runs on every commit."""

    def test_every_nodes_command_script_exists(self) -> None:
        from runners.graph_runner import NODES

        for node, spec in NODES.items():
            command = spec["command"]
            if command[0] != "python3":
                continue
            with self.subTest(node=node):
                self.assertTrue(
                    (REPO_ROOT / command[1]).exists(),
                    f"{node} runs {command[1]}, which does not exist",
                )

    def test_every_catalog_command_script_exists(self) -> None:
        catalog = (REPO_ROOT / "catalog" / "tool_catalog.yaml").read_text(encoding="utf-8")
        for match in re.finditer(r"command: python3 (\S+\.py)", catalog):
            with self.subTest(command=match.group(1)):
                self.assertTrue(
                    (REPO_ROOT / match.group(1)).exists(),
                    f"tool_catalog references {match.group(1)}, which does not exist",
                )

    def test_every_task_contract_tool_reference_exists(self) -> None:
        for task_yaml in sorted((REPO_ROOT / "tasks").glob("*/task.yaml")):
            text = task_yaml.read_text(encoding="utf-8")
            for match in re.finditer(r"tools/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.py", text):
                with self.subTest(task=task_yaml.parent.name, path=match.group(0)):
                    self.assertTrue(
                        (REPO_ROOT / match.group(0)).exists(),
                        f"{task_yaml} references {match.group(0)}, which does not exist",
                    )


if __name__ == "__main__":
    unittest.main()
