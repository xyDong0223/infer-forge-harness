"""Tests for the environment subcommand's attach wiring.

Bug this pins: `environment --contract` had no way to pass task_runner's
`--attach-pod` (Imported Context) through, so every reproof created a new
FedDeployment instead of re-proving the existing prepared Pod.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cli.adaptation import _environment_command, _parser


class TestEnvironmentAttachWiring(unittest.TestCase):
    def test_parser_accepts_attach_pod_alongside_contract(self) -> None:
        args = _parser().parse_args(
            [
                "--state", "/tmp/x.db",
                "environment",
                "--run-id", "run-1",
                "--contract", "/external/generated-environment.yaml",
                "--attach-pod", "dongxinyu03-vllm-abc-0",
            ]
        )
        self.assertEqual(args.attach_pod, "dongxinyu03-vllm-abc-0")

    def test_command_forwards_attach_pod_to_task_runner(self) -> None:
        command = _environment_command(
            Path("contract.yaml"), Path("/artifacts"), "dongxinyu03-vllm-abc-0"
        )
        self.assertIn("--attach-pod", command)
        self.assertEqual(
            command[command.index("--attach-pod") + 1], "dongxinyu03-vllm-abc-0"
        )
        self.assertTrue(command[1].endswith("cli/deployment/proof.py"))
        self.assertIn("--execute", command)

    def test_without_attach_pod_the_flag_is_absent(self) -> None:
        command = _environment_command(Path("contract.yaml"), None, None)
        self.assertNotIn("--attach-pod", command)
        self.assertNotIn("--artifact-dir", command)

    def test_attach_pod_ignored_for_imported_status(self) -> None:
        # --status imports a finished proof; no pod is created either way,
        # and the parser must not make the two sources mutually exclusive
        # with the attach option.
        args = _parser().parse_args(
            ["--state", "/tmp/x.db", "environment", "--run-id", "run-1",
             "--status", "status.json"]
        )
        self.assertIsNone(args.attach_pod)


if __name__ == "__main__":
    unittest.main()
