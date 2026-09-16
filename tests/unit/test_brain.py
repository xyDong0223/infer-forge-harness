"""The brain contract: what a decider may answer, and what happens when it lies.

The point of these tests is not the happy path — any parser accepts a valid
answer. The point is that a malformed, out-of-enum, or missing answer can never
translate into a cluster action: it must degrade to BLOCKED or a re-ask. The
LLM-in-the-loop design is only safe if the loop's door only opens on a valid key.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.brain import (  # noqa: E402
    AgentBrain,
    BrainError,
    Decision,
    DecisionRequest,
    FailureEvidence,
    RuleBrain,
    brain_from_config,
)


def request(reason: str = "Check 0 == ret failed") -> DecisionRequest:
    return DecisionRequest(
        model="Qwen3-8B", backend="p800",
        failure=FailureEvidence(node="service_proof", state="DEPLOYMENT_FAILED",
                                reason=reason),
        attempts_remaining=3,
        skill={"id": "environment-proof", "method": {"sha256": "abc"}},
    )


class DecisionContractTest(unittest.TestCase):
    def test_a_valid_decision_parses(self):
        decision = Decision.from_dict({
            "next_action": "RETRY_WITH_PARAMS",
            "diagnosis": "KV cache overcommits HBM at gpu_memory_utilization=0.92",
            "params": {"gpu_memory_utilization": 0.85},
            "confidence": 0.8,
        })
        self.assertEqual(decision.next_action, "RETRY_WITH_PARAMS")
        self.assertEqual(decision.params["gpu_memory_utilization"], 0.85)

    def test_an_unknown_action_is_rejected(self):
        with self.assertRaises(BrainError):
            Decision.from_dict({"next_action": "DELETE_THE_CLUSTER",
                                "diagnosis": "turn it off and on again"})

    def test_a_decision_without_a_diagnosis_is_rejected(self):
        with self.assertRaises(BrainError):
            Decision.from_dict({"next_action": "RETRY", "diagnosis": "  "})

    def test_confidence_out_of_range_is_rejected(self):
        with self.assertRaises(BrainError):
            Decision.from_dict({"next_action": "RETRY", "diagnosis": "x",
                                "confidence": 5})

    def test_the_request_carries_the_failure_and_the_budget(self):
        payload = request().to_dict()
        self.assertEqual(payload["failure"]["reason"], "Check 0 == ret failed")
        self.assertEqual(payload["attempts_remaining"], 3)
        self.assertIn("RETRY", payload["available_actions"])
        self.assertIn("BLOCKED", payload["available_actions"])
        self.assertEqual(payload["skill"]["id"], "environment-proof")


class AgentBrainFileProtocolTest(unittest.TestCase):
    """The decider is an external process answering through files."""

    def setUp(self) -> None:
        import tempfile

        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: _rm(self.tmp))

    def test_a_valid_file_answer_is_returned(self):
        def decider(request_path: str, response_path: str) -> None:
            payload = json.loads(Path(request_path).read_text())
            Path(response_path).write_text(json.dumps({
                "next_action": "RUN_TRIAGE",
                "diagnosis": f"vendor kernel failed: {payload['failure']['reason']}",
                "facts": ["server returned non-zero"],
                "confidence": 0.7,
            }))

        command = ["python3", "-c",
                   "import sys, json, pathlib;"
                   "req, resp = sys.argv[1], sys.argv[2];"
                   "pathlib.Path(resp).write_text(json.dumps({"
                   "'next_action': 'RUN_TRIAGE',"
                   "'diagnosis': 'vendor kernel failed',"
                   "'confidence': 0.7}))"]
        brain = AgentBrain(self.tmp, command=command)
        decision = brain.decide(request())
        self.assertEqual(decision.next_action, "RUN_TRIAGE")

    def test_a_garbled_answer_is_reasked_then_blocked(self):
        # Both attempts write garbage: the brain must stop, not improvise.
        command = ["python3", "-c",
                   "import sys, pathlib;"
                   "pathlib.Path(sys.argv[2]).write_text('not json at all')"]
        brain = AgentBrain(self.tmp, command=command)
        decision = brain.decide(request())
        self.assertEqual(decision.next_action, "BLOCKED")

    def test_the_request_file_records_what_the_decider_sees(self):
        command = ["python3", "-c", "import sys; sys.exit(1)"]
        brain = AgentBrain(self.tmp, command=command)
        brain.decide(request("dummy"))
        requests = sorted(self.tmp.glob("tasks/decision/attempts/*/input/decision_request.json"))
        self.assertEqual(len(requests), 2)
        written = json.loads(requests[0].read_text(encoding="utf-8"))
        self.assertEqual(written["model"], "Qwen3-8B")

    def test_requests_and_responses_survive_reasks_and_new_invocations(self):
        command = ["python3", "-c", "import sys,pathlib;"
                   "pathlib.Path(sys.argv[2]).write_text('[]')"]
        brain = AgentBrain(self.tmp, command=command)
        self.assertEqual(brain.decide(request("first")).next_action, "BLOCKED")
        original = {path: path.read_bytes() for path in self.tmp.rglob("*.json")}
        self.assertEqual(brain.decide(request("second")).next_action, "BLOCKED")
        self.assertEqual(len(list(self.tmp.rglob(AgentBrain.REQUEST_NAME))), 4)
        for path, content in original.items():
            self.assertEqual(path.read_bytes(), content)

    def test_source_workdir_is_rejected_before_decider(self):
        from unittest.mock import patch
        from core.storage import WritePolicyError

        with patch("engine.brain.subprocess.run") as run:
            with self.assertRaises(WritePolicyError):
                AgentBrain(ROOT / "runtime-brain", command=["never"])
        run.assert_not_called()


class RuleBrainTest(unittest.TestCase):
    def test_oom_gets_a_param_retry(self):
        decision = RuleBrain().decide(request("XPU out of memory allocating KV cache"))
        self.assertEqual(decision.next_action, "RETRY_WITH_PARAMS")

    def test_an_unmatched_failure_blocks_instead_of_guessing(self):
        decision = RuleBrain().decide(request("something entirely novel"))
        self.assertEqual(decision.next_action, "BLOCKED")


class BrainConfigTest(unittest.TestCase):
    def test_an_unknown_brain_type_is_an_error(self):
        with self.assertRaises(BrainError):
            brain_from_config({"brain": "oracle"}, Path("/tmp"))


def _rm(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
