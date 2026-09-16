"""The recovery loop: budget, history, and who is allowed to declare success.

The controller is the only component that may rerun a failed node, and even it
may not call the result recovered — the rerun's own validator state decides.
These tests pin that: recovery on a passing rerun, BLOCKED honoured, budget
exhaustion, and the history the brain must see so it cannot re-prescribe a
repair that already failed.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.brain import BrainError, Decision, DecisionRequest, FailureEvidence  # noqa: E402
from engine.recovery import BLOCKED, RECOVERED, RecoveryController  # noqa: E402
from engine.recovery import default_actions  # noqa: E402


class ScriptedBrain:
    """Returns queued decisions; records every request it was asked."""

    def __init__(self, decisions: list[Decision]):
        self.decisions = list(decisions)
        self.requests: list[DecisionRequest] = []

    def decide(self, request: DecisionRequest) -> Decision:
        self.requests.append(request)
        if not self.decisions:
            return Decision.blocked("script exhausted")
        return self.decisions.pop(0)


def make_request() -> DecisionRequest:
    return DecisionRequest(
        model="Qwen3-8B", backend="p800",
        failure=FailureEvidence(node="service_proof", state="DEPLOYMENT_FAILED",
                                reason="Check 0 == ret failed"),
        context={"gpu_memory_utilization": "0.92", "pod": "kdp-pod-0"},
    )


class RecoveryLoopTest(unittest.TestCase):
    def test_a_retry_that_passes_recovers(self):
        reruns = []

        def rerun(decision):
            reruns.append(decision.params)
            return True, "DEPLOYMENT_READY"

        brain = ScriptedBrain([
            Decision(next_action="RETRY", diagnosis="transient readiness", confidence=0.5),
        ])
        controller = RecoveryController(brain, rerun, actions={}, budget=3)
        outcome = controller.recover(make_request())
        self.assertEqual(outcome.status, RECOVERED)
        self.assertEqual(outcome.final_state, "DEPLOYMENT_READY")
        self.assertEqual(reruns, [{}])

    def test_params_reach_the_rerun(self):
        captured = {}

        def rerun(decision):
            captured.update(decision.params)
            return True, "DEPLOYMENT_READY"

        brain = ScriptedBrain([
            Decision(next_action="RETRY_WITH_PARAMS", diagnosis="hbm overcommitted",
                     params={"gpu_memory_utilization": 0.85}, confidence=0.8),
        ])
        RecoveryController(brain, rerun, actions={}, budget=3).recover(make_request())
        self.assertEqual(captured, {"gpu_memory_utilization": 0.85})

    def test_a_blocked_brain_stops_the_loop(self):
        def rerun(decision):
            self.fail("rerun must not happen after BLOCKED")

        brain = ScriptedBrain([Decision.blocked("unknown failure shape")])
        outcome = RecoveryController(brain, rerun, actions={}, budget=3).recover(make_request())
        self.assertEqual(outcome.status, BLOCKED)

    def test_budget_exhaustion_ends_blocked_not_looping(self):
        def rerun(decision):
            return False, "DEPLOYMENT_FAILED"

        brain = ScriptedBrain([
            Decision(next_action="RETRY", diagnosis="try again", confidence=0.5),
            Decision(next_action="RETRY", diagnosis="try again", confidence=0.5),
            Decision(next_action="RETRY", diagnosis="try again", confidence=0.5),
        ])
        outcome = RecoveryController(brain, rerun, actions={}, budget=3).recover(make_request())
        self.assertEqual(outcome.status, BLOCKED)
        self.assertEqual(len(brain.requests), 3)

    def test_the_brain_sees_history_and_shrinking_budget(self):
        def rerun(decision):
            return False, "DEPLOYMENT_FAILED"

        brain = ScriptedBrain([
            Decision(next_action="RETRY_WITH_PARAMS", diagnosis="one",
                     params={"gpu_memory_utilization": 0.85}, confidence=0.6),
            Decision(next_action="RETRY_WITH_PARAMS", diagnosis="two",
                     params={"gpu_memory_utilization": 0.75}, confidence=0.6),
        ])
        RecoveryController(brain, rerun, actions={}, budget=2).recover(make_request())
        second = brain.requests[1]
        self.assertEqual(second.attempts_remaining, 1)
        self.assertEqual(len(second.history), 1)
        self.assertEqual(second.history[0]["decision"]["params"],
                         {"gpu_memory_utilization": 0.85})
        self.assertEqual(second.history[0]["outcome"], "node_still_failing")
        # Context travels so the decider knows what it may vary.
        self.assertEqual(second.context["gpu_memory_utilization"], "0.92")

    def test_an_action_without_an_executor_stops_the_loop(self):
        def rerun(decision):
            self.fail("no rerun is warranted for an unexecutable action")

        brain = ScriptedBrain([
            Decision(next_action="PLACE_PATCH", diagnosis="vendor gap, fall back", confidence=0.9),
        ])
        outcome = RecoveryController(brain, rerun, actions={}, budget=3).recover(make_request())
        self.assertEqual(outcome.status, BLOCKED)
        self.assertEqual(outcome.attempts[-1]["outcome"], "no_executor_registered")

    def test_a_failing_action_consumes_budget_and_continues(self):
        calls = []

        def failing_action(decision):
            calls.append(decision.next_action)
            raise RuntimeError("triage could not reach the pod")

        def rerun(decision):
            return True, "DEPLOYMENT_READY"

        brain = ScriptedBrain([
            Decision(next_action="RUN_TRIAGE", diagnosis="capture the call", confidence=0.7),
            Decision(next_action="RETRY", diagnosis="state may have settled", confidence=0.4),
        ])
        outcome = RecoveryController(brain, rerun,
                                     actions={"RUN_TRIAGE": failing_action},
                                     budget=3).recover(make_request())
        self.assertEqual(outcome.status, RECOVERED)
        self.assertEqual(calls, ["RUN_TRIAGE"])
        self.assertEqual(outcome.attempts[0]["outcome"], "action_failed")

    def test_budget_zero_is_refused_at_construction(self):
        with self.assertRaises(ValueError):
            RecoveryController(ScriptedBrain([]), lambda d: (False, "x"), {}, budget=0)

    def test_a_malformed_brain_error_surfaces(self):
        class BrokenBrain:
            def decide(self, request):
                raise BrainError("decider unreachable")

        with self.assertRaises(BrainError):
            RecoveryController(BrokenBrain(), lambda d: (False, "x"), {}, budget=1).recover(
                make_request()
            )


def test_action_retries_preserve_previous_output_and_inventory(tmp_path):
    import json

    def run(command, **kwargs):
        out = Path(command[command.index("--out") + 1])
        (out / "report.json").write_text(json.dumps({"command": command}))
        return SimpleNamespace(returncode=0, stdout="fixture output", stderr="")

    actions = default_actions(ROOT, {"subject": "demo", "pod": "fixture"}, tmp_path / "recovery")
    decision = Decision("RUN_TRIAGE", "capture")
    with patch("engine.recovery.subprocess.run", side_effect=run):
        first = actions["RUN_TRIAGE"](decision)
        report = Path(first["artifacts"]) / "report.json"
        original = report.read_bytes()
        second = actions["RUN_TRIAGE"](decision)
    assert first["artifacts"] != second["artifacts"]
    assert report.read_bytes() == original
    assert len(list(tmp_path.rglob("manifest.json"))) == 2


if __name__ == "__main__":
    unittest.main()
