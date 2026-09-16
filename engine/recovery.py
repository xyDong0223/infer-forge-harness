"""The recovery loop: run -> fail -> decide -> act -> rerun, under a budget.

The graph runner walks happy-path edges well; what it never did is come back
from a failed node. `on_failure` edges pointed at tasks the executor refused
to run (the MANUAL set), so a new model — which fails by default — reached the
first interesting decision and stopped to wait for a person.

This controller closes that loop without weakening the evidence gates: the
Brain chooses, an action executor performs, the node is re-run, and only the
node's own validator decides whether the failure is actually gone. The
controller never marks anything recovered by itself.
"""

from __future__ import annotations

import subprocess
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from core.storage import ArtifactStore, RunPaths, ensure_external
from engine.brain import (
    Brain,
    Decision,
    DecisionRequest,
    RERUN_ACTIONS,
    decide_with_budget,
)

# What the caller must provide per action: perform it and return a JSON-able
# result. RETRY is not in the registry — it is the rerun itself.
Action = Callable[[Decision], dict[str, Any]]

RECOVERED = "RECOVERED"
BLOCKED = "BLOCKED"


@dataclass
class RecoveryOutcome:
    status: str
    attempts: list[dict[str, Any]] = field(default_factory=list)
    final_state: str = ""
    last_decision: Decision | None = None
    final_artifacts: str = ""
    recovery_artifacts: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status, "attempts": self.attempts,
            "final_state": self.final_state,
            "final_artifacts": self.final_artifacts,
            "recovery_artifacts": self.recovery_artifacts,
            "last_decision": self.last_decision.to_dict() if self.last_decision else None,
        }


class RecoveryController:
    """Drive one failed node back to a passing validator, or to a clean stop.

    ``rerun`` re-executes the failed node and returns ``(passed, state)``;
    ``actions`` maps every non-RETRY action to an executor. Both are injected
    because the controller must stay independent of how a node runs — graph
    subprocess, deployment runner, or test double.
    """

    def __init__(self, brain: Brain, rerun: Callable[[Decision], tuple[bool, str]],
                 actions: dict[str, Action], budget: int = 3):
        if budget < 1:
            raise ValueError("recovery budget must be at least 1")
        self.brain = brain
        self.rerun = rerun
        self.actions = actions
        self.budget = budget

    def recover(self, request: DecisionRequest) -> RecoveryOutcome:
        history: list[dict[str, Any]] = []
        outcome = RecoveryOutcome(status=BLOCKED, attempts=history,
                                  final_state=request.failure.state)
        remaining = self.budget
        while remaining > 0:
            decision = decide_with_budget(
                self.brain,
                DecisionRequest(
                    model=request.model, backend=request.backend,
                    failure=request.failure, history=list(history),
                    attempts_remaining=remaining, context=request.context,
                ),
            )
            outcome.last_decision = decision
            if decision.next_action == "BLOCKED":
                history.append({"decision": decision.to_dict(), "outcome": "blocked"})
                return outcome
            if decision.next_action in RERUN_ACTIONS:
                # The rerun below IS the action; params travel to it through
                # the caller's rerun closure, which owns the node's command.
                executor = None
            else:
                executor = self.actions.get(decision.next_action)
                if executor is None:
                    # The brain named an action this deployment cannot perform.
                    # That is a contract mismatch: stop rather than improvise.
                    history.append({"decision": decision.to_dict(),
                                    "outcome": "no_executor_registered"})
                    return outcome
            if executor is not None:
                try:
                    result = executor(decision)
                    history.append({"decision": decision.to_dict(),
                                    "outcome": "action_complete", "result": result})
                except Exception as error:  # noqa: BLE001 - the failure is the next input
                    history.append({"decision": decision.to_dict(),
                                    "outcome": "action_failed", "error": str(error)})
                    remaining -= 1
                    continue
            if decision.next_action in RERUN_ACTIONS:
                passed, state = self.rerun(decision)
                outcome.final_state = state
                if passed:
                    outcome.status = RECOVERED
                    history.append({"decision": decision.to_dict(),
                                    "outcome": "node_passed", "state": state})
                    return outcome
                history.append({"decision": decision.to_dict(),
                                "outcome": "node_still_failing", "state": state})
            remaining -= 1
        return outcome


def default_actions(repo_root: Path, context: dict[str, str],
                    run_dir: Path) -> dict[str, Action]:
    """Executors for the actions that are self-contained commands.

    Injected context (subject, pod, artifact paths) is formatted into each
    command, so the Brain's params reach the tools without the controller
    knowing any tool's interface. Actions the caller must wire itself (the
    rerun, rollback against a live adapter) are simply absent; an action
    without an executor stops the loop instead of guessing.
    """

    paths = RunPaths(ensure_external(run_dir) / "actions")

    def _run(command: list[str], action: str) -> dict[str, Any]:
        attempt = paths.allocate_attempt(action)
        command += ["--out", str(attempt.output)]
        (attempt.input / "command.json").write_text(json.dumps(command), encoding="utf-8")
        try:
            result = subprocess.run(command, cwd=repo_root, text=True, capture_output=True,
                                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        except OSError as error:
            (attempt.logs / "error.txt").write_text(str(error), encoding="utf-8")
            ArtifactStore(attempt.root).register(identity=attempt.identity, outcome="BLOCKED")
            raise
        (attempt.logs / "stdout.log").write_text(result.stdout, encoding="utf-8")
        (attempt.logs / "stderr.log").write_text(result.stderr, encoding="utf-8")
        ArtifactStore(attempt.root).register(
            identity=attempt.identity, outcome="EXECUTED" if result.returncode == 0 else "REWORK",
        )
        return {"returncode": result.returncode, "artifacts": str(attempt.output),
                "stdout_tail": result.stdout.strip()[-500:],
                "stderr_tail": result.stderr.strip()[-500:]}

    def run_triage(decision: Decision) -> dict[str, Any]:
        command = ["python3", "cli/operators/triage.py"]
        pod = context.get("pod")
        if pod:
            command += ["--pod", pod]
        return _run(command, "triage")

    def place_patch(decision: Decision) -> dict[str, Any]:
        command = ["python3", "cli/operators/place_patch.py"]
        pod = context.get("pod")
        if pod:
            command += ["--pod", pod]
        return _run(command, "patch")

    def dispatch_operator_task(decision: Decision) -> dict[str, Any]:
        command = ["python3", "cli/operators/operator_lifecycle.py", "dispatch",
                   "--subject", context.get("subject", "")]
        return _run(command, "operator_dispatch")

    def rediscover(decision: Decision) -> dict[str, Any]:
        command = ["python3", "cli/discovery/scan_model_support.py"]
        return _run(command, "rediscover")

    return {
        "RUN_TRIAGE": run_triage,
        "PLACE_PATCH": place_patch,
        "DISPATCH_OPERATOR_TASK": dispatch_operator_task,
        "REDISCOVER": rediscover,
    }
