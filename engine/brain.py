"""The decision contract that turns a failed node into a next action.

The harness already knew *how* to run things; what it never had was the piece
that decides what to do when a run fails. Every failure edge either stopped the
walk (`MANUAL`, `NEEDS_HUMAN`) or produced a ticket for someone else. That is
the gap this module closes: a failure becomes a structured `DecisionRequest`,
a Brain answers with a structured `Decision`, and the recovery controller
executes it — under a budget, with the evidence gates untouched.

The contract is deliberately transport-agnostic. The LLM does not live in this
process: `AgentBrain` writes the request to disk and reads the answer back, so
the decider can be a model API call, an agent session, or an operator with an
editor. What is NOT negotiable is the response schema — a malformed answer is
rejected and re-asked, and after that the loop stops, because an unparseable
brain must never translate into an unvalidated cluster action.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from core.storage import ArtifactStore, RunPaths, WritePolicyError, ensure_external

# Aligned with the Diagnosis Agent contract in AGENTS.md. Each value maps to an
# executor the recovery controller owns; a Brain cannot invent new ones.
NEXT_ACTIONS = frozenset({
    "RETRY",                    # rerun the failed node unchanged
    "RETRY_WITH_PARAMS",        # rerun with context overrides (params)
    "RUN_TRIAGE",               # mat-006: capture the failing call's real arguments
    "PLACE_PATCH",              # mat-007: install the reversible torch fallback
    "DISPATCH_OPERATOR_TASK",   # durable operator request, bring-up continues
    "REDISCOVER",               # rerun operator discovery from new evidence
    "ROLLBACK",                 # delete resources this run created
    "BLOCKED",                  # terminal: no autonomous action is justified
})

# Decisions whose action is "change something, then rerun the node".
RERUN_ACTIONS = frozenset({"RETRY", "RETRY_WITH_PARAMS"})


class BrainError(RuntimeError):
    """A Brain answer that does not satisfy the contract."""


@dataclass
class FailureEvidence:
    """What the failed node left behind, in the order a decider should read it."""

    node: str
    state: str
    reason: str
    artifacts: list[str] = field(default_factory=list)
    environment: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node, "state": self.state, "reason": self.reason,
            "artifacts": list(self.artifacts), "environment": dict(self.environment),
        }


@dataclass
class DecisionRequest:
    """Everything a decider is allowed to know before choosing an action."""

    model: str
    backend: str
    failure: FailureEvidence
    # Prior decisions and their outcomes for this node. Without it a decider
    # will happily re-prescribe the exact repair that just failed.
    history: list[dict[str, Any]] = field(default_factory=list)
    available_actions: list[str] = field(default_factory=lambda: sorted(NEXT_ACTIONS))
    attempts_remaining: int = 0
    # Current node context (launch parameters, pod name, paths). A decider
    # prescribing RETRY_WITH_PARAMS can only vary what is actually here.
    context: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model, "backend": self.backend,
            "failure": self.failure.to_dict(), "history": list(self.history),
            "available_actions": list(self.available_actions),
            "attempts_remaining": self.attempts_remaining,
            "context": dict(self.context),
        }


@dataclass
class Decision:
    """A decider's answer. Facts and hypotheses are separated, not mixed prose."""

    next_action: str
    diagnosis: str
    facts: list[str] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    evidence_refs: list[str] = field(default_factory=list)

    @classmethod
    def blocked(cls, reason: str) -> "Decision":
        return cls(next_action="BLOCKED", diagnosis=reason, confidence=1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "next_action": self.next_action, "diagnosis": self.diagnosis,
            "facts": list(self.facts), "hypotheses": list(self.hypotheses),
            "params": dict(self.params), "confidence": self.confidence,
            "evidence_refs": list(self.evidence_refs),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Decision":
        """Parse and enforce the contract. Raises BrainError on violation."""
        if not isinstance(payload, dict):
            raise BrainError("decision must be an object")
        action = payload.get("next_action")
        if not isinstance(action, str) or action not in NEXT_ACTIONS:
            raise BrainError(
                f"next_action {action!r} is not one of {sorted(NEXT_ACTIONS)}"
            )
        diagnosis = payload.get("diagnosis")
        if not isinstance(diagnosis, str) or not diagnosis.strip():
            raise BrainError("diagnosis must be a non-empty string")
        params = payload.get("params", {})
        if not isinstance(params, dict):
            raise BrainError("params must be an object")
        confidence = payload.get("confidence", 0.0)
        if not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
            raise BrainError("confidence must be a number in [0, 1]")
        for key in ("facts", "hypotheses", "evidence_refs"):
            if not isinstance(payload.get(key, []), list):
                raise BrainError(f"{key} must be a list")
        return cls(
            next_action=action,
            diagnosis=diagnosis,
            facts=list(payload.get("facts", [])),
            hypotheses=list(payload.get("hypotheses", [])),
            params=params,
            confidence=float(confidence),
            evidence_refs=list(payload.get("evidence_refs", [])),
        )


class Brain(Protocol):
    def decide(self, request: DecisionRequest) -> Decision: ...


class AgentBrain:
    """Delegate the decision to an external decider through files.

    The decider receives the full request as JSON and must write a valid
    Decision back. Two modes:

    - ``command``: a subprocess that reads the request path as its first
      argument and writes the response path. Wrap any model API here.
    - file mode: the request is written and we wait for the response file to
      appear. An agent session (or a human, in the worst case) is the decider.

    A response that fails ``Decision.from_dict`` is re-asked once, then the
    brain returns BLOCKED: a garbled answer must degrade to "stop", never to
    a guessed cluster action.
    """

    REQUEST_NAME = "decision_request.json"
    RESPONSE_NAME = "decision.json"

    def __init__(self, workdir: Path, command: list[str] | None = None,
                 poll_seconds: float = 2.0, timeout_seconds: float = 600.0):
        self.workdir = ensure_external(workdir)
        self.command = command
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds

    def decide(self, request: DecisionRequest) -> Decision:
        paths = RunPaths(self.workdir)
        for attempt in range(2):
            workspace = paths.allocate_attempt("decision")
            request_path = workspace.input / self.REQUEST_NAME
            response_path = workspace.output / self.RESPONSE_NAME
            request_path.write_text(
                json.dumps(request.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            store = ArtifactStore(workspace.root)
            if self.command:
                try:
                    result = subprocess.run(
                        self.command + [str(request_path), str(response_path)],
                        text=True, capture_output=True, timeout=self.timeout_seconds,
                        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                    )
                except (OSError, subprocess.TimeoutExpired) as error:
                    (workspace.logs / "error.txt").write_text(str(error), encoding="utf-8")
                    store.register(identity=workspace.identity, outcome="BLOCKED")
                    if attempt == 0:
                        continue
                    return Decision.blocked(f"decider command failed: {error}")
                (workspace.logs / "stdout.log").write_text(result.stdout, encoding="utf-8")
                (workspace.logs / "stderr.log").write_text(result.stderr, encoding="utf-8")
                if result.returncode != 0:
                    store.register(identity=workspace.identity, outcome="BLOCKED")
                    if attempt == 0:
                        continue
                    return Decision.blocked(
                        f"decider command failed: {result.stderr.strip()[-300:]}"
                    )
            else:
                if not self._await(response_path):
                    store.register(identity=workspace.identity, outcome="BLOCKED")
                    return Decision.blocked(
                        f"no decision.json appeared within {self.timeout_seconds}s"
                    )
            try:
                store.path(response_path.relative_to(workspace.root))
                payload = json.loads(response_path.read_text(encoding="utf-8"))
                decision = Decision.from_dict(payload)
                store.register(identity=workspace.identity, outcome=decision.next_action)
                return decision
            except WritePolicyError:
                raise
            except (OSError, ValueError, BrainError) as error:
                store.register(identity=workspace.identity, outcome="BLOCKED")
                if attempt == 0:
                    continue
                return Decision.blocked(f"decider response rejected: {error}")
        return Decision.blocked("decider did not answer")  # unreachable, kept for clarity

    def _await(self, response_path: Path) -> bool:
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            if response_path.exists():
                return True
            time.sleep(self.poll_seconds)
        return False


# Known failure shapes and their cheapest justified repair. This is not the
# intelligence — it is the floor the loop never falls below when no decider is
# reachable, and the deterministic reference for tests.
_RULES: list[tuple[tuple[str, ...], str, dict[str, Any]]] = [
    (("oom", "out of memory", "cuda oom", "xpu oom"), "RETRY_WITH_PARAMS",
     {"gpu_memory_utilization": "lower"}),
    (("readiness", "timed out", "timeout", "deadline"), "RETRY", {}),
    (("importerror", "modulenotfound", "attributeerror", "cannot import"), "RUN_TRIAGE", {}),
    (("not implemented", "capabilit", "operator", "kernel not found"), "DISPATCH_OPERATOR_TASK", {}),
    (("numerical", "mismatch", "cosine", "accuracy"), "RUN_TRIAGE", {}),
]


class RuleBrain:
    """Deterministic classifier: the safety net, never the feature."""

    def __init__(self, rules: list[tuple[tuple[str, ...], str, dict[str, Any]]] | None = None):
        self.rules = rules or _RULES

    def decide(self, request: DecisionRequest) -> Decision:
        text = f"{request.failure.reason} {request.failure.state}".lower()
        for needles, action, params in self.rules:
            if any(needle in text for needle in needles):
                return Decision(
                    next_action=action,
                    diagnosis=f"rule matched on failure text: {needles[0]!r}",
                    params=params, confidence=0.3,
                )
        return Decision.blocked("no rule matched the failure text")


def brain_from_config(config: dict[str, Any], workdir: Path) -> Brain:
    """Build the configured brain. Unknown type is an error, not a fallback."""
    kind = config.get("brain", "rule")
    if kind == "rule":
        return RuleBrain()
    if kind == "agent":
        command = config.get("decide_command")
        if isinstance(command, str):
            command = command.split()
        return AgentBrain(workdir=workdir, command=command,
                          timeout_seconds=float(config.get("decide_timeout", 600)))
    raise BrainError(f"unknown brain type {kind!r}: expected 'rule' or 'agent'")


def decide_with_budget(brain: Brain, request: DecisionRequest) -> Decision:
    """Guard the brain against answers the budget cannot pay for."""
    decision = brain.decide(request)
    if request.attempts_remaining <= 0 and decision.next_action != "BLOCKED":
        return Decision.blocked("recovery budget exhausted before this decision")
    return decision
