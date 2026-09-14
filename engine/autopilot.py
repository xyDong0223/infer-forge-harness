"""Model-driven autonomous adaptation loop.

The loop deliberately keeps decisions data-driven: workflows provide tools and
gates, while an agent (or policy) chooses the next action from observations.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable
from pathlib import Path
import json


@dataclass
class Observation:
    kind: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class AutopilotState:
    goal: dict[str, Any]
    observations: list[Observation] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    status: str = "running"
    next_action: str | None = None


class AutopilotRunner:
    """Observe/plan/act/validate loop with bounded autonomous recovery."""
    def __init__(self, goal: dict[str, Any], tools: dict[str, Callable], policy: Callable | None = None, max_attempts: int = 100):
        self.state = AutopilotState(goal)
        self.tools, self.policy, self.max_attempts = tools, policy, max_attempts

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.state.__dict__, default=lambda o: o.__dict__, ensure_ascii=False, indent=2))

    def load(self, path: str | Path) -> bool:
        p = Path(path)
        if not p.exists(): return False
        data = json.loads(p.read_text())
        self.state = AutopilotState(data["goal"], [Observation(**o) for o in data.get("observations", [])], data.get("attempts", []), data.get("status", "running"), data.get("next_action"))
        return True

    def run(self) -> AutopilotState:
        preflight = self.tools.get("preflight")
        if preflight:
            try: self.state.observations.append(Observation("preflight", preflight(self.state)))
            except Exception as exc:
                self.state.observations.append(Observation("external_block", {"error": str(exc)}))
                self.state.status = "blocked"; return self.state
        while self.state.status == "running" and len(self.state.attempts) < self.max_attempts:
            action = self.policy(self.state) if self.policy else self._default_action()
            self.state.next_action = action
            self.state.observations.append(Observation("checkpoint", {"action": action, "attempt": len(self.state.attempts) + 1}))
            if action in {"complete", "blocked"}:
                self.state.status = "succeeded" if action == "complete" else "blocked"
                break
            fn = self.tools.get(action)
            if fn is None:
                self.state.observations.append(Observation("missing_tool", {"action": action}))
                self.state.status = "blocked"; break
            try:
                result = fn(self.state)
                self.state.attempts.append({"action": action, "ok": True, "result": result})
                self.state.observations.append(Observation(action, result if isinstance(result, dict) else {"value": result}))
            except Exception as exc:  # noqa: BLE001 - failures become evidence for the next decision
                category = self._classify_failure(str(exc))
                self.state.attempts.append({"action": action, "ok": False, "error": str(exc)})
                self.state.observations.append(Observation("failure", {"action": action, "error": str(exc), "category": category}))
                rollback = self.tools.get("rollback")
                if rollback:
                    try: rollback(self.state)
                    except Exception as rollback_exc: self.state.observations.append(Observation("rollback_failure", {"error": str(rollback_exc)}))
        if self.state.status == "running": self.state.status = "blocked"
        return self.state

    @staticmethod
    def _classify_failure(message: str) -> str:
        text = message.lower()
        for key, words in {"infra": ("kube", "permission", "namespace", "pvc"), "memory": ("oom", "memory"), "build": ("compile", "build"), "timeout": ("timeout", "timed out")}.items():
            if any(w in text for w in words): return key
        return "runtime"

    def _default_action(self) -> str:
        if not self.state.observations: return "profile_environment"
        kinds = {o.kind for o in self.state.observations}
        if "failure" in kinds:
            failures = sum(1 for o in self.state.observations if o.kind == "failure")
            if failures < 4: return "adapt"
        if "profile_model" not in kinds: return "profile_model"
        if "adapt" not in kinds: return "adapt"
        if "deployment" not in kinds: return "deployment"
        if "validate" not in kinds: return "validate"
        return "complete"

    @classmethod
    def from_registry(cls, goal: dict[str, Any], registry: dict[str, Callable], **kwargs):
        return cls(goal, registry, **kwargs)
