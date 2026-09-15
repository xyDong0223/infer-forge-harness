"""Shared plumbing for the per-task tool CLIs.

Every tool under `tools/` is one task node: it loads its task contract, runs
probes through the adapter, and writes a state file. The failure type here is
the one shape all of them raise — `state` names the terminal state the runner
should route to, `reason` is the human-readable cause. Subclass it with the
task-specific name; the (state, reason) contract must not grow per-file copies
of the same body.
"""

from __future__ import annotations


class ToolFailed(RuntimeError):
    """A task tool failed with a terminal state and a reason."""

    def __init__(self, state: str, reason: str) -> None:
        super().__init__(f"{state}: {reason}")
        self.state = state
        self.reason = reason
