"""Shared task execution errors, independent of command-line parsing."""


class ToolFailed(RuntimeError):
    """A task tool failed with a terminal state and a reason."""

    def __init__(self, state: str, reason: str) -> None:
        super().__init__(f"{state}: {reason}")
        self.state = state
        self.reason = reason
