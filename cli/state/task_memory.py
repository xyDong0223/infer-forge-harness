"""Durable Task Loop memory for agent-driven execution.

The Task remains the stable goal. Each graph node or manual investigation is a
Loop Block with a local target, exit condition, and execution record. This
module intentionally stores only coordination state; evidence stays in the
artifact directories and the Journal remains the source of fact provenance.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
from engine.state.task_memory import execute


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
