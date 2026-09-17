"""Read or explicitly rebuild a Task Memory view from its authoritative Journal."""

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
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--run-id")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--show", action="store_true")
    action.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    if args.rebuild and args.journal is None:
        parser.error("--rebuild requires --journal and --run-id")
    if (args.journal is None) != (args.run_id is None):
        parser.error("--journal and --run-id must be supplied together")
    try:
        return execute(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Task Memory: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
