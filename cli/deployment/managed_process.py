"""Standalone process protocol; exit zero is not task PASS or Pod readiness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from operations.deployment import managed_process


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def main():
    parser = _Parser(description=__doc__)
    parser.add_argument("command", choices=("start", "inspect", "cancel", "_monitor"),
                        metavar="{start,inspect,cancel}")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--request", type=Path)
    try:
        args = parser.parse_args()
        if args.command == "_monitor":
            if args.request is not None:
                raise ValueError("internal monitor reads only its reserved request")
            return managed_process.monitor(args.root)
        if args.request is None:
            raise ValueError("--request is required")
        request = managed_process.load_request(args.request)
        result = getattr(managed_process, args.command)(args.root, request)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError) as error:
        print(json.dumps({"protocol_status": "REJECTED", "error_type": type(error).__name__,
                          "error": str(error), "task_verdict": None, "ledger_integrated": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
