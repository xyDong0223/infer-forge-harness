"""Coordinate asynchronous XPU operator work and serialized integration.

This module deliberately does not generate or grade kernels. It creates durable
requests for the xpu-op-gen subagent, freezes a serving baseline, and records
whether one candidate is ready to be integrated against that baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
from cli.common import run_managed_tool
from operations.operators.operator_lifecycle import execute


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    dispatch_parser = sub.add_parser("dispatch")
    dispatch_parser.add_argument("--gaps", type=Path, required=True)
    dispatch_parser.add_argument("--out", type=Path, required=True)
    dispatch_parser.add_argument("--subject", required=True)
    dispatch_parser.add_argument("--baseline-id")

    baseline_parser = sub.add_parser("freeze-baseline")
    baseline_parser.add_argument("--service", type=Path, required=True)
    baseline_parser.add_argument("--accuracy", type=Path, required=True)
    baseline_parser.add_argument("--out", type=Path, required=True)
    baseline_parser.add_argument("--subject", required=True)
    baseline_parser.add_argument("--env", action="append", default=[])

    integration_parser = sub.add_parser("integrate")
    integration_parser.add_argument("--baseline", type=Path, required=True)
    integration_parser.add_argument("--candidate", type=Path)
    integration_parser.add_argument("--out", type=Path, required=True)
    integration_parser.add_argument("--subject", required=True)
    args = parser.parse_args()
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(run_managed_tool(main, task_id="operator-lifecycle"))
