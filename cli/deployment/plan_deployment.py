"""MAT-005 Resource & Deployment Plan: derive launch parameters, with sources.

Every parameter carries where it came from. That is the whole point: the launch
parameters were the one thing in this repository still being typed by hand, and a
typed parameter is indistinguishable from a measured one once it lands in a
contract.

Derivations use facts already established upstream — the checkpoint's own dtype
and position limit from MAT-001, weight bytes from the fingerprint, device memory
from the catalog, and the patch's limitations from MAT-007 (a shape-dynamic
fallback forces eager execution). Where no fact supports a value, the parameter is
marked `assumption` so a reader can tell the difference.
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
from operations.deployment.plan_deployment import CONTRACT, execute


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-request", required=True)
    parser.add_argument("--classification", required=True, help="mat-004 gap_classification.json")
    parser.add_argument("--placed-patch", help="mat-007 placement report, if one exists")
    parser.add_argument("--device", default="p800")
    parser.add_argument("--out", required=True)
    parser.add_argument("--user-id", help="Resource owner's user ID supplied by the user (legacy fallback: USER_ID)")
    parser.add_argument(
        "--runtime-artifact-root", type=Path,
        help="external run root for downstream service attempts (not the planner output)",
    )
    args = parser.parse_args()
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(run_managed_tool(main, task_id=CONTRACT.parent.name))
