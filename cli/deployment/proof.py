"""Plan-first task runner for infer-forge-harness."""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
from runners.task_runner import run


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or plan an inference engineering task")
    parser.add_argument("contract", type=Path)
    parser.add_argument("--target", type=Path, help="Requested target; must agree with the contract")
    parser.add_argument("--subject", help="Bind the model identity of an unbound --target")
    parser.add_argument("--execute", action="store_true", help="Run against the real cluster")
    parser.add_argument("--output", type=Path, help="Where to write the plan (plan mode)")
    parser.add_argument("--artifact-dir", type=Path, help="External run root, or allocated attempt output/")
    parser.add_argument("--run-id", help="Durable run identity for allocated attempts")
    parser.add_argument("--user-id", help="Resource owner's user ID, supplied by the user (legacy fallback: USER_ID)")
    parser.add_argument(
        "--phase",
        choices=["all", "environment", "service"],
        default="environment",
        help="Default: environment (MiniMax baseline only). Use service after target adaptation; "
             "all explicitly selects legacy standalone deployment. A contract's task_type overrides this",
    )
    parser.add_argument(
        "--attach-pod",
        help="Prove against an already prepared Pod instead of creating one (Imported Context)",
    )
    parser.add_argument(
        "--server-log",
        help="Override execution.server_log with a path of this attempt's own. "
             "A reproof (e.g. the MAT-006 triage rerun) must not truncate the "
             "log of the attempt it exists to explain.",
    )
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
