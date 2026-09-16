"""MAT-016 Support Matrix: change the claim only when evidence changes.

The matrix is the one artifact people read instead of reading evidence, so the
entry has to carry what it is true of: which checkpoint revision, which stack
commit, and which conditions the result depended on. An entry that says
"supported" without the launch conditions is how a table becomes something nobody
trusts.

Nothing is inferred. The status comes from recorded facts — a deployment proof and
an accuracy differential — and a missing fact keeps the status where it was.
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
from operations.validation.update_support_matrix import CONTRACT, execute


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--hardware", default="Kunlunxin-3-P800")
    parser.add_argument("--deployment-status", type=Path, help="service proof status.json")
    parser.add_argument("--accuracy", type=Path, help="mat-013 accuracy_differential.json")
    parser.add_argument("--budget-status", type=Path, help="mem-001 budget_status.json")
    parser.add_argument("--plan", type=Path, help="mat-005 deployment_plan.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--apply", action="store_true", help="write catalog/support_matrix.yaml")
    args = parser.parse_args()
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(run_managed_tool(main, task_id=CONTRACT.parent.name))
