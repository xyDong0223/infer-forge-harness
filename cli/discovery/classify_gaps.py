"""MAT-004 Gap Classification: turn two static findings into a next action.

Consumes MAT-002's support verdict and MAT-003's capability match. Produces a
class, because different classes have completely different next steps: a missing
registration means writing a model, a version lag means upgrading or
cherry-picking, a missing capability means an operator, and finding nothing
statically means the answer is only obtainable by running the thing.

That last class is the one worth naming. Qwen3-8B has no static gap at all and
still fails at runtime, so `NO_STATIC_GAP` explicitly routes to a deployment
attempt and then to triage — it is not a claim that the model works.
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
from operations.discovery.classify_gaps import CONTRACT, execute


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-card", required=True, help="mat-002 model_support.json")
    parser.add_argument("--capability-match", required=True, help="mat-003 capability_match.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(run_managed_tool(main, task_id=CONTRACT.parent.name))
