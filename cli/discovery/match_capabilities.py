"""MAT-003 Capability Match: what the model demands vs what the installation provides.

Consumes the ModelRequest from MAT-001, the ModelSupportCard from MAT-002 and the
pod from the environment proof. Produces a per-axis match with graded evidence.

It deliberately cannot conclude support. Qwen3-8B matches on every axis and still
dies in an attention kernel during decode warmup, so the deliverable is a set of
candidate requirements for MAT-004 to classify — not a verdict on whether the
model runs.
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
from operations.discovery.match_capabilities import CONTRACT, MatchFailed, execute


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-request", required=True, help="model_request.yaml from MAT-001")
    parser.add_argument("--support-card", required=True, help="model_support.json from MAT-002")
    parser.add_argument("--pod")
    parser.add_argument("--env-status", help="status.json of the environment proof")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    return execute(args)


if __name__ == "__main__":
    try:
        raise SystemExit(run_managed_tool(main, task_id=CONTRACT.parent.name))
    except MatchFailed as failure:
        print(f"{failure.state}: {failure.reason}", file=sys.stderr)
        raise SystemExit(1) from failure
