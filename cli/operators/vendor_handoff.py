"""MAT-020 Vendor Handoff: produce a ticket the vendor can act on.

A terminal state, not a failure. When a triage lands on a binary layer there is
nothing further to fix here, and the deliverable is a package someone else can
work from: what was called, with which arguments, in which environment, what was
expected, what happened, and — the part usually missing — whether it reproduces
outside the server.

That last field is why this tool exists rather than a wiki page. Qwen3-8B does not
reproduce in isolation, and a ticket that implies it does wastes the vendor's time
and comes back rejected.
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
from operations.operators.vendor_handoff import CONTRACT, execute


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triage", help="mat-006 triage_report.json (single-ticket form)")
    parser.add_argument("--findings", help="findings yaml (package form)")
    parser.add_argument("--environment", help="fingerprint or its digest; taken from the yaml in package form")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(run_managed_tool(main, task_id=CONTRACT.parent.name))
