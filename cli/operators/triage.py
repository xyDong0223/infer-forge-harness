"""Execute MAT-006 failure triage as one orchestrated sequence.

The graph runner stopped here with "operator-driven" because the task is not a
single command: instrument the failing call, rerun the service proof so it
fails under instrumentation, pull the real arguments out of the pod, replay
them in isolation to find the boundary, then restore. Each step already had a
tool; what was missing was the sequence — which meant every kernel failure on
a new model ended in a MANUAL stop.

The sequence is fixed by the contract, not chosen here. The only decision this
runner makes is the verdict, and even that is mechanical: the isolated call
failing too is VENDOR_KERNEL_GAP, passing is RUNTIME_STATE_DEPENDENT. The
independent triage validator has the final word before TRIAGE_READY is written.
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
from runners.triage_executor import CONTRACT, run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod", default=None,
                        help="the pod to triage in; defaults to the environment proof's pod")
    parser.add_argument("--contract-instance", type=Path, default=None)
    parser.add_argument("--failure-status", type=Path,
                        help="persisted failed-node status for diagnosis before a deployment plan exists")
    # Accepted (and used as the pod source when --pod is absent) because the
    # graph's failure_triage node resolves its EnvironmentProof input as this
    # flag; refusing it made the node unrunnable from the graph.
    parser.add_argument("--env-status", type=Path, default=None,
                        help="environment proof status.json; supplies --pod when omitted")
    parser.add_argument("--call", default="speculative_attention")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        return run(args)
    except ValueError as error:
        if str(error) != "a pod is required: pass --pod or --env-status":
            raise
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(run_managed_tool(main, task_id=CONTRACT.parent.name))
