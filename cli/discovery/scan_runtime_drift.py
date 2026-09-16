"""MAT-027 Runtime Drift Scan: does the plugin still load against this engine.

The cheapest gate in the graph, and the one whose absence cost the most. A model
scan answers "is this architecture registered"; it does not answer "does the
registered code import here". When the installed engine has moved past the version
the plugin targets, every drifted symbol is a separate ImportError that surfaces
only when something imports that module -- and for an attention backend that means
after the weights are loaded. GLM-5.2 paid for six of them one full 707 GiB load at
a time before anyone ran this check.

Imports every module of the plugin package in one pass, and for each failure indexes
the engine's own source to say where the missing symbol lives now. Needs the pod
(the plugin is only installed there) and nothing else: no launch parameter, no
weights, no XPU.
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
from operations.discovery.scan_runtime_drift import CONTRACT, DriftScanFailed, execute


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod", help="pod from the environment proof; overrides --env-status")
    parser.add_argument("--env-status", help="status.json of the environment proof")
    parser.add_argument("--plugin", default="vllm_kunlun", help="installed plugin package")
    parser.add_argument("--engine", default="vllm", help="installed engine package")
    parser.add_argument("--probe-timeout", type=int, default=900)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    return execute(args)


if __name__ == "__main__":
    try:
        raise SystemExit(run_managed_tool(main, task_id=CONTRACT.parent.name))
    except DriftScanFailed as failure:
        print(f"{failure.state}: {failure.reason}", file=sys.stderr)
        raise SystemExit(1) from failure
