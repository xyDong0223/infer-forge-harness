"""MAT-002 Model Scan: what will actually run this architecture, in this pod.

Scans the installed runtime rather than a checkout, because the four override
mechanisms in vLLM-Kunlun (module redirection, post-import patching, OOT
registration, install-time overwrite) mean a repository and an installation can
disagree about what is registered.

Consumes the ModelRequest from MAT-001 and the pod from the environment proof, so
it needs no pod of its own and no launch parameter.
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
from operations.discovery.scan_model_support import CONTRACT, PROXY, ScanFailed, execute


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-request", required=True, help="model_request.yaml from MAT-001")
    parser.add_argument("--pod", help="pod from the environment proof; overrides --env-status")
    parser.add_argument("--env-status", help="status.json of the environment proof")
    parser.add_argument("--proxy", default=PROXY, help="in-pod proxy for the upstream lookups")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    return execute(args)


if __name__ == "__main__":
    try:
        raise SystemExit(run_managed_tool(main, task_id=CONTRACT.parent.name))
    except ScanFailed as failure:
        print(f"{failure.state}: {failure.reason}", file=sys.stderr)
        raise SystemExit(1) from failure
