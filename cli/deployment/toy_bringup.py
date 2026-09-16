"""MAT-028 Toy Bring-up: exercise the code path before paying for the weights.

Between "the modules import" (MAT-027) and "the server answers" (KDP-001b) sits a
band of pure contract: abstract methods the engine now requires, a factory whose
return shape changed, a KV-cache tensor whose rank the layer slices wrongly, a MoE
mapping helper that moved to module scope. None of it depends on the real weights.

GLM-5.2 found five such defects, each one at the end of a full 707 GiB load. A
few-layer copy of the same config with dummy weights reaches all five in under a
minute, on the same kernels, because the dimensions that select kernels are kept and
only depth and expert count are shrunk.

Consumes the ModelRequest for the config to copy and the pod from the environment
proof. Produces a position -- which stage was reached -- not an opinion.
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
from operations.deployment.toy_bringup import BringupFailed, CONTRACT, execute


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-request", required=True, help="model_request.yaml from MAT-001")
    parser.add_argument("--pod", help="pod from the environment proof; overrides --env-status")
    parser.add_argument("--env-status", help="status.json of the environment proof")
    parser.add_argument("--layers", type=int, default=4, help="toy depth floor")
    parser.add_argument("--experts", type=int, default=8, help="toy routed expert floor")
    parser.add_argument("--tp-size", type=int, default=1,
                        help="1 keeps it cheap; the planned TP also covers collectives")
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--probe-timeout", type=int, default=1800)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    return execute(args)


if __name__ == "__main__":
    try:
        raise SystemExit(run_managed_tool(main, task_id=CONTRACT.parent.name))
    except BringupFailed as failure:
        print(f"{failure.state}: {failure.reason}", file=sys.stderr)
        raise SystemExit(1) from failure
