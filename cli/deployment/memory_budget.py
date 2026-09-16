"""Reconcile a vLLM-Kunlun startup log against real P800 device counters.

The upstream capacity planner
(https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS, skill
`llm-serving-capacity-planner`) already decomposes a vLLM/SGLang startup log
into weights / KV pool / graph capture. It cannot do two things on Kunlun:

1. it has no P800 entry in `references/gpu-specs.json`, so total HBM is guessed;
2. it expects `nvidia-smi`, which does not exist in a P800 container.

This Tool supplies both from the cluster — `xpu_smi` rendered in nvidia-smi CSV
shape and the device facts from `catalog/xpu_specs.yaml` — then reconciles the
log-derived categories against what the cards actually report. It only collects
and computes; the pass/fail decision belongs to `validators/memory_validator.py`.
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
from operations.deployment.memory_budget import (
    BudgetError,
    CONTRACT,
    DEFAULT_IDLE_THRESHOLD_MIB,
    execute,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod", help="pod to snapshot; omit only with --xpu-smi-file")
    parser.add_argument("--server-log", help="in-pod server log path to pull")
    parser.add_argument("--log-file", help="local server log, instead of pulling from the pod")
    parser.add_argument("--xpu-smi-file", help="pre-captured `xpu_smi -m` output")
    parser.add_argument("--analyzer", help="capacity analyzer script or its repository root")
    parser.add_argument("--device", default="p800")
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="reconcile only this card (must be active); default reconciles every active rank",
    )
    parser.add_argument(
        "--idle-threshold-mib",
        type=int,
        default=DEFAULT_IDLE_THRESHOLD_MIB,
        help="cards below this usage are idle and excluded from the reconciliation scope",
    )
    parser.add_argument("--out", required=True, help="evidence directory to write")
    args = parser.parse_args()
    return execute(args)


if __name__ == "__main__":
    try:
        raise SystemExit(run_managed_tool(main, task_id=CONTRACT.parent.name))
    except BudgetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
