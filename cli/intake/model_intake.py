"""MAT-001 Model Intake: pin a model identity to a reproducible revision.

Intake answers exactly one question — *which* model, *which* revision, *where*,
against *which* stack commit — and nothing about how it will be run. Launch
parameters belong to the deployment plan, so a wrong dtype guess here cannot
contaminate the identity fact.

Two facts cannot be established from this host: the weights live on a PVC that
is only visible inside the cluster, and the stack ref must be resolved to a
commit. So intake creates a throwaway XPU-free Pod, fingerprints the checkpoint
through `tools/probe/model_fingerprint_probe.py`, removes the Pod, and resolves
the ref with `git ls-remote`.
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
from operations.intake.model_intake import CONTRACT, IntakeFailed, execute


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True, help="e.g. Qwen3-8B")
    parser.add_argument("--model-path", required=True, help="in-container path, e.g. /mnt/cluster/...")
    parser.add_argument("--pvc", default="rapidfs-baige-v3-pvc")
    parser.add_argument("--hardware", default="Kunlunxin-3-P800")
    parser.add_argument("--stack-ref", default="v0.25.1-dev")
    parser.add_argument("--attempt-id", required=True, help="unique per run, e.g. 20260907-1")
    parser.add_argument("--out", required=True)
    parser.add_argument("--attach-pod", help="reuse a ready owned pod instead of creating one")
    parser.add_argument("--image", default="iregistry.baidu-int.com/hac_test/aiak-inference-llm:vLLM-Kunlun-Base")
    parser.add_argument("--dedicated-pool", default="aihcq-sirzhpwo0g1u")
    parser.add_argument("--ttl-seconds", type=int, default=900)
    parser.add_argument("--ready-timeout", type=int, default=300)
    parser.add_argument("--probe-timeout", type=int, default=1800)
    args = parser.parse_args()
    return execute(args)


if __name__ == "__main__":
    try:
        raise SystemExit(run_managed_tool(main, task_id=CONTRACT.parent.name))
    except IntakeFailed as failure:
        print(f"{failure.state}: {failure.reason}", file=sys.stderr)
        raise SystemExit(1) from failure
