"""MAT-013 Accuracy Differential: is the served model still computing the model.

A deployment proof shows the server answers. It does not show the answer is right,
and a wrong attention kernel produces fluent text — which is exactly the risk after
replacing a decode kernel with a torch fallback.

The comparison is at the distribution, not the wording. One forward pass over a
fixed prompt, greedy, and the top-k next-token distribution from the server is
compared against a reference implementation of the same checkpoint. Comparing
generated strings would be both weaker (a rank-3 swap deep in a sentence is
invisible) and unstable (any sampling difference reads as a failure).

The reference is `transformers` on CPU, on the same weights: slow, independent of
the accelerator, and therefore the only thing on this cluster that can disagree
with the accelerator in a meaningful way.
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
from operations.validation.accuracy_differential import AccuracyFailed, CONTRACT, execute


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod", required=True)
    parser.add_argument("--model-request", required=True, help="mat-001 model_request.yaml")
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--reference-timeout", type=int, default=3600)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    return execute(args)


if __name__ == "__main__":
    try:
        raise SystemExit(run_managed_tool(main, task_id=CONTRACT.parent.name))
    except AccuracyFailed as failure:
        print(f"{failure.state}: {failure.reason}", file=sys.stderr)
        raise SystemExit(1) from failure
