"""Sequence the three correctness tasks that used to stop the walk as MANUAL.

mat-021, mat-022 and mat-023 share one shape: exercise the real path, produce an
independent reference, compare, and prove with a discriminating control that the
comparison could have failed. The probes and tools already did the hard parts —
`sliding_window_decode_probe` and `block_sparse_attention_probe` compute CPU
float32 references and negative controls in the pod; `accuracy_differential`
compares the served model against transformers on CPU. What was missing was the
sequence and the packaging into each contract's evidence shape, which is what
kept all three in the MANUAL set and every correctness question in a person's
queue.

The verdicts stay mechanical and the validators stay independent: a control that
does not discriminate is AMBIGUOUS, never a pass; a probe that cannot produce
the tensors grades nothing.
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
from runners.correctness_executor import run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("kernel", "end-to-end", "long-context"))
    parser.add_argument("--pod", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-request", type=Path, default=None)
    parser.add_argument("--served-model-name", default="")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--context-len", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--max-relative-l2", type=float, default=0.02)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(run_managed_tool(main, task_id="correctness-executor"))
