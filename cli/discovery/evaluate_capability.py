"""MAT-008 Capability Evaluation: run the capability, do not read about it.

MAT-003 grades static evidence — `REGISTRY` beats `MODULE` beats `ABSENT` — and
Qwen3-8B is the standing proof that the top of that scale is not support: every
axis matched and the server died in a decode kernel. This Task adds the tier above
`REGISTRY`: `EXERCISED`, meaning the capability ran on this hardware and its
numbers were compared against a reference that does not share the accelerator.

One contract covers four dimensions rather than four near-identical contracts,
because the method is the same in each: take the smallest unit that still carries
the real convention risk, run the operators the serving path uses, compare against
a float32 reference, and prove the comparison could have failed.

`--list-dimensions` is what the graph fans out over. It reads the CapabilityMatch
and names only the dimensions this model actually demands, so a dense bf16 model
does not get a quantization verdict it never needed.
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
from operations.discovery.evaluate_capability import CONTRACT, EvaluationFailed, execute


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capability-match", help="capability_match.json from MAT-003")
    parser.add_argument("--list-dimensions", action="store_true",
                        help="print the dimensions this model demands, as JSON")
    parser.add_argument("--aggregate", action="store_true", help="fan-in over --child dirs")
    parser.add_argument("--child", action="append", type=Path, default=[])
    parser.add_argument("--dimension", choices=["quantization", "swa", "moe", "multimodal"])
    parser.add_argument("--subject")
    parser.add_argument("--pod")
    parser.add_argument("--model-path", help="weights to exercise; may differ from the subject")
    parser.add_argument("--tensor", help="override the contract's tensor")
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    return execute(args)


if __name__ == "__main__":
    try:
        raise SystemExit(run_managed_tool(main, task_id=CONTRACT.parent.name))
    except EvaluationFailed as failure:
        print(f"{failure.state}: {failure.reason}", file=sys.stderr)
        raise SystemExit(1) from failure
