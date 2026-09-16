"""Execute MAT-007 patch placement as one orchestrated sequence.

Placing the reversible torch fallback is also not a single command: apply the
patch, rerun the service proof so the server itself validates it, compare the
fallback numerically against the vendor kernel where that kernel works, and
record the placement. The apply tool already guarantees reversibility (a
.kdp_backup and the KDP_DECODE_KERNEL runtime switch); this runner adds the
validation halves the contract demands and stops at PATCH_REJECTED when either
fails — a patch that only makes the server start is a rewrite, not a patch.
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
from runners.patch_executor import CONTRACT, run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod", required=True)
    parser.add_argument("--contract-instance", type=Path, default=None)
    parser.add_argument("--triage", type=Path, default=None,
                        help="triage report with the captured failure geometry")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(run_managed_tool(main, task_id=CONTRACT.parent.name))
