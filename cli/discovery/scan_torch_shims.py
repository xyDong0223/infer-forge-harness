"""MAT-029 Shim Handoff: no torch shim serves without an operator request.

A torch shim that stands in for a vendor kernel is a legitimate way to keep
bring-up moving -- and a silent way to ship un-optimised hot-path arithmetic
forever. During GLM-5.2's adaptation, three shims
(kv_spans_from_batches, kunlun_convert_req_index_to_global_index,
kunlun_concat_and_cache_mla) were written mid-loop, the service went green, and
not one of them became an operator request. The dispatch path (MAT-024) existed
the whole time; nothing connected the place shims are born to it.

This tool closes that path. The in-pod probe nets candidate shims out of the
installed plugin; every signal must be explained by the shim registry --
declared, dispatched, or waived with a reason -- and every non-waived entry is
immediately turned into a durable operator request through the same
operator_lifecycle dispatch MAT-024 uses, so the candidate-integration loop
(MAT-026) can pick it up later. HANDOFF_FOUND is a work list, not a stop.
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
from operations.discovery.scan_torch_shims import CONTRACT, ShimScanFailed, execute


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod", help="pod from the environment proof; overrides --env-status")
    parser.add_argument("--env-status", help="status.json of the environment proof")
    parser.add_argument("--plugin", default="vllm_kunlun", help="installed plugin package")
    parser.add_argument(
        "--registry",
        help="agent-maintained shim registry JSON (entries: name, location, replaced_kernel, "
        "call_frequency, semantics_basis, status, reason)",
    )
    parser.add_argument("--subject", help="subject for operator requests (default: plugin name)")
    parser.add_argument("--probe-timeout", type=int, default=600)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    return execute(args)


if __name__ == "__main__":
    try:
        raise SystemExit(run_managed_tool(main, task_id=CONTRACT.parent.name))
    except ShimScanFailed as failure:
        print(f"{failure.state}: {failure.reason}", file=sys.stderr)
        raise SystemExit(1) from failure
