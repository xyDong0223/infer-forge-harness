"""Instrument a vendor kernel call site so a server failure yields its arguments.

The server reports `Check 0 == ret failed` from inside the engine and nothing
about the tensors involved, which makes the failure impossible to reproduce. This
wraps the call, appends the real arguments to a JSONL file when it raises, and
re-raises unchanged.

It edits an installed file, so it always writes a `.kdp_backup` beside it and
`--restore` puts the original back. Instrumentation is a debugging state, not a
deployment state: restore before any measurement is taken.
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.kunlun_p800.adapter import KunlunP800Adapter  # noqa: E402

DEFAULT_TARGET = (
    "/opt/vllm_kunlun/lib/python3.10/site-packages/vllm_kunlun/v1/attention/backends/kunlun_attn.py"
)

WRAPPER = '''

def _kdp_traced_kernel(_kdp_fn, **kwargs):
    """Record the real arguments when a vendor kernel returns non-zero."""
    import json, os
    import torch as _torch
    try:
        return _kdp_fn(**kwargs)
    except Exception as error:
        def describe(value):
            if isinstance(value, _torch.Tensor):
                info = {"shape": list(value.shape), "dtype": str(value.dtype),
                        "device": str(value.device), "contiguous": value.is_contiguous(),
                        "storage_offset": value.storage_offset()}
                if value.numel() and value.dtype in (_torch.int32, _torch.int64):
                    info["min"] = int(value.min().item())
                    info["max"] = int(value.max().item())
                return info
            return value
        payload = {key: describe(value) for key, value in kwargs.items()}
        payload["kernel"] = getattr(_kdp_fn, "__name__", str(_kdp_fn))
        payload["error"] = "{}: {}".format(type(error).__name__, error)
        path = os.environ.get("KDP_TRACE_PATH", "/tmp/kdp_kernel_failure.jsonl")
        with open(path, "a") as handle:
            handle.write(json.dumps(payload, default=str) + "\\n")
        raise
'''

PATCH_SCRIPT = '''
import ast, shutil, sys
target, call, anchor = sys.argv[1], sys.argv[2], sys.argv[3]
wrapper = sys.stdin.read()
source = open(target).read()
backup = target + ".kdp_backup"
try:
    open(backup)
except FileNotFoundError:
    shutil.copy(target, backup)
if "_kdp_traced_kernel" not in source:
    source = source.replace("import kunlun_ops\\n", "import kunlun_ops\\n" + wrapper, 1)
needle = "kunlun_ops.%s(\\n%s" % (call, anchor)
if needle not in source:
    print("ANCHOR_NOT_FOUND")
    raise SystemExit(1)
source = source.replace(
    needle, "_kdp_traced_kernel(kunlun_ops.%s, \\n%s" % (call, anchor), 1
)
ast.parse(source)
open(target, "w").write(source)
print("INSTRUMENTED")
'''

RESTORE_SCRIPT = '''
import os, shutil, sys
target = sys.argv[1]
backup = target + ".kdp_backup"
if not os.path.exists(backup):
    print("NO_BACKUP")
    raise SystemExit(1)
shutil.move(backup, target)
for cached in (os.path.join(os.path.dirname(target), "__pycache__"),):
    shutil.rmtree(cached, ignore_errors=True)
print("RESTORED")
'''


def run_in_pod(adapter: KunlunP800Adapter, pod: str, script: str, args: str, stdin: str = "") -> str:
    payload = base64.b64encode(script.encode()).decode()
    feed = f"printf %s {base64.b64encode(stdin.encode()).decode()} | base64 -d | " if stdin else ""
    command = (
        f"echo {payload} | base64 -d > /tmp/kdp_instrument_step.py && "
        f"{feed}python3 /tmp/kdp_instrument_step.py {args}"
    )
    result = adapter.exec(pod, command, timeout=300)
    return (result.stdout + result.stderr).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod", required=True)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--call", default="speculative_attention", help="kunlun_ops function name")
    parser.add_argument(
        "--anchor",
        default="                    out=output[:num_decode_tokens],",
        help="first argument line, to pick one call site out of several",
    )
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()

    adapter = KunlunP800Adapter()
    adapter.assert_owned(args.pod)
    if args.restore:
        print(run_in_pod(adapter, args.pod, RESTORE_SCRIPT, args.target))
        return 0
    print(
        run_in_pod(
            adapter,
            args.pod,
            PATCH_SCRIPT,
            f"{args.target} {args.call} {args.anchor!r}",
            stdin=WRAPPER,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
