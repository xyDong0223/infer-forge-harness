"""Replay a captured vendor-kernel failure and sweep around it.

Model-agnostic by construction: the base case comes from the arguments
`tools/instrument_kernel_trace.py` captured in the server, so the geometry is
whatever that model actually used. Nothing here knows about Qwen3 — the earlier
version hardcoded 32/8/128 and would have needed editing for every new model.

    python3 kernel_ut_replay.py --from-trace /tmp/kdp_kernel_failure.jsonl
    python3 kernel_ut_replay.py --heads 64 --kv-heads 8 --head-dim 128   # no trace

Run inside a pod with free XPUs. `--kernel` selects which `kunlun_ops` symbol to
exercise, so the same sweep works for whichever call a triage lands on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import vllm_kunlun  # noqa: F401 - installs the torch_xmlir mapping and custom ops
import kunlun_ops  # top-level vendor package, as imported by kunlun_attn.py

FALLBACK_GEOMETRY = {
    "head_num": 32,
    "kv_head_num": 8,
    "head_dim": 128,
    "block_size": 64,
    "batch_num": 1,
    "context_len": 128,
    "max_context_len": 32768,
}
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def geometry_from_trace(path: Path) -> dict:
    """Read the last captured failure and turn it into a base case."""
    records = [json.loads(line) for line in path.read_text().splitlines() if line.startswith("{")]
    if not records:
        raise SystemExit(f"no captured failure in {path}")
    trace = records[-1]
    lens = trace.get("context_lens_cpu") or {}
    return {
        "head_num": int(trace["head_num"]),
        "kv_head_num": int(trace["kv_head_num"]),
        "head_dim": int(trace["head_dim"]),
        "block_size": int(trace["block_size"]),
        "batch_num": int(trace["batch_num"]),
        "context_len": int(lens.get("max", 128)),
        "max_context_len": int(trace["max_context_len"]),
        "blocks": int((trace.get("k_cache") or {}).get("shape", [1024])[0]),
        "dtype": (trace.get("k_cache") or {}).get("dtype", "torch.bfloat16").split(".")[-1],
    }


def build_case(geometry: dict, dtype: torch.dtype, **overrides: int) -> dict:
    spec = {**geometry, **overrides}
    block_size = spec["block_size"]
    heads, kv_heads, dim = spec["head_num"], spec["kv_head_num"], spec["head_dim"]
    batch, context_len = spec["batch_num"], spec["context_len"]
    max_blocks = max(1, (spec["max_context_len"] + block_size - 1) // block_size)
    used = max(1, (context_len + block_size - 1) // block_size)
    blocks = spec.get("blocks") or (batch * used + 1)
    kv_cache = torch.zeros((2, blocks, kv_heads, block_size, dim), dtype=dtype, device="cuda")
    tables = torch.zeros((batch, max_blocks), dtype=torch.int32, device="cuda")
    for request in range(batch):
        for block in range(used):
            tables[request, block] = (request * used + block) % blocks
    lens = torch.full((batch,), context_len, dtype=torch.int32)
    return {
        "out": torch.zeros((batch, heads, dim), dtype=dtype, device="cuda"),
        "q": torch.randn((batch, 1, heads, dim), dtype=dtype, device="cuda"),
        "k_cache": kv_cache[0],
        "v_cache": kv_cache[1],
        "context_lens_cpu": lens,
        "context_lens_xpu": lens.to("cuda"),
        "batch_num": batch,
        "qlen": 1,
        "max_context_len": spec["max_context_len"],
        "head_num": heads,
        "head_dim": dim,
        "scale": dim**-0.5,
        "kv_head_num": kv_heads,
        "block_size": block_size,
        "max_num_blocks_per_seq": max_blocks,
        "max_window_size": -1,
        "block_tables": tables,
        "sink": None,
    }


def attempt(kernel, label: str, geometry: dict, dtype: torch.dtype, **overrides: int) -> dict:
    try:
        case = build_case(geometry, dtype, **overrides)
    except Exception as error:
        return {"case": label, "ok": False, "stage": "setup", "error": f"{type(error).__name__}: {error}"}
    try:
        kernel(**case)
        torch.cuda.synchronize()
        return {"case": label, "ok": True, "finite": bool(torch.isfinite(case["out"]).all().item())}
    except Exception as error:
        return {"case": label, "ok": False, "stage": "kernel", "error": f"{type(error).__name__}: {error}"}


def sweep(kernel, geometry: dict, dtype: torch.dtype) -> list[dict]:
    """Vary one dimension at a time around the captured case."""
    results = [attempt(kernel, "captured_case", geometry, dtype)]
    base_batch, base_ctx = geometry["batch_num"], geometry["context_len"]
    for batch in sorted({1, 2, 8, 32, base_batch // 2 or 1, base_batch, base_batch * 2}):
        results.append(attempt(kernel, f"batch_num={batch}", geometry, dtype, batch_num=batch))
    for context in sorted({1, 2, 16, base_ctx, base_ctx * 8, geometry["max_context_len"]}):
        results.append(attempt(kernel, f"context_len={context}", geometry, dtype, context_len=context))
    for block_size in (16, 32, 64, 128, 256):
        results.append(attempt(kernel, f"block_size={block_size}", geometry, dtype, block_size=block_size))
    for name, candidate in DTYPES.items():
        results.append(attempt(kernel, f"dtype={name}", geometry, candidate))
    # GQA ratios around the model's own, since a kernel may only handle some.
    heads, kv_heads = geometry["head_num"], geometry["kv_head_num"]
    for ratio_kv in sorted({1, 2, kv_heads // 2 or 1, kv_heads, heads}):
        if heads % ratio_kv == 0:
            results.append(
                attempt(kernel, f"heads={heads}/{ratio_kv}", geometry, dtype, kv_head_num=ratio_kv)
            )
    for dim in sorted({64, 128, geometry["head_dim"], 256}):
        results.append(attempt(kernel, f"head_dim={dim}", geometry, dtype, head_dim=dim))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-trace", type=Path, help="JSONL written by instrument_kernel_trace")
    parser.add_argument("--kernel", default="speculative_attention", help="kunlun_ops symbol")
    parser.add_argument("--heads", type=int)
    parser.add_argument("--kv-heads", type=int)
    parser.add_argument("--head-dim", type=int)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    geometry = dict(FALLBACK_GEOMETRY)
    if args.from_trace:
        geometry.update(geometry_from_trace(args.from_trace))
    for key, value in (("head_num", args.heads), ("kv_head_num", args.kv_heads),
                       ("head_dim", args.head_dim)):
        if value:
            geometry[key] = value
    dtype = DTYPES[geometry.get("dtype", args.dtype)] if args.from_trace else DTYPES[args.dtype]

    kernel = getattr(kunlun_ops, args.kernel)
    results = sweep(kernel, geometry, dtype)
    if args.json:
        print(json.dumps({"state": "UT_COMPLETE", "kernel": args.kernel,
                          "geometry": geometry, "results": results}))
        return 0
    print(f"kernel={args.kernel} geometry={geometry}")
    for entry in results:
        mark = "PASS" if entry["ok"] else "FAIL"
        detail = "" if entry["ok"] else f"  {entry.get('stage')}: {entry.get('error', '')[:110]}"
        print(f"{mark}  {entry['case']}{detail}")
    passed = sum(1 for entry in results if entry["ok"])
    print(f"\n{passed}/{len(results)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
