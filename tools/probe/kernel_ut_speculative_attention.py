"""Unit test that isolates `kunlun_ops.speculative_attention` from the server.

The server only reports `Check 0 == ret failed` from inside decode warmup, which
says nothing about which shape triggers it. This runs the kernel directly with
Qwen3-8B's dimensions and sweeps one variable at a time, so the failure gets a
boundary instead of a symptom.

Argument names and the KV layout are taken from the call site,
`vllm_kunlun/v1/attention/backends/kunlun_attn.py`, non-speculative decode branch:
KV cache is `(2, num_blocks, num_kv_heads, block_size, head_size)` and the kernel
receives `qlen=1`, `max_context_len=max_model_len`, `block_size=k_cache.shape[2]`.

Run inside a pod with free XPUs:
    python3 kernel_ut_speculative_attention.py [--json]
"""

from __future__ import annotations

import argparse
import json
import traceback

import torch
import vllm_kunlun  # noqa: F401 - installs the torch_xmlir mapping and custom ops
import kunlun_ops  # top-level vendor package, as imported by kunlun_attn.py

QWEN3_8B = {"head_num": 32, "kv_head_num": 8, "head_dim": 128}


def build_case(
    batch_num: int,
    context_len: int,
    block_size: int,
    max_context_len: int,
    dtype: torch.dtype,
    head_num: int,
    kv_head_num: int,
    head_dim: int,
) -> dict:
    max_blocks = (max_context_len + block_size - 1) // block_size
    num_blocks = max(batch_num * ((context_len + block_size - 1) // block_size), 1) + 1
    kv_cache = torch.zeros(
        (2, num_blocks, kv_head_num, block_size, head_dim), dtype=dtype, device="cuda"
    )
    block_tables = torch.zeros((batch_num, max_blocks), dtype=torch.int32, device="cuda")
    used = (context_len + block_size - 1) // block_size
    for request in range(batch_num):
        for block in range(used):
            block_tables[request, block] = (request * used + block) % num_blocks
    lens_cpu = torch.full((batch_num,), context_len, dtype=torch.int32)
    return {
        "out": torch.zeros((batch_num, head_num, head_dim), dtype=dtype, device="cuda"),
        "q": torch.randn((batch_num, 1, head_num, head_dim), dtype=dtype, device="cuda"),
        "k_cache": kv_cache[0],
        "v_cache": kv_cache[1],
        "context_lens_cpu": lens_cpu,
        "context_lens_xpu": lens_cpu.to("cuda"),
        "batch_num": batch_num,
        "qlen": 1,
        "max_context_len": max_context_len,
        "head_num": head_num,
        "head_dim": head_dim,
        "scale": head_dim**-0.5,
        "kv_head_num": kv_head_num,
        "block_size": block_size,
        "max_num_blocks_per_seq": max_blocks,
        "max_window_size": -1,
        "block_tables": block_tables,
        "sink": None,
    }


def attempt(label: str, **case: object) -> dict:
    dtype = case.pop("dtype")
    try:
        kwargs = build_case(dtype=dtype, **case)  # type: ignore[arg-type]
    except Exception as error:
        return {"case": label, "ok": False, "stage": "setup", "error": f"{type(error).__name__}: {error}"}
    try:
        kunlun_ops.speculative_attention(**kwargs)
        torch.cuda.synchronize()
        finite = bool(torch.isfinite(kwargs["out"]).all().item())
        return {"case": label, "ok": True, "output_finite": finite}
    except Exception as error:
        return {
            "case": label,
            "ok": False,
            "stage": "kernel",
            "error": f"{type(error).__name__}: {error}",
            "trace": traceback.format_exc(limit=3).splitlines()[-1],
        }


def sweep() -> list[dict]:
    base = dict(batch_num=1, context_len=128, block_size=64, max_context_len=32768,
                dtype=torch.bfloat16, **QWEN3_8B)
    results = [attempt("baseline", **dict(base))]

    for batch_num in (1, 2, 8, 32, 51, 256):
        results.append(attempt(f"batch_num={batch_num}", **{**base, "batch_num": batch_num}))
    for context_len in (1, 2, 16, 63, 64, 65, 128, 1024, 4096, 32768):
        results.append(attempt(f"context_len={context_len}", **{**base, "context_len": context_len}))
    for block_size in (16, 32, 64, 128, 256):
        results.append(attempt(f"block_size={block_size}", **{**base, "block_size": block_size}))
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        results.append(attempt(f"dtype={dtype}", **{**base, "dtype": dtype}))
    # The call site passes max_model_len, not the batch's real maximum, so the
    # kernel may be sizing scratch space from a much larger number.
    for max_context_len in (128, 4096, 32768, 40960, 131072):
        results.append(
            attempt(f"max_context_len={max_context_len}",
                    **{**base, "max_context_len": max_context_len})
        )
    # GQA ratio: Qwen3-8B is 32/8. A kernel may only handle certain ratios.
    for head_num, kv_head_num in ((32, 32), (32, 8), (32, 4), (32, 2), (32, 1), (16, 8), (8, 8)):
        results.append(
            attempt(f"heads={head_num}/{kv_head_num}",
                    **{**base, "head_num": head_num, "kv_head_num": kv_head_num})
        )
    for head_dim in (64, 128, 192, 256):
        results.append(attempt(f"head_dim={head_dim}", **{**base, "head_dim": head_dim}))
    results.extend(sweep_runtime_shapes())
    return results


def sweep_runtime_shapes() -> list[dict]:
    """Cases where the server's tensors differ from a textbook call.

    The plain sweep passes on every Qwen3-8B shape, so whatever breaks inside the
    server is not the shape itself. These probe what still differs between a
    direct call and the real decode path: a block table narrower than
    `max_context_len` implies, index dtypes the scheduler happens to produce, and
    tensors that are slices of a larger activation buffer.
    """
    results: list[dict] = []
    common = dict(dtype=torch.bfloat16, **QWEN3_8B)
    heads, dim = QWEN3_8B["head_num"], QWEN3_8B["head_dim"]

    def call(label: str, mutate) -> None:
        try:
            case = build_case(batch_num=2, context_len=128, block_size=64,
                              max_context_len=32768, **common)  # type: ignore[arg-type]
            mutate(case)
            kunlun_ops.speculative_attention(**case)
            torch.cuda.synchronize()
            results.append({"case": label, "ok": True})
        except Exception as error:
            results.append({"case": label, "ok": False, "stage": "kernel",
                            "error": f"{type(error).__name__}: {error}"})

    # The call site passes max_context_len=max_model_len while
    # max_num_blocks_per_seq comes from the scheduler's much narrower table.
    for width in (1, 2, 8):
        def narrow(case: dict, width: int = width) -> None:
            case["block_tables"] = case["block_tables"][:, :width].contiguous()
            case["max_num_blocks_per_seq"] = width

        call(f"narrow_block_table(width={width})", narrow)

    call(
        "context_lens_int64",
        lambda case: case.update(
            context_lens_cpu=case["context_lens_cpu"].to(torch.int64),
            context_lens_xpu=case["context_lens_xpu"].to(torch.int64),
        ),
    )
    call(
        "block_tables_int64",
        lambda case: case.update(block_tables=case["block_tables"].to(torch.int64)),
    )
    call(
        "out_is_slice_of_larger_buffer",
        lambda case: case.update(
            out=torch.zeros((64, heads, dim), dtype=torch.bfloat16, device="cuda")[:2]
        ),
    )
    call(
        "q_is_view_of_larger_buffer",
        lambda case: case.update(
            q=torch.randn((64, heads, dim), dtype=torch.bfloat16, device="cuda")[:2].view(
                -1, 1, heads, dim
            )
        ),
    )
    call("context_len_exceeds_max", lambda case: case.update(max_context_len=64))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    results = sweep()
    if args.json:
        print(json.dumps({"state": "UT_COMPLETE", "results": results}))
        return 0
    for entry in results:
        mark = "PASS" if entry["ok"] else "FAIL"
        detail = "" if entry["ok"] else f"  {entry.get('stage')}: {entry.get('error', '')[:120]}"
        finite = "" if entry.get("output_finite", True) else "  (non-finite output)"
        print(f"{mark}  {entry['case']}{finite}{detail}")
    failed = [entry for entry in results if not entry["ok"]]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
