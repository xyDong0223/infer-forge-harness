"""In-pod sliding-window decode probe. Prints JSON to stdout.

Answers two questions with one geometry, both of which were open after Qwen3-8B:

- Does the torch decode fallback implement the sliding window correctly? Until now
  it refused any `max_window_size >= 0`, so a sliding-window model fell back to the
  vendor kernel — the same kernel that fails inside the server.
- Does the vendor kernel agree about what `max_window_size` means? The backend
  passes `self.sliding_window` straight through, with no `(window - 1, 0)`
  adjustment of the kind flash-attn makes, so the off-by-one is worth measuring
  rather than assuming.

The reference is float32 attention computed on the CPU from the same paged cache,
so neither implementation is graded against the other. The control ignores the
window: at a context longer than the window it must disagree with the reference, or
the geometry does not actually test windowing.
"""

from __future__ import annotations

import argparse
import importlib.util
import json


def load_fallback(path: str):
    spec = importlib.util.spec_from_file_location("torch_paged_decode", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def build_case(heads: int, kv_heads: int, dim: int, block_size: int, batch: int,
               context_len: int, window: int, seed: int, dtype):
    import torch

    torch.manual_seed(seed)
    used = max(1, (context_len + block_size - 1) // block_size)
    max_blocks = used
    blocks = batch * used + 1
    # A zeros cache would make every window produce the same output, so the probe
    # would pass while measuring nothing.
    kv_cache = torch.randn((2, blocks, kv_heads, block_size, dim), dtype=dtype, device="cuda")
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
        "max_context_len": max_blocks * block_size,
        "head_num": heads,
        "head_dim": dim,
        "scale": dim**-0.5,
        "kv_head_num": kv_heads,
        "block_size": block_size,
        "max_num_blocks_per_seq": max_blocks,
        "max_window_size": window,
        "block_tables": tables,
        "sink": None,
    }


def cpu_reference(case: dict, window: int):
    """Float32 windowed attention on the CPU, from the same paged cache."""
    import torch

    heads, kv_heads = case["head_num"], case["kv_head_num"]
    dim, block_size = case["head_dim"], case["block_size"]
    group = heads // kv_heads
    k_cache = case["k_cache"].cpu().to(torch.float32)
    v_cache = case["v_cache"].cpu().to(torch.float32)
    tables = case["block_tables"].cpu().to(torch.int64)
    lens = case["context_lens_cpu"].to(torch.int64)
    query = case["q"].cpu().to(torch.float32).reshape(-1, heads, dim)

    outputs = []
    for row in range(query.shape[0]):
        length = int(lens[row])
        keys, values = [], []
        for position in range(length):
            block = int(tables[row, position // block_size])
            offset = position % block_size
            keys.append(k_cache[block, :, offset, :])
            values.append(v_cache[block, :, offset, :])
        keys = torch.stack(keys, dim=1)      # (kv_heads, length, dim)
        values = torch.stack(values, dim=1)
        lowest = 0 if window < 0 else max(0, length - window)
        keys, values = keys[:, lowest:, :], values[:, lowest:, :]
        grouped = query[row].reshape(kv_heads, group, dim)
        scores = torch.einsum("kgd,kld->kgl", grouped, keys) * case["scale"]
        weights = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("kgl,kld->kgd", weights, values).reshape(heads, dim))
    return torch.stack(outputs)


def metrics(candidate, reference) -> dict:
    import torch

    a = candidate.cpu().to(torch.float32)
    b = reference.to(torch.float32)
    flat_a, flat_b = a.flatten(), b.flatten()
    return {
        "cosine": float(torch.dot(flat_a, flat_b) / (flat_a.norm() * flat_b.norm() + 1e-30)),
        "relative_l2": float((a - b).norm() / (b.norm() + 1e-30)),
    }


def main() -> int:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fallback", required=True, help="path to patches/torch_paged_decode.py")
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--context-len", type=int, default=96)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--kernel", default="speculative_attention")
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--cosine-floor", type=float, default=0.9999)
    parser.add_argument("--max-relative-l2", type=float, default=0.01)
    args = parser.parse_args()

    import vllm_kunlun  # noqa: F401 - installs the torch_xmlir mapping and custom ops
    import kunlun_ops

    fallback = load_fallback(args.fallback)
    dtype = torch.bfloat16
    geometry = {"heads": args.heads, "kv_heads": args.kv_heads, "head_dim": args.head_dim,
                "block_size": args.block_size, "batch": args.batch,
                "context_len": args.context_len, "window": args.window}
    result: dict = {
        "dimension": "msa",
        "operators": [f"kunlun_ops.{args.kernel}", "patches/torch_paged_decode.py"],
        "geometry": geometry,
        "thresholds": {"min_cosine": args.cosine_floor,
                       "max_relative_l2": args.max_relative_l2},
        "gate": "relative_l2 of the torch fallback against a CPU float32 windowed reference",
        "cases": [],
    }
    if args.context_len <= args.window:
        result["state"] = "EVALUATION_INCONCLUSIVE"
        result["error"] = (
            f"context_len {args.context_len} does not exceed window {args.window}, so a windowed "
            "and an unwindowed decode are the same computation"
        )
        print(json.dumps(result))
        return 1

    def case(window: int) -> dict:
        return build_case(args.heads, args.kv_heads, args.head_dim, args.block_size,
                          args.batch, args.context_len, window, args.seed, dtype)

    reference = cpu_reference(case(args.window), args.window)

    try:
        torch_case = case(args.window)
        fallback.torch_paged_decode(**torch_case)
        torch.cuda.synchronize()
        result["cases"].append({
            "case": "torch_fallback_vs_cpu_reference",
            "reference": "float32 windowed attention on the CPU from the same paged cache",
            **metrics(torch_case["out"], reference),
        })
    except Exception as error:
        result["state"] = "EVALUATION_ERROR"
        result["error"] = f"torch fallback: {type(error).__name__}: {error}"
        print(json.dumps(result))
        return 1

    # The vendor kernel is measured, not graded: it is the definition of the
    # convention the backend expects, and disagreement here is a finding about the
    # window's meaning rather than a defect in the fallback.
    try:
        vendor_case = case(args.window)
        getattr(kunlun_ops, args.kernel)(**vendor_case)
        torch.cuda.synchronize()
        vendor = {"case": "vendor_kernel_vs_cpu_reference",
                  "reference": "float32 windowed attention on the CPU from the same paged cache",
                  **metrics(vendor_case["out"], reference)}
        # Which off-by-one the vendor actually implements, if any.
        offsets = {}
        for offset in (-1, 0, 1):
            candidate = cpu_reference(vendor_case, args.window + offset)
            offsets[str(offset)] = metrics(vendor_case["out"], candidate)["relative_l2"]
        vendor["best_window_offset"] = min(offsets, key=lambda key: offsets[key])
        vendor["relative_l2_by_offset"] = offsets
        # The decisive comparison, added after the first run: if the vendor output
        # matches an *unwindowed* reference to bf16 precision, `max_window_size` is
        # not an off-by-one away from the mask — it is being ignored.
        unwindowed = metrics(vendor_case["out"], cpu_reference(vendor_case, -1))
        vendor["vs_unwindowed_reference"] = unwindowed
        vendor["ignores_window"] = unwindowed["relative_l2"] <= args.max_relative_l2
        result["cases"].append(vendor)
    except Exception as error:
        result["vendor_kernel"] = {"error": f"{type(error).__name__}: {error}"}

    # Control: ignore the window. At a context longer than the window this must
    # disagree with the reference, or the geometry proves nothing about windowing.
    try:
        control_case = case(-1)
        fallback.torch_paged_decode(**control_case)
        torch.cuda.synchronize()
        control_metrics = metrics(control_case["out"], reference)
        control_error: str | None = None
    except Exception as error:
        control_metrics, control_error = {}, f"{type(error).__name__}: {error}"
    result["control"] = {
        "case": "ignore_the_window",
        "description": "the same fallback with max_window_size=-1, i.e. full attention",
        **control_metrics,
        "error": control_error,
        "discriminates": control_error is not None
        or control_metrics.get("relative_l2", 0.0) > args.max_relative_l2,
    }

    primary = result["cases"][0]
    passed = (primary["relative_l2"] <= args.max_relative_l2
              and primary["cosine"] >= args.cosine_floor)
    if not result["control"]["discriminates"]:
        result["state"] = "EVALUATION_INCONCLUSIVE"
    else:
        result["state"] = "EXERCISED_PASS" if passed else "EXERCISED_FAIL"
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
