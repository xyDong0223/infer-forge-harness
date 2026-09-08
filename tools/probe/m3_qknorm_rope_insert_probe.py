"""MiniMax-M3 fused qk-norm / RoPE / KV-insert probe. Prints JSON to stdout.

The op `_C::fused_minimax_m3_qknorm_rope_kv_insert` is the only vLLM custom op the M3
nvidia variant depends on, and it is unregistered here. Registering it is not the risky
part; the risky part is that it does three things at once — norm, RoPE, and a scatter
into two different paged caches — so a wrong cache layout or a missed RoPE produces a
service that runs and returns fluent, wrong text.

So every branch is graded against a float32 reference computed here, and the cache
writes are graded by *reading them back out* at the slots they were written to, not by
trusting that the write happened. The control omits RoPE from the reference: if that
still passes, the comparison cannot see whether RoPE was applied at all.

The layout under test is the platform's, not upstream's:
`(2, num_blocks, num_kv_heads, block_size, head_size)` from
`vllm_kunlun/ops/paged_attn.py:43`, which is also what the msa_* kernels read. Upstream
M3 assumes `(num_blocks, 2, 128, num_kv_heads, head_dim)`, so the disagreement is real
and belongs to the model code rather than to the cache.
"""

from __future__ import annotations

import argparse
import importlib.util
import json


def load(path: str):
    spec = importlib.util.spec_from_file_location("m3_fused_qknorm_rope_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def metrics(candidate, reference) -> dict:
    import torch

    a = candidate.detach().cpu().to(torch.float32).flatten()
    b = reference.detach().cpu().to(torch.float32).flatten()
    return {
        "cosine": float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-30)),
        "relative_l2": float((a - b).norm() / (b.norm() + 1e-30)),
    }


def reference(module, qkv, weights, cos, sin, shape: dict, rope: bool):
    """Norm and (optionally) RoPE, in float32, from the untouched fused tensor."""
    import torch

    head_dim, index_dim = shape["head_dim"], shape["index_dim"]
    sizes = [shape["heads"] * head_dim, shape["kv_heads"] * head_dim,
             shape["kv_heads"] * head_dim,
             shape["index_heads"] * index_dim, index_dim]
    q, k, v, index_q, index_k = qkv.to(torch.float32).split(sizes, dim=-1)

    def prepare(tensor, heads, dim, weight):
        normed = module._gemma_norm_per_head(
            tensor.reshape(-1, heads, dim), weight.to(torch.float32), shape["eps"])
        if not rope:
            return normed
        return module._neox_partial_rope(normed, cos, sin, shape["rotary_dim"])

    return {
        "q": prepare(q, shape["heads"], head_dim, weights["q"]),
        "k": prepare(k, shape["kv_heads"], head_dim, weights["k"]),
        "v": v.reshape(-1, shape["kv_heads"], head_dim),
        "index_q": prepare(index_q, shape["index_heads"], index_dim, weights["index_q"]),
        "index_k": prepare(index_k, 1, index_dim, weights["index_k"]),
    }


def read_back(cache, which: int, slot_mapping, block_size: int, head_major: bool = True):
    """Gather what is actually in the cache at the slots that were written."""
    import torch

    target = cache[which] if cache.dim() == 5 else cache
    flat = slot_mapping.view(-1).to(torch.long)
    if head_major:
        rows = [target[slot // block_size, :, slot % block_size, :] for slot in flat.tolist()]
    else:
        rows = [target[slot // block_size, slot % block_size, :, :] for slot in flat.tolist()]
    return torch.stack(rows)


def main() -> int:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", required=True,
                        help="path to patches/m3_fused_qknorm_rope_probe.py")
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--index-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--rotary-dim", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-layout", choices=("bhld", "blhd"), default="bhld",
                        help="bhld is vllm_kunlun/ops/paged_attn.py:43; blhd is what M3's "
                             "own spec allocates, and the fifth launch proved both occur")
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--cosine-floor", type=float, default=0.9999)
    parser.add_argument("--max-relative-l2", type=float, default=0.01)
    args = parser.parse_args()

    if args.device == "cuda":
        import vllm_kunlun  # noqa: F401 - installs the torch_xmlir mapping

    module = load(args.implementation)
    torch.manual_seed(args.seed)
    device, dtype = args.device, torch.bfloat16
    shape = {
        "heads": args.heads, "kv_heads": args.kv_heads, "index_heads": args.index_heads,
        "head_dim": args.head_dim, "index_dim": args.head_dim,
        "rotary_dim": args.rotary_dim, "eps": 1e-6,
    }
    result: dict = {
        "dimension": "m3_fused_qknorm_rope_insert",
        "operators": ["_C::fused_minimax_m3_qknorm_rope_kv_insert (torch stand-in)"],
        "geometry": {**shape, "block_size": args.block_size, "blocks": args.blocks,
                     "tokens": args.tokens, "layer": "sparse (num_index_heads > 0)"},
        "cache_layout": args.cache_layout,
        "thresholds": {"min_cosine": args.cosine_floor,
                       "max_relative_l2": args.max_relative_l2},
        "gate": "relative_l2 of every branch, and of what the caches actually contain",
        "cases": [],
    }

    head_dim, index_dim = shape["head_dim"], shape["index_dim"]
    width = (args.heads * head_dim + 2 * args.kv_heads * head_dim
             + args.index_heads * index_dim + index_dim)
    qkv = torch.randn((args.tokens, width), dtype=dtype, device=device)
    original = qkv.clone()
    weights = {name: torch.randn(head_dim, dtype=dtype, device=device)
               for name in ("q", "k", "index_q", "index_k")}
    positions = torch.arange(args.tokens, dtype=torch.long, device=device)
    cos_sin_cache = torch.randn((args.tokens + 8, args.rotary_dim), dtype=dtype, device=device)
    cos, sin = cos_sin_cache.index_select(0, positions).to(torch.float32).chunk(2, dim=-1)

    head_major = args.cache_layout == "bhld"
    if head_major:
        kv_shape = (2, args.blocks, args.kv_heads, args.block_size, head_dim)
        index_shape = (2, args.blocks, 1, args.block_size, index_dim)
    else:
        kv_shape = (2, args.blocks, args.block_size, args.kv_heads, head_dim)
        index_shape = (2, args.blocks, args.block_size, 1, index_dim)
    kv_cache = torch.zeros(kv_shape, dtype=dtype, device=device)
    index_cache = torch.zeros(index_shape, dtype=dtype, device=device)
    # Deliberately not slots 0..N: a probe that writes the identity mapping cannot
    # catch a block/offset swap, which is the mistake this layout invites.
    slot_mapping = torch.tensor(
        [(3 + 5 * token) % (args.blocks * args.block_size) for token in range(args.tokens)],
        dtype=torch.int32, device=device)
    index_slot_mapping = torch.tensor(
        [(11 + 7 * token) % (args.blocks * args.block_size) for token in range(args.tokens)],
        dtype=torch.int32, device=device)
    q_out = torch.zeros((args.tokens, args.heads * head_dim), dtype=dtype, device=device)
    index_q_out = torch.zeros((args.tokens, args.index_heads * index_dim),
                              dtype=dtype, device=device)

    try:
        module.fused_minimax_m3_qknorm_rope_kv_insert(
            qkv, weights["q"], weights["k"], cos_sin_cache, positions,
            args.heads, args.kv_heads, args.rotary_dim, shape["eps"],
            weights["index_q"], weights["index_k"], args.index_heads,
            slot_mapping, index_slot_mapping, kv_cache, index_cache,
            args.block_size, q_out, index_q_out, "auto")
        if device == "cuda":
            torch.cuda.synchronize()
    except Exception as error:
        result["state"] = "EVALUATION_ERROR"
        result["error"] = f"{type(error).__name__}: {error}"
        print(json.dumps(result))
        return 1

    expected = reference(module, original, weights, cos, sin, shape, rope=True)
    result["cases"] += [
        {"case": "q_out_norm_and_rope", **metrics(q_out, expected["q"])},
        {"case": "index_q_out_norm_and_rope", **metrics(index_q_out, expected["index_q"])},
        {"case": "k_cache_readback",
         "reference": "normed and roped k, gathered back from the slots written",
         **metrics(read_back(kv_cache, 0, slot_mapping, args.block_size, head_major),
                   expected["k"])},
        {"case": "v_cache_readback",
         "reference": "v is inserted unchanged; no norm and no rope on the value",
         **metrics(read_back(kv_cache, 1, slot_mapping, args.block_size, head_major),
                   expected["v"])},
        {"case": "index_cache_readback",
         "reference": "normed and roped index_k, gathered back from its own slot mapping",
         **metrics(read_back(index_cache, 0, index_slot_mapping, args.block_size,
                             head_major), expected["index_k"])},
    ]
    # Nothing may be written outside the slots that were mapped.
    written = {int(slot) for slot in slot_mapping.view(-1).tolist()}
    stray = 0
    for slot in range(args.blocks * args.block_size):
        if slot in written:
            continue
        block, offset = slot // args.block_size, slot % args.block_size
        lane = (kv_cache[0, block, :, offset, :] if head_major
                else kv_cache[0, block, offset, :, :])
        stray += int(lane.abs().sum().item() != 0.0)
    result["cases"].append({"case": "no_writes_outside_the_slot_mapping",
                            "unmapped_slots_touched": stray})
    return decide(args, result, module, original, weights, cos, sin, shape,
                  kv_cache, slot_mapping, torch, head_major)


def decide(args, result, module, original, weights, cos, sin, shape,
           kv_cache, slot_mapping, torch, head_major) -> int:
    """The control, then the verdict."""
    # Control: grade the same cache contents against a reference that skips RoPE. If
    # that passes too, the comparison cannot tell whether RoPE ran.
    without_rope = reference(module, original, weights, cos, sin, shape, rope=False)
    control = metrics(read_back(kv_cache, 0, slot_mapping, args.block_size, head_major),
                      without_rope["k"])
    result["control"] = {
        "case": "reference_without_rope",
        "description": "the same readback graded against norm-only k",
        **control,
        "discriminates": control["relative_l2"] > args.max_relative_l2,
    }

    numeric = [case for case in result["cases"] if "relative_l2" in case]
    passed = all(case["relative_l2"] <= args.max_relative_l2
                 and case["cosine"] >= args.cosine_floor for case in numeric)
    stray = next(case["unmapped_slots_touched"] for case in result["cases"]
                 if case["case"] == "no_writes_outside_the_slot_mapping")
    if not result["control"]["discriminates"]:
        result["state"] = "EVALUATION_INCONCLUSIVE"
    elif stray:
        result["state"] = "EXERCISED_FAIL"
        result["error"] = f"{stray} unmapped slots were written"
    else:
        result["state"] = "EXERCISED_PASS" if passed else "EXERCISED_FAIL"
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



