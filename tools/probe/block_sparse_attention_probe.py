"""In-pod MSA block-sparse attention probe. Prints JSON to stdout.

MiniMax-M3's sparse attention is a triton kernel upstream, shared by the nvidia and
amd variants, and triton is not the execution path here. The vendor library does ship
the three kernels the path needs — `msa_block_score`, `msa_block_score_topk_transform`
and `msa_sparse_attention` — but nothing in vLLM-Kunlun calls them, so their
conventions are documented and unexercised. Wiring a model against an unexercised
kernel produces a service that runs and is wrong.

So this probe establishes the conventions before any model code is written, against a
float32 reference computed off the accelerator from the same paged cache:

- what `msa_block_score` reduces a block to. The docstring says "block-level
  similarity" and takes a `score_type`; whether that is a max, a mean, a sum or a
  log-sum-exp over the block's tokens is measured against all four rather than assumed.
  Upstream's triton kernel answers `tl.max(qk, axis=1)` and omits the scale in decode
  because only ordering is consumed, so whether this kernel applies `softmax_scale` is
  measured too.
- whether the top-k transform reserves a local block, which changes the selected set.
- whether sparse attention over the selected blocks matches attention restricted to
  those same blocks.

Measured constraint, found by sweeping the arguments: `msa_block_score` accepts only
`head_num_kv == 1`. Every other axis swept — page size 64/128, score block 64/128
independently, prefill and decode, one or many queries per request, all three LOD
branches, batch 1 — is accepted. One KV head is consistent with the index cache
upstream declares as `[num_blocks, 128, head_dim]`, so the constraint is the shape of
the lightning indexer rather than a limitation.

The control feeds the reference a different block set: if that still passes, the
comparison is not sensitive to block selection and proves nothing.
"""

from __future__ import annotations

import argparse
import json

CANDIDATES = ("max", "max_unscaled", "mean", "sum", "logsumexp")


def build_cache(batch: int, heads: int, kv_heads: int, dim: int, page: int,
                context_len: int, seed: int, dtype):
    """A paged cache plus one decode query per request."""
    import torch

    torch.manual_seed(seed)
    pages_per_seq = (context_len + page - 1) // page
    pages = batch * pages_per_seq
    # Randn, not zeros: a zeroed cache makes every block score identical, and every
    # block selection then looks correct.
    k_cache = torch.randn((pages, kv_heads, page, dim), dtype=dtype, device="cuda")
    v_cache = torch.randn((pages, kv_heads, page, dim), dtype=dtype, device="cuda")
    tables = torch.arange(pages, dtype=torch.int32, device="cuda").reshape(batch, pages_per_seq)
    query = torch.randn((batch, heads, dim), dtype=dtype, device="cuda")
    lengths = torch.full((batch,), context_len, dtype=torch.int32)
    return {
        "q": query,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "block_tables": tables,
        "lengths_cpu": lengths,
        "lengths_xpu": lengths.to("cuda"),
        "cu_q_cpu": torch.arange(batch + 1, dtype=torch.int32),
        "cu_kv_cpu": torch.cat([
            torch.zeros(1, dtype=torch.int32), torch.cumsum(lengths, 0).to(torch.int32)
        ]),
        "pages_per_seq": pages_per_seq,
        "batch": batch,
        "heads": heads,
        "kv_heads": kv_heads,
        "dim": dim,
        "page": page,
        "context_len": context_len,
        "scale": dim**-0.5,
    }


def gather(case: dict):
    """The keys and values of each request, in position order, as float32 on the CPU."""
    import torch

    k_cache = case["k_cache"].cpu().to(torch.float32)
    v_cache = case["v_cache"].cpu().to(torch.float32)
    tables = case["block_tables"].cpu().to(torch.int64)
    page, length = case["page"], case["context_len"]
    keys, values = [], []
    for request in range(case["batch"]):
        request_k, request_v = [], []
        for position in range(length):
            block = int(tables[request, position // page])
            request_k.append(k_cache[block, :, position % page, :])
            request_v.append(v_cache[block, :, position % page, :])
        keys.append(torch.stack(request_k, dim=1))     # (kv_heads, length, dim)
        values.append(torch.stack(request_v, dim=1))
    return torch.stack(keys), torch.stack(values)      # (batch, kv_heads, length, dim)


def token_scores(case: dict, scaled: bool = True):
    """Scaled q·k per (request, head, position), float32 off the accelerator."""
    import torch

    keys, _ = gather(case)
    query = case["q"].cpu().to(torch.float32)
    group = case["heads"] // case["kv_heads"]
    grouped = query.reshape(case["batch"], case["kv_heads"], group, case["dim"])
    scores = torch.einsum("bkgd,bkld->bkgl", grouped, keys)
    if scaled:
        scores = scores * case["scale"]
    return scores.reshape(case["batch"], case["heads"], case["context_len"])


def reference_block_scores(case: dict, block_size: int, how: str):
    """Reduce each block of token scores to one number, several ways.

    Which one the kernel implements is the open question; naming a guess and checking
    only that guess would confirm whatever the kernel does. `max_unscaled` is included
    because upstream's decode kernel deliberately drops `softmax_scale` — it consumes
    only the block ordering, which the scale does not change.
    """
    import torch

    scores = token_scores(case, scaled=how != "max_unscaled")
    length = case["context_len"]
    blocks = (length + block_size - 1) // block_size
    padded = block_size * blocks - length
    if padded:
        filler = {"max": float("-inf"), "max_unscaled": float("-inf"),
                  "logsumexp": float("-inf")}.get(how, 0.0)
        scores = torch.nn.functional.pad(scores, (0, padded), value=filler)
    tiled = scores.reshape(case["batch"], case["heads"], blocks, block_size)
    if how in ("max", "max_unscaled"):
        return tiled.amax(dim=-1)
    if how == "mean":
        return tiled.mean(dim=-1)
    if how == "sum":
        return tiled.sum(dim=-1)
    return torch.logsumexp(tiled, dim=-1)


def metrics(candidate, reference) -> dict:
    import torch

    a = candidate.detach().cpu().to(torch.float32).flatten()
    b = reference.detach().cpu().to(torch.float32).flatten()
    finite = torch.isfinite(a) & torch.isfinite(b)
    a, b = a[finite], b[finite]
    return {
        "cosine": float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-30)),
        "relative_l2": float((a - b).norm() / (b.norm() + 1e-30)),
        "compared_elements": int(finite.sum()),
    }


def dense_reference(case: dict):
    """Full attention over the whole context, float32 on the CPU."""
    import torch

    keys, values = gather(case)
    query = case["q"].cpu().to(torch.float32)
    group = case["heads"] // case["kv_heads"]
    grouped = query.reshape(case["batch"], case["kv_heads"], group, case["dim"])
    scores = torch.einsum("bkgd,bkld->bkgl", grouped, keys) * case["scale"]
    weights = torch.softmax(scores, dim=-1)
    out = torch.einsum("bkgl,bkld->bkgd", weights, values)
    return out.reshape(case["batch"], case["heads"], case["dim"])


def sparse_reference(case: dict, topk_idx, block_size: int):
    """Attention restricted to the selected blocks, float32 on the CPU.

    topk_idx is per kv head, so every query head in a group sees the same blocks.
    """
    import torch

    keys, values = gather(case)
    query = case["q"].cpu().to(torch.float32)
    group = case["heads"] // case["kv_heads"]
    length = case["context_len"]
    selection = topk_idx.detach().cpu().to(torch.int64)
    out = torch.zeros((case["batch"], case["kv_heads"], group, case["dim"]))
    for request in range(case["batch"]):
        for head in range(case["kv_heads"]):
            positions: list[int] = []
            for block in selection[request, head].tolist():
                if block < 0:
                    continue
                start = block * block_size
                positions += [p for p in range(start, min(start + block_size, length))]
            positions = sorted(set(positions))
            if not positions:
                continue
            index = torch.tensor(positions, dtype=torch.int64)
            selected_k = keys[request, head].index_select(0, index)
            selected_v = values[request, head].index_select(0, index)
            scores = (query[request].reshape(case["kv_heads"], group, case["dim"])[head]
                      @ selected_k.transpose(0, 1)) * case["scale"]
            out[request, head] = torch.softmax(scores, dim=-1) @ selected_v
    return out.reshape(case["batch"], case["heads"], case["dim"])


def reference_topk(block_scores, topk: int, blocks: int, reserve_local: bool):
    """Top-k blocks per row, with and without forcing the last block in.

    The docstring says one local block is kept; whether that displaces a top-k choice
    is the difference between two selected sets, so both are computed.
    """
    import torch

    scores = block_scores.detach().cpu().to(torch.float32)[..., :blocks].clone()
    local = blocks - 1
    if reserve_local:
        scores[..., local] = float("inf")
    take = min(topk, blocks)
    chosen = scores.topk(take, dim=-1).indices
    return chosen


def set_agreement(kernel_idx, reference_idx) -> dict:
    """Selected blocks compared as sets: order carries no meaning here."""
    rows = kernel_idx.reshape(-1, kernel_idx.shape[-1]).tolist()
    other = reference_idx.reshape(-1, reference_idx.shape[-1]).tolist()
    exact, overlap = 0, 0.0
    for left, right in zip(rows, other):
        left_set = {value for value in left if value >= 0}
        right_set = {value for value in right if value >= 0}
        if left_set == right_set:
            exact += 1
        union = len(left_set | right_set) or 1
        overlap += len(left_set & right_set) / union
    return {
        "rows": len(rows),
        "exact_set_match_fraction": exact / max(1, len(rows)),
        "mean_jaccard": overlap / max(1, len(rows)),
    }


def lods(case: dict):
    return {
        "lod_seqlens_q_cpu": case["cu_q_cpu"],
        "lod_seqlens_q_xpu": case["cu_q_cpu"].to("cuda"),
        "lod_seqlens_kv_cpu": case["cu_kv_cpu"],
        "lod_seqlens_kv_xpu": case["cu_kv_cpu"].to("cuda"),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    # Two stages, two geometries, following M3: the indexer scores against a
    # single-head index cache, the main heads attend against the paged K/V cache.
    parser.add_argument("--index-heads", type=int, default=4,
                        help="num_idx_heads; upstream sets this equal to num_kv_heads")
    parser.add_argument("--heads", type=int, default=16, help="main attention heads")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=64, help="score/top-k block size")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--context-len", type=int, default=512)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--cosine-floor", type=float, default=0.9999)
    parser.add_argument("--max-relative-l2", type=float, default=0.02)
    return parser.parse_args()


def main() -> int:
    import torch

    args = parse_args()

    import vllm_kunlun  # noqa: F401 - installs the torch_xmlir mapping and custom ops
    import kunlun_ops

    dtype = torch.bfloat16
    blocks = (args.context_len + args.block_size - 1) // args.block_size
    result: dict = {
        "dimension": "block_sparse",
        "operators": [
            "kunlun_ops.msa_block_score",
            "kunlun_ops.msa_block_score_topk_transform",
            "kunlun_ops.msa_sparse_attention",
        ],
        "geometry": {
            "index_heads": args.index_heads, "index_cache_kv_heads": 1,
            "heads": args.heads, "kv_heads": args.index_heads,
            "head_dim": args.head_dim, "page_size": args.page_size,
            "block_size": args.block_size, "batch": args.batch,
            "context_len": args.context_len, "topk": args.topk, "score_blocks": blocks,
            "phase": "decode (prefill_len=-1, one query per request)",
        },
        "conventions": {
            "index_cache_kv_heads": "msa_block_score accepts head_num_kv == 1 only; measured",
            "head_dim_v": "0 is documented as 'same as head_dim' but is rejected; pass it explicitly",
            "topk_idx_layout": "kernels want [total_q, heads, topk]; upstream triton uses "
                               "[heads, total_q, topk], so a transpose is part of the port",
            "selection_per_kv_head": "num_idx_heads == num_kv_heads upstream, so each kv head "
                                     "carries its own selection and no cross-head reduction happens",
        },
        "thresholds": {"min_cosine": args.cosine_floor, "max_relative_l2": args.max_relative_l2},
        "gate": "relative_l2 of sparse attention against float32 attention over the same blocks",
        "cases": [],
    }
    if args.topk >= blocks:
        result["state"] = "EVALUATION_INCONCLUSIVE"
        result["error"] = (
            f"topk {args.topk} covers all {blocks} blocks, so sparse and dense attention are the "
            "same computation and nothing about selection is tested"
        )
        print(json.dumps(result))
        return 1
    if args.heads % args.index_heads:
        result["state"] = "EVALUATION_INCONCLUSIVE"
        result["error"] = f"heads {args.heads} is not a multiple of index_heads {args.index_heads}"
        print(json.dumps(result))
        return 1

    index = build_cache(args.batch, args.index_heads, 1, args.head_dim,
                        args.page_size, args.context_len, args.seed, dtype)
    attend = build_cache(args.batch, args.heads, args.index_heads, args.head_dim,
                         args.page_size, args.context_len, args.seed + 1, dtype)
    return run(args, index, attend, blocks, result, kunlun_ops, torch, dtype)


def run(args, index, attend, blocks, result, kunlun_ops, torch, dtype) -> int:
    """Stage one: what a block score is, and what the top-k transform selects."""
    total_q = index["batch"]
    score = torch.zeros((total_q, args.index_heads, blocks), dtype=torch.float32, device="cuda")
    score_out = torch.zeros((total_q, args.index_heads, index["dim"]),
                            dtype=torch.float32, device="cuda")
    try:
        kunlun_ops.msa_block_score(
            q=index["q"], k_cache=index["k_cache"], v_cache=index["v_cache"],
            score=score, out=score_out, block_tables=index["block_tables"],
            batch_num=index["batch"], prefill_len=-1, max_seqlen_q=1,
            max_seqlen_k=index["context_len"], head_num=args.index_heads, head_num_kv=1,
            head_dim=index["dim"], head_dim_v=index["dim"], block_size_k=args.block_size,
            softmax_scale=index["scale"], max_blocks_per_seq=index["pages_per_seq"],
            page_block_size=index["page"], sink=None, score_type=0,
            use_tfloat32_gemm=False, **lods(index))
        torch.cuda.synchronize()
        distances = {
            how: metrics(score, reference_block_scores(index, args.block_size, how))
            for how in CANDIDATES
        }
        best = min(distances, key=lambda how: distances[how]["relative_l2"])
        result["cases"].append({
            "case": "block_score_reduction_semantics",
            "reference": "q·k per token from the index cache, reduced per block several ways, "
                         "float32 on the CPU",
            "relative_l2_by_reduction": {how: distances[how]["relative_l2"] for how in CANDIDATES},
            "best_reduction": best,
            "matches_a_known_reduction": distances[best]["relative_l2"] <= args.max_relative_l2,
            "applies_softmax_scale": distances.get("max", {}).get("relative_l2", 1.0)
            < distances.get("max_unscaled", {}).get("relative_l2", 0.0),
            **distances[best],
        })
        # Reported rather than gated: the docstring advertises an optional attention
        # output from the same call, and at score_type=0 it comes back untouched.
        result["cases"].append({
            "case": "block_score_optional_attention_output",
            "reference": "float32 full attention over the index cache",
            "output_is_all_zero": bool(score_out.abs().sum().item() == 0.0),
            **metrics(score_out, dense_reference(index)),
        })
    except Exception as error:
        result["state"] = "EVALUATION_ERROR"
        result["error"] = f"msa_block_score: {type(error).__name__}: {error}"
        print(json.dumps(result))
        return 1

    topk_idx = torch.full((total_q, args.index_heads, args.topk), -1,
                          dtype=torch.int32, device="cuda")
    prefix = (index["lengths_cpu"] - 1).to(torch.int32)
    try:
        kunlun_ops.msa_block_score_topk_transform(
            block_scores=score, context_qlens_cpu=index["cu_q_cpu"],
            context_qlens_xpu=index["cu_q_cpu"].to("cuda"),
            prefix_lens_cpu=prefix, prefix_lens_xpu=prefix.to("cuda"),
            topk_idx=topk_idx, topk=args.topk, block_size=args.block_size,
            max_num_blocks_per_seq=blocks, local_blocks=1)
        torch.cuda.synchronize()
        plain = set_agreement(topk_idx, reference_topk(score, args.topk, blocks, False))
        reserved = set_agreement(topk_idx, reference_topk(score, args.topk, blocks, True))
        result["cases"].append({
            "case": "topk_transform_vs_cpu_reference",
            "reference": "top-k over the kernel's own block scores, with and without a "
                         "reserved local block",
            "without_reserved_local_block": plain,
            "with_reserved_local_block": reserved,
            "reserves_a_local_block": reserved["exact_set_match_fraction"]
            > plain["exact_set_match_fraction"],
            "selection_understood": max(plain["exact_set_match_fraction"],
                                        reserved["exact_set_match_fraction"]) == 1.0,
            "invalid_slots": int((topk_idx < 0).sum()),
        })
    except Exception as error:
        result["cases"].append({
            "case": "topk_transform_vs_cpu_reference",
            "error": f"{type(error).__name__}: {error}",
        })
        # Selection is unavailable, so pick the blocks here: stage two still has
        # something to verify, and the report says the selection was not the kernel's.
        topk_idx = reference_topk(score, args.topk, blocks, False).to(torch.int32).to("cuda")
    return finish(args, attend, blocks, result, kunlun_ops, torch, dtype, topk_idx.contiguous())


def finish(args, case, blocks, result, kunlun_ops, torch, dtype, topk_idx) -> int:
    """Stage two: sparse attention over the selected blocks, and the control."""
    sparse_out = torch.zeros((case["batch"], case["heads"], case["dim"]),
                             dtype=dtype, device="cuda")
    try:
        kunlun_ops.msa_sparse_attention(
            q=case["q"], k_cache=case["k_cache"], v_cache=case["v_cache"],
            block_tables=case["block_tables"], topk_idx=topk_idx, out=sparse_out,
            topk_num=args.topk, topk_block_size=args.block_size, batch_num=case["batch"],
            head_num=case["heads"], head_num_kv=case["kv_heads"], head_dim=case["dim"],
            head_dim_v=case["dim"], max_seqlen_q=1, max_seqlen_k=case["context_len"],
            max_blocks_per_seq=case["pages_per_seq"], paged_block_size=case["page"],
            softmax_scale=case["scale"], prefill_len=-1, **lods(case))
        torch.cuda.synchronize()
        primary = {
            "case": "sparse_attention_vs_cpu_reference",
            "reference": "float32 attention restricted to the same selected blocks",
            **metrics(sparse_out, sparse_reference(case, topk_idx, args.block_size)),
        }
        # Reported alongside: if sparse attention matched dense attention this closely,
        # the geometry is not actually sparse and the gate would pass for free.
        primary["vs_dense_reference"] = metrics(sparse_out, dense_reference(case))["relative_l2"]
        result["cases"].append(primary)
    except Exception as error:
        result["state"] = "EVALUATION_ERROR"
        result["error"] = f"msa_sparse_attention: {type(error).__name__}: {error}"
        print(json.dumps(result))
        return 1

    # Control: grade the kernel against attention over a *different* block set. A
    # comparison that still passes is insensitive to selection and proves nothing.
    shifted = torch.where(topk_idx >= 0, (topk_idx + 1) % blocks, topk_idx)
    try:
        control = metrics(sparse_out, sparse_reference(case, shifted, args.block_size))
        control_error = None
    except Exception as error:
        control, control_error = {}, f"{type(error).__name__}: {error}"
    result["control"] = {
        "case": "reference_over_a_different_block_set",
        "description": "every selected block index shifted by one, so the reference reads other keys",
        **control,
        "error": control_error,
        "discriminates": control_error is not None
        or control.get("relative_l2", 0.0) > args.max_relative_l2,
    }

    primary = next(entry for entry in result["cases"]
                   if entry["case"] == "sparse_attention_vs_cpu_reference")
    passed = (primary["relative_l2"] <= args.max_relative_l2
              and primary["cosine"] >= args.cosine_floor)
    incomplete = [entry["case"] for entry in result["cases"] if entry.get("error")]
    if incomplete:
        # One kernel of the chain verified is not the chain verified, so a passing
        # primary case does not license EXERCISED_PASS.
        result["state"] = "EVALUATION_INCONCLUSIVE"
        result["incomplete_cases"] = incomplete
        result["error"] = f"{len(incomplete)} case(s) did not run: {', '.join(incomplete)}"
    elif not result["control"]["discriminates"]:
        result["state"] = "EVALUATION_INCONCLUSIVE"
    else:
        result["state"] = "EXERCISED_PASS" if passed else "EXERCISED_FAIL"
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





