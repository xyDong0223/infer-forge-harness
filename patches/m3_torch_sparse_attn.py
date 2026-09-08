"""Torch replacements for MiniMax-M3's triton block-sparse attend kernels.

The seventh launch got past the indexer (`patches/m3_torch_index_topk.py`) and died one
step later, in the attend:

    vllm/models/minimax_m3/common/ops/sparse_attn.py:439 _gqa_sparse_fwd_kernel
    triton.runtime.errors.OutOfResources: out of resource: shared memory,
        Required: 69636, Hardware limit: 49152

Same reason as the indexer, worse margin — 42% over budget rather than 2% — which is why
tuning block sizes is not the route on P800.

The arithmetic here is the one MAT-008's block_sparse dimension already measured against
the vendor kernels (`tools/probe/block_sparse_attention_probe.py:sparse_reference`), with
three things the probe did not need and serving does: GQA groups sharing one kv head's
selection, requests segmented by cu_seqlens, and the causal mask.

Two layouts are read off the tensors rather than assumed, both measured the hard way:
the main cache is `(2, num_blocks, block_size, num_kv_heads, head_dim)` here while
upstream's comments say `(num_blocks, 2, 128, ...)`, and topk_idx is head-major
`[num_kv_heads, total_q, topk]` on this side of the port.
"""

from __future__ import annotations

import os

import torch

SPARSE_BLOCK_SIZE = 128
# A differential switch, not a feature. With M3_FORCE_DENSE_ATTEND=1 the attend ignores
# the selection and reads the whole visible context, which is mathematically a superset
# of any block selection. If the served output becomes sane under it, the fault is in the
# selection (or in how it is oriented); if the output stays garbled, the fault is upstream
# of the attend — the qk-norm, the rope, or the cache writes.
FORCE_DENSE = os.environ.get("M3_FORCE_DENSE_ATTEND") == "1"


_reported = {"layout": False}


def _split_kv(kv_cache: torch.Tensor):
    """Return the (k, v) halves of the paged cache, whichever axis carries the pair."""
    if kv_cache.dim() == 5 and kv_cache.shape[0] == 2:
        return kv_cache[0], kv_cache[1]
    if kv_cache.dim() == 5 and kv_cache.shape[1] == 2:
        return kv_cache[:, 0], kv_cache[:, 1]
    raise NotImplementedError(f"unsupported kv cache shape {tuple(kv_cache.shape)}")


def _paged_layout(kv_cache: torch.Tensor, num_kv_heads: int):
    """(block_size, head_major) for the paged cache, with the block size from the config.

    Do not try to read the layout off the tensor: at TP=8 this rank holds a single kv head,
    which makes (num_blocks, num_kv_heads, block_size, head_dim) and
    (num_blocks, block_size, num_kv_heads, head_dim) shape-identical apart from a size-1
    axis. The dense stand-in found this out the hard way -- its cache turned out to be
    head-major (BHLD) while this comment previously asserted BLHD for both.
    """
    from vllm.config import get_current_vllm_config

    block_size = None
    try:
        block_size = int(get_current_vllm_config().cache_config.block_size)
    except Exception:
        block_size = None
    half = _split_kv(kv_cache)[0]
    if block_size is None:
        # get_current_vllm_config() is only set while the model is being built, not during
        # a forward, so fall back to the axis that cannot be the head axis. With one kv
        # head per rank exactly one of the two candidate axes is != num_kv_heads, which is
        # enough to disambiguate -- and it matches the convention the model itself uses
        # when it passes `self.kv_cache.size(2)` as the paged block size.
        if half.shape[1] == num_kv_heads and half.shape[2] != num_kv_heads:
            block_size = int(half.shape[2])
        elif half.shape[2] == num_kv_heads and half.shape[1] != num_kv_heads:
            block_size = int(half.shape[1])
        else:
            raise NotImplementedError(
                f"cannot infer the paged block size from {tuple(kv_cache.shape)}"
            )
    if half.shape[1] == num_kv_heads and half.shape[2] == block_size:
        head_major = True
    elif half.shape[1] == block_size and half.shape[2] == num_kv_heads:
        head_major = False
    else:
        raise NotImplementedError(
            f"cache layout {tuple(half.shape)} matches neither heads={num_kv_heads} "
            f"block_size={block_size} ordering"
        )
    if not _reported["layout"]:
        _reported["layout"] = True
        print(
            f"[M3_SPARSE_ATTN] cache {tuple(kv_cache.shape)} block_size={block_size} "
            f"head_major={head_major}",
            flush=True,
        )
    return block_size, head_major


def _positions(table_row: torch.Tensor, blocks: list[int], length: int) -> list[int]:
    """Absolute token positions covered by the selected blocks, clipped to length."""
    positions: list[int] = []
    for block in blocks:
        if block < 0:
            continue
        start = block * SPARSE_BLOCK_SIZE
        positions.extend(range(start, min(start + SPARSE_BLOCK_SIZE, length)))
    return sorted(set(positions))


def _attend_row(q_row, k_cache, v_cache, table_row, selected, length, position,
                num_kv_heads, group, scale, out_row, block_size, head_major):
    """One query token: per kv head, attend over that head's selected blocks.

    The *selection* is in units of SPARSE_BLOCK_SIZE (the model's sparse_block_size); the
    *cache* is paged in units of block_size from the config. They happen to be equal at 128
    in this deployment, which is exactly why keeping them separate matters.
    """
    for head in range(num_kv_heads):
        if FORCE_DENSE:
            positions = list(range(min(length, position + 1)))
        else:
            positions = [p for p in _positions(table_row, selected[head], length)
                         if p <= position]
        if not positions:
            continue
        index = torch.tensor(positions, device=q_row.device)
        pages = (index // block_size).to(torch.long)
        offsets = (index % block_size).to(torch.long)
        physical = table_row.to(torch.long)[pages]
        if head_major:
            keys = k_cache[physical, head, offsets, :].to(torch.float32)
            values = v_cache[physical, head, offsets, :].to(torch.float32)
        else:
            keys = k_cache[physical, offsets, head, :].to(torch.float32)
            values = v_cache[physical, offsets, head, :].to(torch.float32)
        lo, hi = head * group, (head + 1) * group
        queries = q_row[lo:hi].to(torch.float32)
        logits = (queries @ keys.transpose(0, 1)) * scale
        weights = torch.softmax(logits, dim=-1)
        out_row[lo:hi] = (weights @ values).to(out_row.dtype)


def _selection_view(topk_idx: torch.Tensor, total_q: int, num_kv_heads: int):
    """Return a [total_q, num_kv_heads, topk] view, whichever way round it arrived.

    Upstream's triton kernels document topk_idx as head-major [num_kv_heads, total_q,
    topk], but the MSA path allocates a *token-major* shared buffer ([total_q, H, MK])
    and hands out strided views of it, so both orders reach this function. The eighth
    launch found that out the hard way: `index 1 is out of bounds for dimension 1 with
    size 1`. Rather than pick one, match the dimensions against the two known extents,
    and if neither fits, say what arrived instead of indexing into it blindly.
    """
    shape = tuple(topk_idx.shape)
    if len(shape) != 3:
        raise NotImplementedError(f"topk_idx must be 3-D, got {shape}")
    # Measured on the ninth launch: what arrives is the whole persistent buffer,
    # (max_num_batched_tokens, num_kv_heads, topk) = (8192, 1, 16), for a decode with
    # total_q=1. So dim 0 is token-major *capacity*, not this call's token count, and the
    # caller's `topk[:, :nd, :]` slices the head dim rather than the tokens — harmless
    # only because num_kv_heads is 1 per rank at TP=8. This call's tokens are the leading
    # rows, so take them and ignore the rest of the buffer.
    if shape[1] == num_kv_heads and shape[0] >= total_q:
        return topk_idx[:total_q]
    if shape[0] == num_kv_heads and shape[1] >= total_q:
        return topk_idx.permute(1, 0, 2)[:total_q]
    raise NotImplementedError(
        f"cannot orient topk_idx {shape} against total_q={total_q} and "
        f"num_kv_heads={num_kv_heads}"
    )


def sparse_attn(q, kv_cache, topk_idx, block_table, cu_seqlens_q, seq_lens,
                prefix_lens, max_query_len, num_kv_heads, sm_scale, output) -> None:
    """Prefill: several query tokens per request, causal within the request."""
    # The caller hands in an uninitialised `torch.empty_like(q)`. Rows and heads that end
    # up with no visible keys are skipped by `_attend_row`, so without this the attend
    # returns whatever was in that memory -- measured as NaN by
    # `patches/m3_sparse_attend_selfcheck.py` (got_norm=nan against a finite reference).
    # Zero is the right value for an unattended row.
    output.zero_()
    k_cache, v_cache = _split_kv(kv_cache)
    block_size, head_major = _paged_layout(kv_cache, num_kv_heads)
    group = q.shape[1] // num_kv_heads
    selection = _selection_view(topk_idx, q.shape[0], num_kv_heads)
    if not _reported.get("scale"):
        _reported["scale"] = True
        # The attend stand-in uses whatever sm_scale the impl passes. If that is not
        # head_dim**-0.5 the magnitude of the output shifts, which is exactly the kind of
        # gap the cache-free differential reports (got_norm ~1.9x ref_norm).
        print(
            f"[M3_SPARSE_ATTN] sm_scale={sm_scale} head_dim={q.shape[-1]} "
            f"heads={q.shape[1]}/{num_kv_heads} topk_idx={tuple(topk_idx.shape)}",
            flush=True,
        )
    starts = cu_seqlens_q.tolist()
    lengths = seq_lens.tolist()
    prefixes = prefix_lens.tolist()
    for request in range(len(starts) - 1):
        lo, hi = starts[request], starts[request + 1]
        for row in range(hi - lo):
            token = lo + row
            selected = [selection[token, head].tolist() for head in range(num_kv_heads)]
            _attend_row(q[token], k_cache, v_cache, block_table[request], selected,
                        int(lengths[request]), int(prefixes[request]) + row,
                        num_kv_heads, group, sm_scale, output[token],
                        block_size, head_major)


def sparse_attn_decode(q, kv_cache, topk_idx, block_table, seq_lens, num_kv_heads,
                       sm_scale, output, decode_query_len) -> None:
    """Decode: `decode_query_len` query tokens per request, the last ones in the sequence."""
    output.zero_()  # same uninitialised-output hazard as the prefill entry point
    k_cache, v_cache = _split_kv(kv_cache)
    block_size, head_major = _paged_layout(kv_cache, num_kv_heads)
    group = q.shape[1] // num_kv_heads
    selection = _selection_view(topk_idx, q.shape[0], num_kv_heads)
    lengths = seq_lens.tolist()
    for request in range(len(lengths)):
        length = int(lengths[request])
        for row in range(decode_query_len):
            token = request * decode_query_len + row
            selected = [selection[token, head].tolist() for head in range(num_kv_heads)]
            # The query tokens are the tail of the sequence.
            position = length - decode_query_len + row
            _attend_row(q[token], k_cache, v_cache, block_table[request], selected,
                        length, position, num_kv_heads, group, sm_scale, output[token],
                        block_size, head_major)


def patch() -> None:
    """Rebind in the ops module and in every module that imported the names directly."""
    from vllm.models.minimax_m3.common import sparse_attention
    from vllm.models.minimax_m3.common.ops import sparse_attn as triton_module

    targets = [triton_module, sparse_attention]
    try:
        from vllm.models.minimax_m3.nvidia import sparse_attention_msa
        targets.append(sparse_attention_msa)
    except Exception:
        pass
    for module in targets:
        module.minimax_m3_sparse_attn = sparse_attn                # type: ignore[attr-defined]
        module.minimax_m3_sparse_attn_decode = sparse_attn_decode  # type: ignore[attr-defined]
