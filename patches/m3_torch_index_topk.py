"""Torch replacements for MiniMax-M3's triton index kernels.

The sixth launch reached `Application startup complete` and then died on the first real
request inside the indexer:

    vllm/models/minimax_m3/common/ops/index_topk.py:680 _index_block_score_kernel
    triton.runtime.errors.OutOfResources: out of resource: shared memory,
        Required: 50180, Hardware limit: 49152

Triton does compile and run on P800 — the failure is a shared-memory budget, not a
missing path — but its coverage here is poor enough that tuning block sizes only moves
the problem to the next shape. So these three entry points get torch implementations,
the same way `patches/torch_paged_decode.py` replaced a vendor decode kernel for
Qwen3-8B.

Semantics are taken from the triton kernels they replace, and they are the ones the
MAT-008 block_sparse dimension already measured against the vendor kernels:

- a block score is the max over that block's 128 index-K positions of q·k, with the
  causal mask applied per query position (`tl.max(qk, axis=1)`, index_topk.py:154);
- the top-k keeps `init_blocks` at the start and `local_blocks` at the end regardless
  of score, and pads unused slots with -1;
- the index cache is `[num_blocks, 128, head_dim]`: one head, keys only, shared by all
  index heads. M3 sets num_idx_heads == num_kv_heads.

Layouts here are upstream's, not the vendor kernels': score is `[heads, total_q,
max_block]` and top-k is `[heads, total_q, topk]`, both head-major.
"""

from __future__ import annotations

import torch

SPARSE_BLOCK_SIZE = 128


def _gather_index_keys(cache: torch.Tensor, table_row: torch.Tensor, length: int):
    """The visible index keys of one request, [length, head_dim]."""
    blocks = (length + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    pages = table_row[:blocks].to(torch.long)
    keys = cache.index_select(0, pages).reshape(-1, cache.shape[-1])
    return keys[:length]


def index_score(idx_q, index_kv_cache, block_table, cu_seqlens_q, seq_lens,
                prefix_lens, max_query_len, max_seq_len, num_kv_heads):
    """[heads, total_q, max_block] block scores, float32."""
    total_q, num_idx_heads, head_dim = idx_q.shape
    assert num_idx_heads == num_kv_heads, "M3 expects num_idx_heads == num_kv_heads"
    max_block = (max_seq_len + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    score = idx_q.new_full((num_idx_heads, total_q, max_block), float("-inf"),
                           dtype=torch.float32)
    starts = cu_seqlens_q.tolist()
    lengths = seq_lens.tolist()
    prefixes = prefix_lens.tolist()
    for request in range(len(lengths)):
        lo, hi = starts[request], starts[request + 1]
        if hi <= lo:
            continue
        keys = _gather_index_keys(index_kv_cache, block_table[request],
                                  int(lengths[request])).to(torch.float32)
        q = idx_q[lo:hi].to(torch.float32)                      # [q, heads, dim]
        logits = torch.einsum("qhd,kd->hqk", q, keys)
        # Causal: query token i of this request sits at prefix + i.
        positions = torch.arange(hi - lo, device=q.device) + int(prefixes[request])
        invalid = torch.arange(keys.shape[0], device=q.device)[None, :] > positions[:, None]
        logits = logits.masked_fill(invalid[None, :, :], float("-inf"))
        blocks = (keys.shape[0] + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
        padded = blocks * SPARSE_BLOCK_SIZE - keys.shape[0]
        if padded:
            logits = torch.nn.functional.pad(logits, (0, padded), value=float("-inf"))
        tiled = logits.reshape(num_idx_heads, hi - lo, blocks, SPARSE_BLOCK_SIZE)
        score[:, lo:hi, :blocks] = tiled.amax(dim=-1)
    return score


def index_topk(score, cu_seqlens_q, prefix_lens, max_query_len, topk,
               init_blocks, local_blocks, out=None):
    """[heads, total_q, topk] int32 block ids, -1 in unused slots."""
    heads, total_q, max_block = score.shape
    result = score.new_full((heads, total_q, topk), -1, dtype=torch.int32)
    starts = cu_seqlens_q.tolist()
    prefixes = prefix_lens.tolist()
    for request in range(len(starts) - 1):
        lo, hi = starts[request], starts[request + 1]
        if hi <= lo:
            continue
        positions = torch.arange(hi - lo, device=score.device) + int(prefixes[request])
        # Blocks up to and including the one holding the query token.
        visible = (positions // SPARSE_BLOCK_SIZE) + 1
        for row in range(hi - lo):
            count = int(visible[row])
            keep = min(topk, count)
            forced: list[int] = list(range(min(init_blocks, count)))
            forced += [block for block in range(max(0, count - local_blocks), count)
                       if block not in forced]
            forced = forced[:keep]
            for head in range(heads):
                values = score[head, lo + row, :count].clone()
                for block in forced:
                    values[block] = float("inf")
                chosen = torch.topk(values, keep).indices.to(torch.int32)
                result[head, lo + row, :keep] = chosen
    if out is not None:
        out[:, :total_q, :].copy_(result)
        return out[:, :total_q, :]
    return result


def index_decode(idx_q, index_kv_cache, block_table, seq_lens, max_seq_len,
                 topk_blocks, init_blocks, local_blocks, num_kv_heads,
                 decode_query_len, max_decode_query_len, out=None):
    """Decode is prefill with one query per request and no prefix offset."""
    total_q = idx_q.shape[0]
    batch = total_q // max(1, decode_query_len)
    cu = torch.arange(0, total_q + 1, max(1, decode_query_len),
                      dtype=torch.int32, device=idx_q.device)
    lengths = seq_lens[:batch].to(torch.int32)
    prefixes = (lengths - decode_query_len).clamp_(min=0)
    score = index_score(idx_q, index_kv_cache, block_table, cu, lengths, prefixes,
                        decode_query_len, max_seq_len, num_kv_heads)
    return index_topk(score, cu, prefixes, decode_query_len, topk_blocks,
                      init_blocks, local_blocks, out=out)


def patch() -> None:
    """Rebind the three entry points, in the module and in the indexer that imported them."""
    from vllm.models.minimax_m3.common import indexer
    from vllm.models.minimax_m3.common.ops import index_topk as triton_module

    for module in (triton_module, indexer):
        module.minimax_m3_index_score = index_score          # type: ignore[attr-defined]
        module.minimax_m3_index_topk = index_topk            # type: ignore[attr-defined]
        module.minimax_m3_index_decode = index_decode        # type: ignore[attr-defined]
