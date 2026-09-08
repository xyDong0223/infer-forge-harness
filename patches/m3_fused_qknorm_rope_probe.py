"""Probe-only torch stand-in for `_C::fused_minimax_m3_qknorm_rope_kv_insert`.

MiniMax-M3's upstream implementation is vendored per backend
(`vllm/models/minimax_m3/{nvidia,amd,common}`) and the selector hands any non-ROCm
platform the nvidia variant. That variant calls exactly one vLLM custom op which the
Kunlun plugin does not register, so every attention layer dies with

    AttributeError: '_OpNamespace' '_C' object has no attribute
                    'fused_minimax_m3_qknorm_rope_kv_insert'

This registers a torch implementation of all three branches the op has: the *dense*
one (per-head Gemma RMSNorm on q and k, then partial NeoX RoPE), the *paged insert*
(normed and roped k/v scattered into the main cache by slot_mapping) and the
*lightning index* one (index_q/index_k read out of the same fused tensor, normed and
roped, with index_k scattered into the index cache). It is still not a port — the
arithmetic is torch, not kunlun_ops — but it is verified rather than assumed:
`tools/probe/m3_qknorm_rope_insert_probe.py` reads every value back out of the caches
and grades it against a float32 reference, with a control that omits RoPE.

What it refuses is a quantized cache (`kv_cache_dtype` other than "auto"), because
writing unconverted values into one is wrong quietly instead of loudly.

Semantics come from the wrapper's own docstring at `vllm/_custom_ops.py:2465` and from
the AMD triton norm (`out = x * rstd * (1.0 + w)`, fp32). The cache layout is the
platform's: `(2, num_blocks, num_kv_heads, block_size, head_size)`.
"""

from __future__ import annotations

import torch

SCHEMA = (
    "fused_minimax_m3_qknorm_rope_kv_insert("
    "Tensor qkv, Tensor q_norm_weight, Tensor k_norm_weight, Tensor cos_sin_cache, "
    "Tensor positions, int num_heads, int num_kv_heads, int rotary_dim, float eps, "
    "Tensor? index_q_norm_weight, Tensor? index_k_norm_weight, int num_index_heads, "
    "Tensor? slot_mapping, Tensor? index_slot_mapping, Tensor? kv_cache, "
    "Tensor? index_cache, int block_size, Tensor? q_out, Tensor? index_q_out, "
    "str kv_cache_dtype) -> ()"
)


def _gemma_norm_per_head(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """x is [N, heads, head_dim]; weight is [head_dim], shared across heads."""
    f = x.float()
    rstd = torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)
    return (f * rstd * (1.0 + weight.float())).to(x.dtype)


def _neox_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                       rotary_dim: int) -> torch.Tensor:
    """Rotate the first `rotary_dim` channels of each head, NeoX halves layout.

    cos and sin are [N, rotary_dim // 2]; x is [N, heads, head_dim].
    """
    rotated = x[..., :rotary_dim]
    passthrough = x[..., rotary_dim:]
    half = rotary_dim // 2
    first, second = rotated[..., :half].float(), rotated[..., half:].float()
    cos = cos[:, None, :].float()
    sin = sin[:, None, :].float()
    out = torch.cat([first * cos - second * sin, second * cos + first * sin], dim=-1)
    return torch.cat([out.to(x.dtype), passthrough], dim=-1)


def _slots(slot_mapping: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    flat = slot_mapping.view(-1).to(torch.long)
    return flat // block_size, flat % block_size


def _insert(cache: torch.Tensor, which: int, values: torch.Tensor,
            slot_mapping: torch.Tensor, block_size: int) -> None:
    """Scatter [N, heads, dim] into a paged cache by slot, layout detected not assumed.

    Measured the hard way. vLLM-Kunlun's own paged attention allocates
    ``(2, num_blocks, num_kv_heads, block_size, head_size)`` — BHLD
    (`vllm_kunlun/ops/paged_attn.py:43`), which is also what the msa_* kernels read — so
    that layout was hardcoded here. The fifth M3 launch then died with `index 127 is out
    of bounds for dimension 2 with size 1`: M3's caches come out of upstream's own spec
    as ``(2, num_blocks, block_size, num_kv_heads, head_size)``, BLHD, with block_size
    before the heads. Both layouts exist in one process, so the layout has to be read
    off the tensor rather than assumed.
    """
    blocks, offsets = _slots(slot_mapping, block_size)
    target = cache[which] if cache.dim() == 5 else cache
    if target.dim() == 3:
        # The index cache, exactly as upstream declares it: [num_blocks, 128, head_dim],
        # one implicit head and keys only. Measured on the sixth launch as
        # (6781, 128, 128), which is why this rank is handled rather than refused.
        if values.shape[1] != 1:
            raise NotImplementedError(
                f"a 3-D cache holds one head; got {values.shape[1]}"
            )
        stored = values.reshape(values.shape[0], -1).to(target.dtype)
        for token in range(stored.shape[0]):
            target[blocks[token], offsets[token], :] = stored[token]
        return
    if target.dim() != 4:
        raise NotImplementedError(
            f"unsupported cache rank {tuple(cache.shape)}; expected 3, 4 or 5 dimensions"
        )
    heads = values.shape[1]
    _, first, second, _ = target.shape
    if first == heads and second == block_size and heads != block_size:
        head_major = True
    elif first == block_size and second == heads and heads != block_size:
        head_major = False
    else:
        raise NotImplementedError(
            f"cannot tell BHLD from BLHD for {tuple(target.shape)} with {heads} heads and "
            f"block_size {block_size}; the two are ambiguous when they are equal"
        )
    stored = values.to(target.dtype)
    for token in range(stored.shape[0]):
        if head_major:
            target[blocks[token], :, offsets[token], :] = stored[token]
        else:
            target[blocks[token], offsets[token], :, :] = stored[token]


def fused_minimax_m3_qknorm_rope_kv_insert(
    qkv: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    rotary_dim: int,
    eps: float,
    index_q_norm_weight: torch.Tensor | None = None,
    index_k_norm_weight: torch.Tensor | None = None,
    num_index_heads: int = 0,
    slot_mapping: torch.Tensor | None = None,
    index_slot_mapping: torch.Tensor | None = None,
    kv_cache: torch.Tensor | None = None,
    index_cache: torch.Tensor | None = None,
    block_size: int = 0,
    q_out: torch.Tensor | None = None,
    index_q_out: torch.Tensor | None = None,
    kv_cache_dtype: str = "auto",
) -> None:
    if kv_cache_dtype not in ("auto", ""):
        # A quantized cache is a conversion this stand-in does not do, and writing
        # unconverted values into it would be silently wrong rather than loud.
        raise NotImplementedError(
            f"kv_cache_dtype={kv_cache_dtype!r} needs the cache conversion path"
        )

    head_dim = q_norm_weight.shape[-1]
    q_size, kv_size = num_heads * head_dim, num_kv_heads * head_dim
    sizes = [q_size, kv_size, kv_size]
    index_dim = 0
    if num_index_heads:
        if index_q_norm_weight is None or index_k_norm_weight is None:
            raise ValueError("a sparse layer needs both index norm weights")
        index_dim = index_q_norm_weight.shape[-1]
        # The index branch is read straight out of the same fused tensor:
        # [q | k | v | index_q | index_k], and index_k is single-head.
        sizes += [num_index_heads * index_dim, index_dim]
    parts = qkv.split(sizes, dim=-1)
    q, k, v = parts[0], parts[1], parts[2]

    cos_sin = cos_sin_cache.index_select(0, positions.view(-1).to(torch.long))
    cos, sin = cos_sin.chunk(2, dim=-1)

    q_heads = _gemma_norm_per_head(q.view(-1, num_heads, head_dim), q_norm_weight, eps)
    k_heads = _gemma_norm_per_head(k.view(-1, num_kv_heads, head_dim), k_norm_weight, eps)
    q_heads = _neox_partial_rope(q_heads, cos, sin, rotary_dim)
    k_heads = _neox_partial_rope(k_heads, cos, sin, rotary_dim)

    # In place, because the caller re-splits the same tensor afterwards.
    q.copy_(q_heads.reshape(q.shape))
    k.copy_(k_heads.reshape(k.shape))
    if q_out is not None:
        q_out.copy_(q_heads.reshape(q_out.shape))

    index_k_heads = None
    if num_index_heads:
        index_q, index_k = parts[3], parts[4]
        index_q_heads = _gemma_norm_per_head(
            index_q.view(-1, num_index_heads, index_dim), index_q_norm_weight, eps)
        index_k_heads = _gemma_norm_per_head(
            index_k.view(-1, 1, index_dim), index_k_norm_weight, eps)
        index_q_heads = _neox_partial_rope(index_q_heads, cos, sin, rotary_dim)
        index_k_heads = _neox_partial_rope(index_k_heads, cos, sin, rotary_dim)
        index_k.copy_(index_k_heads.reshape(index_k.shape))
        if index_q_out is not None:
            index_q_out.copy_(index_q_heads.reshape(index_q_out.shape).to(index_q_out.dtype))
        else:
            index_q.copy_(index_q_heads.reshape(index_q.shape))

    if kv_cache is not None and kv_cache.numel():
        if slot_mapping is None or not block_size:
            raise ValueError("inserting into a paged cache needs slot_mapping and block_size")
        _insert(kv_cache, 0, k_heads, slot_mapping, block_size)
        _insert(kv_cache, 1, v.reshape(-1, num_kv_heads, head_dim), slot_mapping, block_size)
    if index_cache is not None and index_cache.numel():
        if index_k_heads is None:
            raise ValueError("an index cache was given but num_index_heads is 0")
        # Upstream: "if index_slot_mapping is omitted, slot_mapping is used for both".
        mapping = index_slot_mapping if index_slot_mapping is not None else slot_mapping
        if mapping is None or not block_size:
            raise ValueError("inserting into the index cache needs a slot mapping")
        _insert(index_cache, 0, index_k_heads, mapping, block_size)


def register() -> None:
    """Define the op in the `_C` namespace the nvidia model reaches for."""
    library = torch.library.Library("_C", "FRAGMENT")
    library.define(SCHEMA)
    for backend in ("CUDA", "CPU"):
        library.impl("fused_minimax_m3_qknorm_rope_kv_insert",
                    fused_minimax_m3_qknorm_rope_kv_insert, backend)
    # Held so the Library object is not garbage collected, which would drop the op.
    globals()["_LIBRARY"] = library
