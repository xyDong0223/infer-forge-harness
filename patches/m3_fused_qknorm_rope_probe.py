"""Probe-only torch stand-in for `_C::fused_minimax_m3_qknorm_rope_kv_insert`.

MiniMax-M3's upstream implementation is vendored per backend
(`vllm/models/minimax_m3/{nvidia,amd,common}`) and the selector hands any non-ROCm
platform the nvidia variant. That variant calls exactly one vLLM custom op which the
Kunlun plugin does not register, so every attention layer dies with

    AttributeError: '_OpNamespace' '_C' object has no attribute
                    'fused_minimax_m3_qknorm_rope_kv_insert'

This registers a torch implementation of the *dense* branch — per-head Gemma RMSNorm
on q and k, then partial NeoX RoPE — which is all the leading dense layers use. It is
not a port: it exists to answer what breaks next, and it deliberately refuses the
paged-cache and index branches instead of guessing a cache layout, so the sparse
layers fail loudly rather than quietly computing the wrong thing.

Semantics come from the wrapper's own docstring at `vllm/_custom_ops.py:2465` and from
the AMD triton norm (`out = x * rstd * (1.0 + w)`, fp32).
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
    if kv_cache is not None or index_cache is not None or num_index_heads:
        raise NotImplementedError(
            "this probe implements the dense branch only; the paged-cache and "
            "lightning-index branches need the real cache layout and belong to a port"
        )

    head_dim = q_norm_weight.shape[-1]
    q_size, kv_size = num_heads * head_dim, num_kv_heads * head_dim
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

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


def register() -> None:
    """Define the op in the `_C` namespace the nvidia model reaches for."""
    library = torch.library.Library("_C", "FRAGMENT")
    library.define(SCHEMA)
    for backend in ("CUDA", "CPU"):
        library.impl("fused_minimax_m3_qknorm_rope_kv_insert",
                    fused_minimax_m3_qknorm_rope_kv_insert, backend)
    # Held so the Library object is not garbage collected, which would drop the op.
    globals()["_LIBRARY"] = library
