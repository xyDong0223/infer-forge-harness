"""Torch replacement for the platform's dense attention in MiniMax-M3's leading layers.

Measured, with no golden weights and no tensor-parallel assumptions
(`patches/m3_dense_attn_selfcheck.py`): during a fresh 5-token prefill the platform's
`Attention.forward` output is off by relL2 **0.326 / 0.438 / 0.076** for layers 0/1/2
against causal attention over the very q/k/v it was handed. The same instrument puts the
torch block-sparse attend at 1-2%. Three broken dense layers feed 60 sparse ones, which is
why the service answered with a prompt-independent LaTeX-ish token.

Everything upstream of it is verified: the input to attention matches a from-checkpoint
fp32 golden at relL2 0.0017, and the qk-norm + partial-RoPE stand-in matches vLLM's own
rotary module bit-for-bit. Every structural explanation for the gap was swept and
eliminated (conventions, causality, scale, kv-head-to-rank mapping, reading past the
sequence bound, sliding windows, V corruption, int8 activation quant, fp8 cache), so this
does not try to fix the kernel -- it replaces it, the same way the triton indexer and
attend were replaced.

Scope: the dense layers only. It writes k/v into the same paged cache the platform would
have written, so decode steps and prefix reuse keep working, and it reads the whole visible
context (dense attention has no block selection).

This is a stand-in, not a port: it is a per-row python loop, correct but slow. The durable
fix belongs in the vendor kernel.
"""

from __future__ import annotations

import os

import torch

ENABLED = os.environ.get("M3_TORCH_DENSE_ATTN", "1") != "0"
# Grade this stand-in with the same instrument that condemned the kernel it replaces: for a
# fresh-sequence prefill, causal attention over the q/k/v of this very call, computed with
# no cache at all. Anything that reads the paged cache is being tested, not trusted.
SELFTEST = os.environ.get("M3_DENSE_SELFTEST") == "1"
SELFTEST_CALLS = int(os.environ.get("M3_DENSE_SELFTEST_CALLS", "3"))

_selftest = {"done": 0}


def _grade(layer, q, k, v, produced, metadata):
    tokens = q.shape[0]
    seq_lens = metadata.seq_lens_tensor.tolist()
    if len(seq_lens) != 1 or int(seq_lens[0]) != tokens:
        print(
            f"[M3_DENSE_SELFTEST] skipped: {len(seq_lens)} requests, seq_lens={seq_lens}, "
            f"tokens={tokens} -- not a single fresh prefill",
            flush=True,
        )
        return
    head_dim = layer.head_dim
    group = layer.num_heads // layer.num_kv_heads
    qq = q.reshape(tokens, layer.num_heads, head_dim).float()
    kk = k.reshape(tokens, layer.num_kv_heads, head_dim).float()
    vv = v.reshape(tokens, layer.num_kv_heads, head_dim).float()
    mask = torch.triu(
        torch.full((tokens, tokens), float("-inf"), device=qq.device), diagonal=1
    )
    reference = torch.empty_like(qq)
    for head in range(layer.num_kv_heads):
        queries = qq[:, head * group : (head + 1) * group]
        logits = torch.einsum("tgd,sd->gts", queries, kk[:, head]) * layer.scaling
        weights = torch.softmax(logits + mask, dim=-1)
        reference[:, head * group : (head + 1) * group] = torch.einsum(
            "gts,sd->tgd", weights, vv[:, head]
        )
    reference = reference.reshape(tokens, -1)
    got = produced.detach().float().reshape(tokens, -1)
    delta = float((got - reference).norm() / reference.norm().clamp_min(1e-30))
    print(
        f"[M3_DENSE_SELFTEST] tokens={tokens} relL2={delta:.6g} "
        f"got_norm={float(got.norm()):.6g} ref_norm={float(reference.norm()):.6g}",
        flush=True,
    )


def _split_kv(kv_cache: torch.Tensor):
    """Return (k, v) views, each [num_blocks, block_size, num_kv_heads, head_dim]."""
    if kv_cache.dim() == 5 and kv_cache.shape[0] == 2:
        return kv_cache[0], kv_cache[1]
    if kv_cache.dim() == 5 and kv_cache.shape[1] == 2:
        return kv_cache[:, 0], kv_cache[:, 1]
    raise NotImplementedError(f"unsupported kv cache shape {tuple(kv_cache.shape)}")


def _head_major(target: torch.Tensor, heads: int, block_size: int) -> bool:
    """Layouts are detected, not assumed -- but with one kv head per rank at TP=8 the two
    layouts are shape-identical, so the block size has to come from the config rather than
    from the tensor. Measured: the dense layers' cache is
    (2, num_blocks, num_kv_heads, block_size, head_dim) -- BHLD, what
    vllm_kunlun/ops/paged_attn.py:43 allocates -- while the sparse layers' M3-allocated
    cache is BLHD. Both live in one process.
    """
    _, first, second, _ = target.shape
    if first == heads and second == block_size:
        return True
    if first == block_size and second == heads:
        return False
    raise NotImplementedError(
        f"cache layout {tuple(target.shape)} matches neither heads={heads} "
        f"block_size={block_size} ordering"
    )


def _block_size(kv_cache: torch.Tensor, num_kv_heads: int) -> int:
    from vllm.config import get_current_vllm_config

    try:
        configured = get_current_vllm_config().cache_config.block_size
    except Exception:
        configured = None
    if configured:
        return int(configured)
    # Fall back to the unambiguous case only.
    half = kv_cache[0] if kv_cache.shape[0] == 2 else kv_cache[:, 0]
    if half.shape[1] == num_kv_heads and half.shape[2] != num_kv_heads:
        return int(half.shape[2])
    if half.shape[2] == num_kv_heads and half.shape[1] != num_kv_heads:
        return int(half.shape[1])
    raise NotImplementedError(
        f"cannot infer the paged block size from {tuple(kv_cache.shape)}"
    )


def _write(cache_half, values, slot_mapping, block_size, head_major):
    blocks = (slot_mapping // block_size).to(torch.long)
    offsets = (slot_mapping % block_size).to(torch.long)
    stored = values.to(cache_half.dtype)
    for token in range(stored.shape[0]):
        if slot_mapping[token] < 0:
            continue
        if head_major:
            cache_half[blocks[token], :, offsets[token], :] = stored[token]
        else:
            cache_half[blocks[token], offsets[token], :, :] = stored[token]


def _read(cache_half, table_row, positions, head, block_size, head_major):
    index = torch.as_tensor(positions, device=cache_half.device)
    pages = (index // block_size).to(torch.long)
    offsets = (index % block_size).to(torch.long)
    physical = table_row.to(torch.long)[pages]
    if head_major:
        return cache_half[physical, head, offsets, :]
    return cache_half[physical, offsets, head, :]


def attend(layer, q, k, v, kv_cache, metadata):
    """Full causal attention over the paged cache, one query row at a time."""
    num_kv_heads = layer.num_kv_heads
    head_dim = layer.head_dim
    tokens = q.shape[0]
    q = q.reshape(tokens, layer.num_heads, head_dim)
    k = k.reshape(tokens, num_kv_heads, head_dim)
    v = v.reshape(tokens, num_kv_heads, head_dim)
    group = layer.num_heads // num_kv_heads

    k_cache, v_cache = _split_kv(kv_cache)
    block_size = _block_size(kv_cache, num_kv_heads)
    head_major = _head_major(k_cache, num_kv_heads, block_size)

    slot_mapping = metadata.slot_mapping.reshape(-1)[:tokens]
    _write(k_cache, k, slot_mapping, block_size, head_major)
    _write(v_cache, v, slot_mapping, block_size, head_major)

    starts = metadata.query_start_loc
    if starts is None:
        # Decode-only batches carry no query_start_loc: one query token per request.
        requests = metadata.seq_lens_tensor.shape[0]
        starts = list(range(requests + 1))
    else:
        starts = starts.tolist()
    seq_lens = metadata.seq_lens_tensor.tolist()
    block_tables = metadata.block_tables
    output = torch.zeros_like(q)
    scale = layer.scaling

    for request in range(len(starts) - 1):
        lo, hi = int(starts[request]), int(starts[request + 1])
        if hi <= lo:
            continue
        length = int(seq_lens[request])
        context = length - (hi - lo)
        table_row = block_tables[request]
        for row in range(hi - lo):
            token = lo + row
            position = context + row
            positions = list(range(position + 1))
            for head in range(num_kv_heads):
                keys = _read(k_cache, table_row, positions, head, block_size, head_major)
                values = _read(v_cache, table_row, positions, head, block_size, head_major)
                queries = q[token, head * group : (head + 1) * group].float()
                logits = (queries @ keys.float().transpose(0, 1)) * scale
                weights = torch.softmax(logits, dim=-1)
                output[token, head * group : (head + 1) * group] = (
                    weights @ values.float()
                ).to(output.dtype)

    return output.reshape(tokens, layer.num_heads * head_dim)


def _bound_cache(attn, virtual_engine):
    """The layer's paged cache, or None while it is still a placeholder.

    Before `bind_kv_cache` runs, `Attention.kv_cache` is a per-virtual-engine list whose
    entries are zero-element tensors -- and indexing a zero-element tensor is itself the
    IndexError that killed launch 21, so probe shapes before indexing.
    """
    cache = getattr(attn, "kv_cache", None)
    if isinstance(cache, (list, tuple)):
        if len(cache) <= virtual_engine:
            return None
        cache = cache[virtual_engine]
    if not isinstance(cache, torch.Tensor) or cache.dim() < 4 or cache.numel() == 0:
        return None
    return cache


def _is_dummy(metadata) -> bool:
    """Warmup batches arrive with the metadata fields present but empty.

    Launch 21 died on `index 0 is out of bounds for dimension 0 with size 0` because the
    profiling run's block table is empty; checking the cache alone is not enough.
    """
    for name in ("block_tables", "seq_lens_tensor", "slot_mapping"):
        value = getattr(metadata, name, None)
        if value is None or value.numel() == 0:
            return True
    return False


def patch() -> None:
    if not ENABLED:
        return
    from vllm.forward_context import get_forward_context
    from vllm.models.minimax_m3.nvidia import model as m3

    original = m3.MiniMaxM3Attention.forward

    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        torch.ops._C.fused_minimax_m3_qknorm_rope_kv_insert(
            qkv,
            self.q_norm.weight,
            self.k_norm.weight,
            self.rotary_emb.cos_sin_cache,
            positions,
            self.num_heads,
            self.num_kv_heads,
            self.rotary_emb.rotary_dim,
            self.q_norm.variance_epsilon,
            None, None, 0, None, None, None, None, 0, None, None, "auto",
        )
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        context = get_forward_context()
        metadata = getattr(context, "attn_metadata", None)
        if isinstance(metadata, dict):
            metadata = metadata.get(self.attn.layer_name)
        cache = _bound_cache(self.attn, getattr(context, "virtual_engine", 0))
        if metadata is None or cache is None or _is_dummy(metadata):
            # Memory-profiling / warmup run: the caches are not bound and the metadata
            # carries empty block tables. Fall back rather than invent a layout, exactly
            # as the fused-insert stand-in does. Real errors are left to raise.
            return original(self, positions, hidden_states)

        attn_output = attend(self, q, k, v, cache, metadata)
        if SELFTEST and _selftest["done"] < SELFTEST_CALLS:
            _selftest["done"] += 1
            _grade(self, q, k, v, attn_output, metadata)
        output, _ = self.o_proj(attn_output)
        return output

    m3.MiniMaxM3Attention.forward = forward
