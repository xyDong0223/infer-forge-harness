"""Pure-torch paged decode attention for the Kunlun attention backend.

Why this exists: on P800 with vLLM-Kunlun 0.25.1.dev0 both vendor decode kernels
fail *inside the server* at Qwen3-8B's decode geometry while accepting identical
arguments in isolation —

  kunlun_ops.speculative_attention   -> ValueError: Check 0 == ret failed
  kunlun_ops.decode_paged_attention  -> [xops_block_ERROR] ... line: 271

Semantics follow HF transformers' Qwen3 attention: GQA by grouping query heads
onto their KV head, softmax in float32, and masking by each request's context
length. The layout is vLLM-Kunlun's own paged cache,
`(num_blocks, num_kv_heads, block_size, head_size)`.

Model coverage is deliberately narrow and enforced rather than assumed: sliding
windows, attention sinks and speculative decode raise `UnsupportedDecode` so the
caller keeps the vendor kernel instead of getting a quietly different result.
Plain MHA/GQA models with a paged cache — Qwen3, Llama-style, most dense decoders
— are covered. MLA models are not: they use a different cache layout and their own
kernel family.

Install into a running pod with `tools/apply_torch_decode_patch.py`; select at
runtime with `KDP_DECODE_KERNEL=torch|decode_paged|speculative`, which keeps the
failing vendor path one environment variable away for reproduction.

Known limitation: the attention span is derived from the batch's real length, so
this is shape-dynamic and cannot be captured in a FULL CUDA graph. Run with
`--enforce-eager`, or make the span static before enabling capture.
"""

from __future__ import annotations

import torch

# Bounds the gathered K/V so a long context cannot allocate the world.
GATHER_BUDGET_ELEMENTS = 1 << 24


class UnsupportedDecode(NotImplementedError):
    """Raised when this fallback would silently compute the wrong attention.

    Qwen3-8B needs neither a sliding window nor attention sinks, so an early
    version simply ignored both arguments. That is safe for Qwen3 and wrong for
    any model that uses them — a sliding-window model would attend to the whole
    context and a sink model would lose its sink. Refusing is the only honest
    behaviour: the caller then keeps the vendor kernel, which does implement them.
    """


def torch_paged_decode(**kwargs: object) -> int:
    """Drop-in replacement for the qlen==1 branch of `speculative_attention`."""
    window = kwargs.get("max_window_size", -1)
    if window is not None and int(window) >= 0:  # type: ignore[arg-type]
        raise UnsupportedDecode(
            f"sliding window {window} is not implemented here; keep the vendor kernel"
        )
    if kwargs.get("sink") is not None:
        raise UnsupportedDecode("attention sinks are not implemented here; keep the vendor kernel")
    if int(kwargs.get("qlen", 1)) != 1:  # type: ignore[arg-type]
        raise UnsupportedDecode("only regular decode (qlen == 1) is implemented here")

    out: torch.Tensor = kwargs["out"]  # type: ignore[assignment]
    query_in: torch.Tensor = kwargs["q"]  # type: ignore[assignment]
    k_cache: torch.Tensor = kwargs["k_cache"]  # type: ignore[assignment]
    v_cache: torch.Tensor = kwargs["v_cache"]  # type: ignore[assignment]
    lens: torch.Tensor = kwargs["context_lens_xpu"].to(torch.int64)  # type: ignore[union-attr]
    cpu_lens: torch.Tensor = kwargs["context_lens_cpu"]  # type: ignore[assignment]
    tables: torch.Tensor = kwargs["block_tables"].to(torch.int64)  # type: ignore[union-attr]
    scale = float(kwargs["scale"])  # type: ignore[arg-type]
    heads, dim = int(kwargs["head_num"]), int(kwargs["head_dim"])  # type: ignore[arg-type]
    kv_heads = int(kwargs["kv_head_num"])  # type: ignore[arg-type]
    block_size = int(kwargs["block_size"])  # type: ignore[arg-type]
    blocks_per_seq = int(kwargs["max_num_blocks_per_seq"])  # type: ignore[arg-type]

    tokens = out.shape[0]
    if tokens == 0:
        return 0
    group = heads // kv_heads

    # The length is read from the CPU copy on purpose: a device-side .item() is a
    # sync that is illegal under graph capture, and during FULL capture the device
    # tensor holds dummy values that once produced an exabyte-sized allocation.
    ceiling = min(blocks_per_seq, k_cache.shape[0]) * block_size
    max_len = max(1, min(int(cpu_lens.max()), ceiling))
    lens = lens.clamp(max=max_len)
    used_blocks = max(1, (max_len + block_size - 1) // block_size)
    span = used_blocks * block_size

    query = query_in.reshape(tokens, heads, dim)
    positions = torch.arange(span, device=out.device)
    chunk = max(1, min(tokens, GATHER_BUDGET_ELEMENTS // max(1, kv_heads * span * dim)))

    for start in range(0, tokens, chunk):
        stop = min(start + chunk, tokens)
        rows = stop - start
        index = tables[start:stop, :used_blocks].clamp_min(0)
        keys = k_cache[index].permute(0, 2, 1, 3, 4).reshape(rows, kv_heads, span, dim)
        values = v_cache[index].permute(0, 2, 1, 3, 4).reshape(rows, kv_heads, span, dim)
        grouped = query[start:stop].reshape(rows, kv_heads, group, dim).to(torch.float32)
        scores = torch.einsum("rkgd,rkld->rkgl", grouped, keys.to(torch.float32)) * scale
        invalid = positions[None, :] >= lens[start:stop, None]
        scores = scores.masked_fill(invalid[:, None, None, :], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        # A zero-length row is all -inf and softmaxes to NaN; warmup uses such rows.
        weights = torch.nan_to_num(weights)
        context = torch.einsum("rkgl,rkld->rkgd", weights, values.to(torch.float32))
        out[start:stop] = context.reshape(rows, heads, dim).to(out.dtype)
    return 0
