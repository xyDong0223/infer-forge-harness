"""Golden reference for MiniMax-M3's first decoder layer, computed on the CPU straight
from the checkpoint.

Every in-process check so far graded one operator against another implementation of the
same operator. All of them passed -- qk-norm/RoPE bit-exact, the int8 dense linear at
0.9% relative L2, the block-sparse attend at 1-2% against cache-free full attention, MoE
routing and activation fixed and reference-checked -- and the served output is still a
prompt-independent LaTeX-ish token. What no per-operator check can catch is a weight
landing in the wrong place: each operator is then individually correct and the composition
is nonsense.

So this recomputes layer 0 end to end in fp32 from the safetensors, and prints exactly the
quantities `patches/m3_layer_trace.py` prints from the running service, to be compared:

    layer0 in[1]                  embeddings                       (replicated)
    dense_attn in[hidden_states]  after input_layernorm            (replicated)
    dense_attn out                o_proj partial on rank 0's shard (sharded)
    mlp in[0]                     after post_attention_layernorm   (replicated)
    mlp out                       down_proj partial on rank 0      (sharded)
    layer0 out[1]                 residual = embeddings + all-reduced attention

Sharding mirrors TP=8: q heads 0-7 of 64, kv head 0 of 4 (4 kv heads across 8 ranks means
each is replicated twice, so rank r reads kv head r//2), o_proj row-parallel over the
first 1024 input channels, gate/up column-parallel over the first 1536 channels, down_proj
row-parallel to match.

Run inside the pod: python3 tools/probe/m3_layer0_golden.py --tokens 758,5505,300,5969,355
"""

import argparse
import json
import os

import torch
from safetensors import safe_open

PREFIX = "language_model.model.layers.0."


def load(root, index, key):
    with safe_open(os.path.join(root, index[key]), framework="pt") as handle:
        return handle.get_tensor(key)


def dequantize(weight, scale):
    """int8 weight times its per-output-channel scale, in fp32, as [out, in]."""
    weight = weight.float()
    scale = scale.float().reshape(-1)
    if weight.shape[0] == scale.numel():
        return weight * scale.reshape(-1, 1)
    if weight.shape[1] == scale.numel():
        return (weight * scale.reshape(1, -1)).transpose(0, 1)
    raise ValueError(f"scale {tuple(scale.shape)} fits neither side of {tuple(weight.shape)}")


def gemma_norm(x, weight, eps):
    rstd = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * rstd * (1.0 + weight.float())


def partial_neox_rope(x, positions, rotary_dim, theta):
    """x is [tokens, heads, head_dim]; rotate the leading rotary_dim channels."""
    half = rotary_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim))
    angles = positions.double().reshape(-1, 1) * inv_freq.reshape(1, -1)
    cos = angles.cos().float()[:, None, :]
    sin = angles.sin().float()[:, None, :]
    first, second = x[..., :half], x[..., half:rotary_dim]
    rotated = torch.cat([first * cos - second * sin, second * cos + first * sin], dim=-1)
    return torch.cat([rotated, x[..., rotary_dim:]], dim=-1)


def causal_attention(q, k, v, scale, causal=True):
    """q [T, H, D], k/v [T, Hkv, D] -> [T, H*D], fp32, GQA by repetition."""
    tokens, heads, dim = q.shape
    kv_heads = k.shape[1]
    group = heads // kv_heads
    mask = (
        torch.triu(torch.full((tokens, tokens), float("-inf")), diagonal=1)
        if causal
        else torch.zeros(tokens, tokens)
    )
    out = torch.empty_like(q)
    for head in range(kv_heads):
        queries = q[:, head * group : (head + 1) * group]
        logits = torch.einsum("tgd,sd->gts", queries, k[:, head]) * scale
        weights = torch.softmax(logits + mask, dim=-1)
        out[:, head * group : (head + 1) * group] = torch.einsum(
            "gts,sd->tgd", weights, v[:, head]
        )
    return out.reshape(tokens, heads * dim)


def _attention_variants(
    args, normed, w_q, w_k, w_v, w_o, q_weight, k_weight, heads, kv_heads, head_dim,
    eps, rotary_dim, theta, lo, hi,
):
    """Which convention does the served attention actually implement?

    The element-wise comparison put layer 0's attention output 25% off the golden while
    the norms agreed to 2% -- so a convention differs rather than a weight being wrong.
    Rather than argue about which, compute the candidates and let the dump pick.
    """
    positions = torch.arange(normed.shape[0])
    q_raw = (normed @ w_q.transpose(0, 1)).reshape(-1, heads, head_dim)
    k_raw = (normed @ w_k.transpose(0, 1)).reshape(-1, kv_heads, head_dim)
    v = (normed @ w_v.transpose(0, 1)).reshape(-1, kv_heads, head_dim)

    def normed_then_roped(x, weight):
        return partial_neox_rope(gemma_norm(x, weight, eps), positions, rotary_dim, theta)

    def roped_then_normed(x, weight):
        return gemma_norm(partial_neox_rope(x, positions, rotary_dim, theta), weight, eps)

    variants = {
        "norm-then-rope, causal": (
            normed_then_roped(q_raw, q_weight), normed_then_roped(k_raw, k_weight), True,
            head_dim**-0.5,
        ),
        "rope-then-norm, causal": (
            roped_then_normed(q_raw, q_weight), roped_then_normed(k_raw, k_weight), True,
            head_dim**-0.5,
        ),
        "no qk-norm, causal": (
            partial_neox_rope(q_raw, positions, rotary_dim, theta),
            partial_neox_rope(k_raw, positions, rotary_dim, theta), True, head_dim**-0.5,
        ),
        "norm, no rope, causal": (
            gemma_norm(q_raw, q_weight, eps), gemma_norm(k_raw, k_weight, eps), True,
            head_dim**-0.5,
        ),
        "norm-then-rope, non-causal": (
            normed_then_roped(q_raw, q_weight), normed_then_roped(k_raw, k_weight), False,
            head_dim**-0.5,
        ),
        "norm-then-rope, scale=1": (
            normed_then_roped(q_raw, q_weight), normed_then_roped(k_raw, k_weight), True, 1.0,
        ),
    }
    path = os.path.join(args.compare, "dense_attn.out.pt")
    if not os.path.exists(path):
        return
    served = torch.load(path).float()
    print("    attention convention sweep (relL2 of the rank shard against the dump):")
    for label, (q, k, causal, scale) in variants.items():
        attn = causal_attention(q, k, v, scale, causal=causal)
        shard = attn[:, lo:hi] @ w_o[:, lo:hi].transpose(0, 1)
        if shard.shape != served.shape:
            continue
        delta = (served - shard).norm() / shard.norm().clamp_min(1e-30)
        print(f"      {label:28s} relL2={float(delta):.6g}")

    # Every q/k convention landed at the same distance, so the difference is orthogonal
    # to q/k: it has to be which kv head this rank reads, or which slice of the heads /
    # o_proj it owns. Sweep both rather than assume.
    q = variants["norm-then-rope, causal"][0]
    k = variants["norm-then-rope, causal"][1]
    heads_per_rank = (hi - lo) // head_dim
    best = None
    print("    shard/kv-head sweep:")
    for kv_head in range(kv_heads):
        one_k = k[:, kv_head : kv_head + 1]
        one_v = v[:, kv_head : kv_head + 1]
        for rank in range(heads // heads_per_rank):
            begin = rank * heads_per_rank * head_dim
            end = begin + heads_per_rank * head_dim
            attn = causal_attention(
                q[:, rank * heads_per_rank : (rank + 1) * heads_per_rank],
                one_k,
                one_v,
                head_dim**-0.5,
            )
            shard = attn @ w_o[:, begin:end].transpose(0, 1)
            if shard.shape != served.shape:
                continue
            delta = float((served - shard).norm() / shard.norm().clamp_min(1e-30))
            if best is None or delta < best[0]:
                best = (delta, kv_head, rank)
    if best is not None:
        print(
            f"      best relL2={best[0]:.6g} at kv_head={best[1]} q_head_block={best[2]} "
            f"(assumed kv_head=0 q_head_block=0)"
        )

    # Nothing about q/k, causality, scale or sharding moved the number, and the served
    # output is slightly *larger* than the golden. That is what attending over slots the
    # sequence does not own looks like: unwritten cache slots hold zeros, a zero key gives
    # logit 0, and softmax hands those slots a large share. The Kunlun paged kernel is
    # already on record ignoring a window bound on another model, so test it directly by
    # padding the context out to the paged block and to a few intermediate lengths.
    tokens = normed.shape[0]
    print("    over-attention sweep (context padded with zero k/v beyond the prompt):")
    for padded in (8, 16, 32, 64, 128):
        if padded <= tokens:
            continue
        pad = padded - tokens
        k_padded = torch.cat([k, torch.zeros(pad, k.shape[1], head_dim)], dim=0)
        v_padded = torch.cat([v, torch.zeros(pad, v.shape[1], head_dim)], dim=0)
        # Causal only among the real tokens; the padding is deliberately left *visible*,
        # which is the whole point -- a kernel that ignores the sequence bound would see it.
        mask = torch.zeros(tokens, padded)
        real = torch.triu(torch.full((tokens, tokens), float("-inf")), diagonal=1)
        mask[:, :tokens] = real
        group = heads // kv_heads
        out = torch.empty(tokens, q.shape[1], head_dim)
        for head in range(k.shape[1]):
            queries = q[:, head * group : (head + 1) * group]
            logits = torch.einsum("tgd,sd->gts", queries, k_padded[:, head]) * head_dim**-0.5
            weights = torch.softmax(logits + mask, dim=-1)
            out[:, head * group : (head + 1) * group] = torch.einsum(
                "gts,sd->tgd", weights, v_padded[:, head]
            )
        attn = out.reshape(tokens, -1)
        shard = attn[:, lo:hi] @ w_o[:, lo:hi].transpose(0, 1)
        delta = float((served - shard).norm() / shard.norm().clamp_min(1e-30))
        print(f"      context={padded:4d} relL2={delta:.6g}")

    # The served output barely moved across every q/k convention, which is what a
    # *degenerate* attention looks like -- one whose weights hardly depend on q and k.
    # Test the degenerate shapes directly: no mixing at all (out = V of the query's own
    # position), uniform mixing over the visible prefix, and everything collapsed onto the
    # first token.
    print("    degenerate-attention sweep:")
    group = heads // kv_heads
    v_expanded = v.repeat_interleave(group, dim=1)
    prefix_mean = torch.stack(
        [v_expanded[: i + 1].mean(dim=0) for i in range(tokens)], dim=0
    )
    shapes = {
        "out = V[own position]": v_expanded,
        "out = mean(V[:i+1])": prefix_mean,
        "out = V[0]": v_expanded[:1].expand(tokens, -1, -1),
    }
    for label, candidate in shapes.items():
        shard = candidate.reshape(tokens, -1)[:, lo:hi] @ w_o[:, lo:hi].transpose(0, 1)
        delta = float((served - shard).norm() / shard.norm().clamp_min(1e-30))
        print(f"      {label:24s} relL2={delta:.6g}")

    # A window bound is the one structural variant left, and the Kunlun kernels are on
    # record ignoring one on another model, so try the small windows a 5-token prompt can
    # actually distinguish.
    # Every q/k variant lands on the same ~0.24 floor, so the difference travels with V.
    # The stand-in for the fused op slices [q | k | v | index_q | index_k] out of one
    # tensor; if it normalised or roped into V's slice, the error would be exactly this
    # shape -- invisible to the rope self-check, which read the same slices back.
    # The paged cache dtype is the one thing between the golden k/v and the kernel k/v.
    # fp8 e4m3 keeps ~2 decimal digits, which would show up as exactly this: a few percent
    # per element, insensitive to every structural choice, and magnitude preserving.
    print("    cache-dtype sweep:")
    for label, dtype in (("bf16", torch.bfloat16), ("fp8_e4m3", torch.float8_e4m3fn),
                         ("fp8_e5m2", torch.float8_e5m2)):
        try:
            k_cast = k.to(dtype).float()
            v_cast = v.to(dtype).float()
        except Exception as error:
            print(f"      {label:10s} unavailable: {error!r}")
            continue
        attn = causal_attention(q, k_cast, v_cast, head_dim**-0.5)
        shard = attn[:, lo:hi] @ w_o[:, lo:hi].transpose(0, 1)
        delta = float((served - shard).norm() / shard.norm().clamp_min(1e-30))
        print(f"      {label:10s} relL2={delta:.6g}")

    print("    V-corruption sweep:")
    v_variants = {
        "V as loaded (baseline)": v,
        "V gemma-normed with k_norm": gemma_norm(v, k_weight, eps),
        "V roped": partial_neox_rope(v, positions, rotary_dim, theta),
        "V normed then roped": partial_neox_rope(
            gemma_norm(v, k_weight, eps), positions, rotary_dim, theta
        ),
    }
    for label, candidate in v_variants.items():
        attn = causal_attention(q, k, candidate, head_dim**-0.5)
        shard = attn[:, lo:hi] @ w_o[:, lo:hi].transpose(0, 1)
        delta = float((served - shard).norm() / shard.norm().clamp_min(1e-30))
        print(f"      {label:28s} relL2={delta:.6g}")

    print("    sliding-window sweep:")
    for window in range(1, tokens + 1):
        mask = torch.full((tokens, tokens), float("-inf"))
        for i in range(tokens):
            mask[i, max(0, i - window + 1) : i + 1] = 0.0
        out = torch.empty(tokens, q.shape[1], head_dim)
        for head in range(k.shape[1]):
            queries = q[:, head * group : (head + 1) * group]
            logits = torch.einsum("tgd,sd->gts", queries, k[:, head]) * head_dim**-0.5
            weights = torch.softmax(logits + mask, dim=-1)
            out[:, head * group : (head + 1) * group] = torch.einsum(
                "gts,sd->tgd", weights, v[:, head]
            )
        shard = out.reshape(tokens, -1)[:, lo:hi] @ w_o[:, lo:hi].transpose(0, 1)
        delta = float((served - shard).norm() / shard.norm().clamp_min(1e-30))
        print(f"      window={window} relL2={delta:.6g}")


def quantize_activation(x):
    """Per-token symmetric int8 round trip, the scheme the served linears apply.

    The checkpoint's input_activations are dynamic per-token, so every linear sees its
    input crushed to 8 bits with the token's own amax setting the step. That matters most
    where a tensor carries a few huge lanes -- the attention output and the clamped SwiGLU
    output both do -- because the small lanes then quantize toward zero.
    """
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.round(x / amax * 127.0).clamp(-127, 127) * (amax / 127.0)


def compare(dump_dir, name, golden):
    """Element-wise relative L2 against a tensor dumped by patches/m3_layer_trace.py."""
    if not dump_dir:
        return
    path = os.path.join(dump_dir, name + ".pt")
    if not os.path.exists(path):
        print(f"    {name}: not dumped")
        return
    served = torch.load(path).float()
    if served.shape != golden.shape:
        print(f"    {name}: shape {tuple(served.shape)} vs golden {tuple(golden.shape)}")
        return
    delta = (served - golden).norm() / golden.norm().clamp_min(1e-30)
    cos = torch.nn.functional.cosine_similarity(
        served.reshape(1, -1), golden.reshape(1, -1)
    ).item()
    print(f"    {name}: relL2={float(delta):.6g} cos={cos:.8f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/mnt/cluster/MiniMax-M3-W8A8-INT8-Dynamic")
    parser.add_argument("--tokens", default="758,5505,300,5969,355")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument(
        "--layers",
        type=int,
        default=1,
        help="how many leading dense layers to walk (M3's first three are dense)",
    )
    parser.add_argument(
        "--compare",
        default="",
        help="directory of tensors dumped by patches/m3_layer_trace.py (M3_TRACE_DUMP)",
    )
    parser.add_argument(
        "--emulate-w8a8",
        action="store_true",
        help="quantize every linear input per-token to int8, as the served kernels do",
    )
    args = parser.parse_args()
    act = quantize_activation if args.emulate_w8a8 else (lambda t: t)

    root = args.model
    index = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
    config = json.load(open(os.path.join(root, "config.json")))
    text = config.get("text_config", config)

    hidden = text["hidden_size"]
    heads = text["num_attention_heads"]
    kv_heads = text["num_key_value_heads"]
    head_dim = text["head_dim"]
    eps = text["rms_norm_eps"]
    theta = float(text["rope_theta"])
    rotary_dim = int(head_dim * text["partial_rotary_factor"])
    intermediate = text["dense_intermediate_size"]
    limit = float(text["swiglu_limit"])
    alpha = float(text.get("swiglu_alpha", 1.702))
    beta = float(text.get("swiglu_beta", 1.0))

    token_ids = [int(t) for t in args.tokens.split(",")]
    embed_key = "language_model.model.embed_tokens.weight"
    with safe_open(os.path.join(root, index[embed_key]), framework="pt") as handle:
        rows = handle.get_slice(embed_key)
        x = torch.stack(
            [torch.tensor(rows[t : t + 1][0].tolist(), dtype=torch.float32) for t in token_ids]
        )
    print(f"layer0 in[1]                 norm={float(x.norm()):.6g}")
    compare(args.compare, "layer0.in_1", x)

    residual = x
    carried = None  # previous layer's full FFN output, added by the next input_layernorm
    for layer in range(args.layers):
        prefix = f"language_model.model.layers.{layer}."

        def load_layer(name):
            return load(root, index, prefix + name)

        def linear(name):
            return dequantize(load_layer(name + ".weight"), load_layer(name + ".weight_scale"))

        if carried is not None:
            residual = residual + carried
        normed = gemma_norm(residual, load_layer("input_layernorm.weight"), eps)

        w_q, w_k, w_v = (linear("self_attn." + n) for n in ("q_proj", "k_proj", "v_proj"))
        w_o = linear("self_attn.o_proj")

        projected = act(normed)
        q = (projected @ w_q.transpose(0, 1)).reshape(-1, heads, head_dim)
        k = (projected @ w_k.transpose(0, 1)).reshape(-1, kv_heads, head_dim)
        v = (projected @ w_v.transpose(0, 1)).reshape(-1, kv_heads, head_dim)
        q = gemma_norm(q, load_layer("self_attn.q_norm.weight"), eps)
        k = gemma_norm(k, load_layer("self_attn.k_norm.weight"), eps)
        positions = torch.arange(len(token_ids))
        q = partial_neox_rope(q, positions, rotary_dim, theta)
        k = partial_neox_rope(k, positions, rotary_dim, theta)

        attn = causal_attention(q, k, v, head_dim**-0.5)
        full_attn = act(attn) @ w_o.transpose(0, 1)

        heads_per_rank = heads // args.tp
        lo = args.rank * heads_per_rank * head_dim
        hi = lo + heads_per_rank * head_dim
        shard_attn = act(attn[:, lo:hi]) @ w_o[:, lo:hi].transpose(0, 1)
        print(
            f"layer{layer} dense_attn in[hidden]  norm={float(normed.norm()):.6g} "
            f"max={float(normed.max()):.6g}"
        )
        print(f"layer{layer} dense_attn out(shard) norm={float(shard_attn.norm()):.6g}")
        if layer == 0:
            compare(args.compare, "dense_attn.in_hidden_states", normed)
            compare(args.compare, "dense_attn.out", shard_attn)
            if args.compare:
                _attention_variants(
                    args,
                    normed,
                    w_q,
                    w_k,
                    w_v,
                    w_o,
                    load_layer("self_attn.q_norm.weight"),
                    load_layer("self_attn.k_norm.weight"),
                    heads,
                    kv_heads,
                    head_dim,
                    eps,
                    rotary_dim,
                    theta,
                    lo,
                    hi,
                )

        residual = residual + full_attn
        normed2 = gemma_norm(residual, load_layer("post_attention_layernorm.weight"), eps)
        print(
            f"layer{layer} mlp in[0]              norm={float(normed2.norm()):.6g} "
            f"max={float(normed2.max()):.6g}"
        )

        w_gate, w_up, w_down = (linear("mlp." + n) for n in ("gate_proj", "up_proj", "down_proj"))
        per_rank = intermediate // args.tp
        rows = slice(args.rank * per_rank, (args.rank + 1) * per_rank)
        projected2 = act(normed2)
        gate_full = projected2 @ w_gate.transpose(0, 1)
        up_full = projected2 @ w_up.transpose(0, 1)
        gate_full = gate_full.clamp(max=limit)
        up_full = up_full.clamp(min=-limit, max=limit)
        activated = gate_full * torch.sigmoid(alpha * gate_full) * (up_full + beta)
        carried = act(activated) @ w_down.transpose(0, 1)
        shard_mlp = act(activated[:, rows]) @ w_down[:, rows].transpose(0, 1)
        print(
            f"layer{layer} mlp out(shard)         norm={float(shard_mlp.norm()):.6g} "
            f"min={float(shard_mlp.min()):.6g}"
        )
        print(f"layer{layer} out[1] residual       norm={float(residual.norm()):.6g}")
        if layer == 0:
            compare(args.compare, "mlp.in_0", normed2)
            compare(args.compare, "mlp.out", shard_mlp)
            compare(args.compare, "layer0.out_0", shard_mlp)
            compare(args.compare, "layer0.out_1", residual)


if __name__ == "__main__":
    main()
