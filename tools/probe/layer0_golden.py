"""Golden reference for a decoder layer of the DeepSeekV2-style MLA family
(MoE + DSA, W8A8 INT8 dynamic quantized), computed on the CPU straight from
the checkpoint. Any model of this family works — shapes and conventions are
read from config.json and the safetensors; GLM5.2-Int-W8A8 is simply the
first model this ran against (see openwiki/harness/experiences/attention-mla.md).

Every per-operator check grades one operator against another implementation of
the same operator, so none of them can catch a weight landing in the wrong
place. This recomputes layer 0 end to end in fp32 from the safetensors
-- never via a transformers forward -- and dumps every intermediate stage as .pt files plus
one JSON status line, so a served-side layer trace can be diffed against it stage by stage.

Architecture (verified against config.json + the DeepseekV2-style plugin snapshot in
/tmp/glm_port, which fixes the tensor layouts):

  hidden 6144, 78 layers, layer 0 is DENSE (first_k_dense_replace), 64 heads,
  q_lora_rank 2048, kv_lora_rank 512, qk_nope 192, qk_rope 64, v_head 256,
  rope_theta 8e6, is_neox_style=False -> interleaved even/odd pair rotation.

  x = embed[token]                                            (fp32; embed may be
                                                               unquantized fp or int8+scale)
  h = rms_norm(x, input_layernorm.weight, eps)
  MLA, full (non-absorbed) formulation:
    q_c   = h @ q_a_proj^T            [T, 2048]
    q_c   = rms_norm(q_c, q_a_layernorm.weight, eps)
    q     = q_c @ q_b_proj^T          [T, 64*(192+64)]; per head: [nope 192 | rope 64]
    kv    = h @ kv_a_proj_with_mqa^T  [T, 512+64]; split [latent 512 | rope 64]
    c_kv  = rms_norm(latent, kv_a_layernorm.weight, eps)   (norm over the 512 latent dims
                                                            only, per DeepseekV2)
    kv_up = c_kv @ kv_b_proj^T        [T, 64*(192+256)]; per head: [nope 192 | v 256]
    rope on q[...,192:256] and the shared k_pe, interleaved pairs, positions = arange(T)
    per head: score = (q_nope . k_nope + q_pe . k_pe) / sqrt(192), causal softmax, out = w . v
    attn  = concat over heads -> [T, 64*256]
    o     = attn @ o_proj^T           [T, 6144]
  h2 = rms_norm(x + o, post_attention_layernorm.weight, eps)
  mlp = down_proj(silu(gate_proj(h2)) * up_proj(h2))       (plain SwiGLU, no alpha/beta)
  out = x + o + mlp

If layer 0 has no kv_b_proj (kv sharing across layers), the lowest-numbered layer that does
own one is used and recorded as "kv_shared_from_layer". If no layer has one, the attention
context stage degrades with an explicit JSON reason; every other stage is still computed.

Discriminating control: kv_b_proj (or q_b_proj as fallback) is perturbed by 1e-2 relative
and the layer recomputed, so a comparison harness that reports a flat metric on a truly
perturbed input is exposed as broken.

Run inside the pod:
    python3 tools/probe/layer0_golden.py /mnt/cluster/GLM-5.2-W8A8-INT8-Dynamic
    python3 tools/probe/layer0_golden.py <model> --prompt-file p.txt \
        --compare /dump/dir [--emulate-w8a8]
"""

import argparse
import json
import os
import re

import torch
from safetensors import safe_open

FIXED_PROMPT = "The quick brown fox jumps over"

LAYER_CANDIDATE_PREFIXES = (
    "model.layers.0.",
    "language_model.model.layers.0.",
)
EMBED_CANDIDATE_KEYS = (
    "model.embed_tokens.weight",
    "language_model.model.embed_tokens.weight",
)


def load(root, index, key):
    """One tensor out of whichever shard the index points it at."""
    with safe_open(os.path.join(root, index[key]), framework="pt") as handle:
        return handle.get_tensor(key)


def dequantize(weight, scale):
    """int8 weight times its per-output-channel scale, in fp32, as [out, in].

    Mirrors m3_layer0_golden.py: the scale may be [out_features] or [out_features, 1],
    and the weight may already be stored transposed.
    """
    weight = weight.float()
    scale = scale.float().reshape(-1)
    if weight.shape[0] == scale.numel():
        return weight * scale.reshape(-1, 1)
    if weight.shape[1] == scale.numel():
        return (weight * scale.reshape(1, -1)).transpose(0, 1)
    raise ValueError(f"scale {tuple(scale.shape)} fits neither side of {tuple(weight.shape)}")


def load_linear(root, index, prefix, name):
    """Dequantized [out, in] fp32 weight for `prefix + name`, or (None, reason)."""
    wkey = prefix + name + ".weight"
    if wkey not in index:
        return None, f"missing {wkey}"
    weight = load(root, index, wkey)
    # The scale sits NEXT to the weight key (X.weight + X.weight_scale),
    # not appended to it (X.weight.weight_scale).
    skey = prefix + name + ".weight_scale"
    if skey in index:
        try:
            return dequantize(weight, load(root, index, skey)), None
        except ValueError as error:
            return None, f"dequantize failed for {wkey}: {error}"
    if weight.dtype in (torch.float32, torch.float16, torch.bfloat16):
        return weight.float(), None
    return None, f"unsupported dtype {weight.dtype} for {wkey}"


def rms_norm(x, weight, eps):
    """Classic RMSNorm (no Gemma +1): x / rms * weight."""
    rstd = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * rstd * weight.float()


def partial_interleaved_rope(x, positions, rotary_dim, theta):
    """x is [T, H, D]; rotate the leading rotary_dim channels as even/odd pairs.

    is_neox_style=False for the DeepSeek family: channel 2i pairs with channel 2i+1 and
    they rotate together (GPT-J interleaved layout), unlike the halved neox layout
    m3_layer0_golden.py implements. Non-rotary trailing channels pass through.
    """
    half = rotary_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim))
    angles = positions.double().reshape(-1, 1) * inv_freq.reshape(1, -1)
    cos = angles.cos().float()[:, None, :]
    sin = angles.sin().float()[:, None, :]
    pairs = x[..., :rotary_dim].reshape(*x.shape[:-1], half, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    rotated = torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1)
    rotated = rotated.reshape(*x.shape[:-1], rotary_dim)
    return torch.cat([rotated, x[..., rotary_dim:]], dim=-1)


def causal_mla_attention(q, k, v, scale):
    """Full (non-absorbed) MLA attention.

    q/k are [T, H, qk_nope+qk_rope] (nope leading, rope trailing), v is [T, H, v_head].
    MQA-style: one shared latent expanded per head by kv_b_proj already, so heads are
    independent standard attention. Returns [T, H * v_head].
    """
    tokens = q.shape[0]
    mask = torch.triu(torch.full((tokens, tokens), float("-inf")), diagonal=1)
    logits = torch.einsum("thd,shd->hts", q, k) * scale
    weights = torch.softmax(logits + mask, dim=-1)
    out = torch.einsum("hts,shd->thd", weights, v)
    return out.reshape(tokens, out.shape[1] * out.shape[2])


def quantize_activation(x):
    """Per-token symmetric int8 round trip, the scheme the served W8A8 linears apply."""
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.round(x / amax * 127.0).clamp(-127, 127) * (amax / 127.0)


def compare(dump_dir, name, golden, results):
    """Element-wise relative L2 + cosine against a dumped .pt from the served trace."""
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
    delta = float((served - golden).norm() / golden.norm().clamp_min(1e-30))
    cos = torch.nn.functional.cosine_similarity(
        served.reshape(1, -1), golden.reshape(1, -1)
    ).item()
    print(f"    {name}: relL2={delta:.6g} cos={cos:.8f}")
    results[name] = {"relL2": delta, "cos": cos}


def find_prefix(index):
    for prefix in LAYER_CANDIDATE_PREFIXES:
        if prefix + "self_attn.q_a_proj.weight" in index:
            return prefix
    for key in index:  # last resort: any layer-0 attention key
        if re.match(r"(?:language_model\.)?model\.layers\.0\.self_attn\.", key):
            return key[: key.index("self_attn.") - len("model.layers.0.") - 1] + "model.layers.0."
    raise SystemExit("no layer-0 prefix found in model.safetensors.index.json")


def find_embed_key(index):
    for key in EMBED_CANDIDATE_KEYS:
        if key in index:
            return key
    for key in index:
        if key.endswith("embed_tokens.weight"):
            return key
    return None


def embed_rows(root, index, key, token_ids):
    """Token rows of the embedding table, fp32, dequantized if the table is int8+scale.

    Only the requested rows are read via the safetensors slice API, so a large vocab
    never gets materialized. Returns (tensor, dtype_note, vocab_size).
    """
    path = os.path.join(root, index[key])
    with safe_open(path, framework="pt") as handle:
        sl = handle.get_slice(key)
        vocab = sl.get_shape()[0]
        raw = torch.stack(
            [torch.tensor(sl[t : t + 1][0].tolist(), dtype=torch.float32) for t in token_ids]
        )
    if key + ".weight_scale" in index:
        scale = load(root, index, key + ".weight_scale").float().reshape(-1)
        rows = scale[token_ids].reshape(-1, 1)
        return raw * rows, "int8+per-row scale, dequantized", vocab
    return raw, "stored fp, read directly", vocab


def mla_attention(h, attn, cfg, act):
    """MLA block from post-layernorm hidden states to o_proj output.

    Returns (stages, degraded) where every key of `stages` is a tensor to dump.
    Degrades stage-by-stage instead of crashing on a missing weight.
    """
    degraded = {}
    stages = {}
    heads = cfg["heads"]
    q_lora, kv_lora = cfg["q_lora_rank"], cfg["kv_lora_rank"]
    nope, rope_dim, v_head = cfg["qk_nope"], cfg["qk_rope"], cfg["v_head"]
    positions = torch.arange(h.shape[0])

    def guard(stage, fn):
        try:
            stages[stage] = fn()
        except Exception as error:  # missing weight handled by load_linear; anything else too
            degraded[stage] = f"{type(error).__name__}: {error}"
            stages[stage] = None

    # ---- q path: q_a_proj -> q_a_layernorm -> q_b_proj --------------------------
    guard("mla.q_a.out", lambda: act(h) @ attn["q_a_proj"].transpose(0, 1))
    if stages["mla.q_a.out"] is None:
        degraded["mla"] = "q_a_proj unavailable; attention and everything downstream degrades"
        return stages, degraded
    q_a = stages["mla.q_a.out"]

    guard(
        "mla.q_a_normed.out",
        lambda: rms_norm(q_a, attn["q_a_layernorm"], cfg["eps"]),
    )
    if stages["mla.q_a_normed.out"] is None:
        return stages, degraded
    q_c = stages["mla.q_a_normed.out"]

    guard("mla.q.out", lambda: act(q_c) @ attn["q_b_proj"].transpose(0, 1))
    if stages["mla.q.out"] is None:
        return stages, degraded
    q = stages["mla.q.out"].reshape(-1, heads, nope + rope_dim)

    # ---- kv path: kv_a_proj_with_mqa -> kv_a_layernorm (latent only) ------------
    guard("mla.kv_a.out", lambda: act(h) @ attn["kv_a_proj_with_mqa"].transpose(0, 1))
    if stages["mla.kv_a.out"] is None:
        degraded["mla"] = "kv_a_proj_with_mqa unavailable; attention degrades"
        return stages, degraded
    kv_a = stages["mla.kv_a.out"]
    c_kv, k_pe = kv_a[..., :kv_lora], kv_a[..., kv_lora:]

    guard("mla.c_kv_normed.out", lambda: rms_norm(c_kv, attn["kv_a_layernorm"], cfg["eps"]))
    if stages["mla.c_kv_normed.out"] is None:
        return stages, degraded
    c_kv = stages["mla.c_kv_normed.out"]

    # ---- rope on the trailing qk_rope channels and the shared k_pe --------------
    q_roped = partial_interleaved_rope(q, positions, rope_dim, cfg["theta"])
    guard(
        "mla.q_pe_roped",
        lambda: q_roped[..., nope:].clone(),
    )
    k_pe_roped = partial_interleaved_rope(
        k_pe.reshape(-1, 1, rope_dim), positions, rope_dim, cfg["theta"]
    )[:, 0]
    guard("mla.k_pe_roped", lambda: k_pe_roped.clone())

    # ---- kv_b_proj (possibly borrowed from the kv-sharing source layer) ---------
    if attn["kv_b_proj"] is None:
        degraded["mla.dense_attn.out"] = (
            "kv_b_proj unavailable in any layer; attention context/o_proj/layer out degrades"
        )
        stages["mla.dense_attn.out"] = None
        stages["o_proj.out"] = None
        return stages, degraded

    kv_up = (act(c_kv) @ attn["kv_b_proj"].transpose(0, 1)).reshape(-1, heads, nope + v_head)
    k_nope, v = kv_up[..., :nope], kv_up[..., nope:]

    k = torch.cat(
        [k_nope, k_pe_roped.reshape(-1, 1, rope_dim).expand(-1, heads, rope_dim)], dim=-1
    )
    attn_out = causal_mla_attention(q_roped, k, v, cfg["scale"])
    stages["mla.dense_attn.out"] = attn_out

    guard("o_proj.out", lambda: act(attn_out) @ attn["o_proj"].transpose(0, 1))
    return stages, degraded


def dense_mlp(h2, mlp, act):
    """Plain SwiGLU dense MLP (layer 0 is dense per first_k_dense_replace)."""
    degraded = {}
    stages = {}
    guard_pairs = (
        ("mlp.gate.out", "gate_proj"),
        ("mlp.up.out", "up_proj"),
    )
    for stage, name in guard_pairs:
        if mlp[name] is None:
            degraded[stage] = f"missing mlp.{name}"
            return stages, degraded
    gate = act(h2) @ mlp["gate_proj"].transpose(0, 1)
    up = act(h2) @ mlp["up_proj"].transpose(0, 1)
    stages["mlp.gate.out"] = gate
    stages["mlp.up.out"] = up
    activated = torch.nn.functional.silu(gate) * up
    stages["mlp.act.out"] = activated
    if mlp["down_proj"] is None:
        degraded["mlp.down.out"] = "missing mlp.down_proj"
        return stages, degraded
    stages["mlp.down.out"] = act(activated) @ mlp["down_proj"].transpose(0, 1)
    return stages, degraded


def run_layer(x, weights, cfg, act):
    """Full layer-0 pipeline: returns (stages, degraded). Mirrors the m3 layer walk."""
    stages, degraded = {}, {}
    stages["layer0.in_1"] = x
    if weights["input_layernorm"] is None:
        degraded["dense_attn.in_hidden_states"] = "missing input_layernorm.weight"
        stages["dense_attn.in_hidden_states"] = None
        return stages, degraded
    normed = rms_norm(x, weights["input_layernorm"], cfg["eps"])
    stages["dense_attn.in_hidden_states"] = normed

    attn_stages, attn_degraded = mla_attention(normed, weights["attn"], cfg, act)
    stages.update(attn_stages)
    degraded.update(attn_degraded)
    o_out = stages.get("o_proj.out")

    if o_out is None:
        degraded["mlp.in_0"] = "attention output unavailable; post-attn stages degrade"
        degraded["layer0.out_1"] = degraded["mlp.in_0"]
        stages["mlp.in_0"] = None
        stages["layer0.out_1"] = None
        return stages, degraded
    residual = x + o_out
    if weights["post_attention_layernorm"] is None:
        degraded["mlp.in_0"] = "missing post_attention_layernorm.weight"
        stages["mlp.in_0"] = None
        stages["layer0.out_1"] = residual
        return stages, degraded
    h2 = rms_norm(residual, weights["post_attention_layernorm"], cfg["eps"])
    stages["mlp.in_0"] = h2

    mlp_stages, mlp_degraded = dense_mlp(h2, weights["mlp"], act)
    stages.update(mlp_stages)
    degraded.update(mlp_degraded)
    down = stages.get("mlp.down.out")
    stages["layer0.out_1"] = residual if down is None else residual + down
    return stages, degraded


def layer_weights(root, index, prefix, cfg):
    """Every layer-0 weight, dequantized. kv_b_proj may be borrowed for shared kv."""
    degraded = {}

    def lin(name):
        weight, reason = load_linear(root, index, prefix, name)
        if weight is None:
            degraded[name] = reason
        return weight

    def norm(name):
        key = prefix + name
        if key not in index:
            degraded[name] = f"missing {key}"
            return None
        return load(root, index, key).float()

    attn = {
        "q_a_proj": lin("self_attn.q_a_proj"),
        "q_a_layernorm": norm("self_attn.q_a_layernorm.weight"),
        "q_b_proj": lin("self_attn.q_b_proj"),
        "kv_a_proj_with_mqa": lin("self_attn.kv_a_proj_with_mqa"),
        "kv_a_layernorm": norm("self_attn.kv_a_layernorm.weight"),
        "kv_b_proj": lin("self_attn.kv_b_proj"),
        "o_proj": lin("self_attn.o_proj"),
    }

    kv_shared_from = None
    if attn["kv_b_proj"] is None:
        # kv sharing across layers: layer 0 reuses another layer's kv_b_proj. Find the
        # lowest-numbered layer that owns one (the first MoE layer in practice).
        owners = sorted(
            int(match.group(1))
            for match in (
                re.match(r"model\.layers\.(\d+)\.self_attn\.kv_b_proj\.weight", key)
                for key in index
            )
            if match
        )
        if owners:
            kv_shared_from = owners[0]
            donor_prefix = prefix.replace(
                "model.layers.0.", f"model.layers.{kv_shared_from}."
            )
            attn["kv_b_proj"], _ = load_linear(
                root, index, donor_prefix, "self_attn.kv_b_proj"
            )
        if attn["kv_b_proj"] is None:
            degraded["self_attn.kv_b_proj"] = (
                "no kv_b_proj in any layer; attention context cannot be computed"
            )

    weights = {
        "attn": attn,
        "mlp": {
            "gate_proj": lin("mlp.gate_proj"),
            "up_proj": lin("mlp.up_proj"),
            "down_proj": lin("mlp.down_proj"),
        },
        "input_layernorm": norm("input_layernorm.weight"),
        "post_attention_layernorm": norm("post_attention_layernorm.weight"),
    }
    return weights, degraded, kv_shared_from


def conf(config, key, default):
    """Read a scalar from config.json tolerating nested text/language configs."""
    for source in (
        config,
        config.get("text_config") or {},
        config.get("language_model") or {},
        (config.get("language_model") or {}).get("text_config") or {},
    ):
        if key in source:
            return source[key]
    return default


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model", nargs="?", default="/mnt/cluster/GLM-5.2-W8A8-INT8-Dynamic")
    parser.add_argument("--prompt-file", default="", help="override the builtin 6-8 token prompt")
    parser.add_argument(
        "--tokens", default="", help="comma-separated token ids, bypassing the tokenizer"
    )
    parser.add_argument(
        "--compare",
        default="",
        help="directory of tensors dumped by the served-side layer trace",
    )
    parser.add_argument(
        "--out-dir", default="layer0_golden_out", help="where the .pt stage dumps go"
    )
    parser.add_argument(
        "--emulate-w8a8",
        action="store_true",
        help="quantize every linear input per-token to int8, as the served kernels do",
    )
    parser.add_argument(
        "--control-scale", type=float, default=1e-2,
        help="relative perturbation for the discriminating control",
    )
    args = parser.parse_args()
    act = quantize_activation if args.emulate_w8a8 else (lambda t: t)

    root = args.model
    index = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
    config = json.load(open(os.path.join(root, "config.json")))

    cfg = {
        "hidden": int(conf(config, "hidden_size", 6144)),
        "heads": int(conf(config, "num_attention_heads", 64)),
        "q_lora_rank": int(conf(config, "q_lora_rank", 2048)),
        "kv_lora_rank": int(conf(config, "kv_lora_rank", 512)),
        "qk_nope": int(conf(config, "qk_nope_head_dim", 192)),
        "qk_rope": int(conf(config, "qk_rope_head_dim", 64)),
        "v_head": int(conf(config, "v_head_dim", 256)),
        "eps": float(conf(config, "rms_norm_eps", 1e-6)),
        "theta": float(conf(config, "rope_theta", 8000000.0)),
    }
    # Softmax scale: 1/sqrt(qk_nope_head_dim) per the layer contract.
    cfg["scale"] = cfg["qk_nope"] ** -0.5

    prefix = find_prefix(index)
    embed_key = find_embed_key(index)

    # ---- tokens ------------------------------------------------------------------
    if args.tokens:
        token_ids = [int(t) for t in args.tokens.split(",")]
        prompt = f"--tokens {args.tokens}"
    else:
        text = open(args.prompt_file).read() if args.prompt_file else FIXED_PROMPT
        prompt = text if args.prompt_file else FIXED_PROMPT
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(root, trust_remote_code=True)
        token_ids = tokenizer.encode(prompt)
        if len(token_ids) and token_ids[0] == getattr(tokenizer, "bos_token_id", None):
            token_ids = token_ids[1:]
        assert 1 <= len(token_ids) <= 32, f"prompt produced {len(token_ids)} tokens"

    if embed_key is None:
        raise SystemExit("embed_tokens.weight not found in the safetensors index")
    x, embed_note, vocab = embed_rows(root, index, embed_key, token_ids)

    weights, load_degraded, kv_shared_from = layer_weights(root, index, prefix, cfg)

    with torch.no_grad():
        stages, degraded = run_layer(x, weights, cfg, act)

        # ---- discriminating control -------------------------------------------------
        control = {"perturbed": None, "relative": args.control_scale}
        target = "kv_b_proj" if weights["attn"]["kv_b_proj"] is not None else "q_b_proj"
        if weights["attn"][target] is not None:
            perturbed_weights = {
                "attn": dict(weights["attn"]),
                "mlp": dict(weights["mlp"]),
                "input_layernorm": weights["input_layernorm"],
                "post_attention_layernorm": weights["post_attention_layernorm"],
            }
            perturbed_weights["attn"][target] = (
                weights["attn"][target] * (1.0 + args.control_scale)
            )
            control["perturbed"] = f"self_attn.{target}"
            control_stages, _ = run_layer(x, perturbed_weights, cfg, act)
            for stage, key in (
                ("attn_out_relL2", "mla.dense_attn.out"),
                ("layer_out_relL2", "layer0.out_1"),
            ):
                if control_stages.get(key) is not None and stages.get(key) is not None:
                    control[stage] = float(
                        (control_stages[key] - stages[key]).norm()
                        / stages[key].norm().clamp_min(1e-30)
                    )
        else:
            control["reason"] = "no perturbable attention weight available"

    # ---- dumps + norms --------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    norms = {}
    for stage, tensor in stages.items():
        if tensor is None:
            continue
        torch.save(tensor.cpu().float(), os.path.join(args.out_dir, stage + ".pt"))
        norms[stage] = float(tensor.float().norm())

    compare_results = {}
    if args.compare:
        print("stage comparison against the served dump:")
        for stage, tensor in stages.items():
            if tensor is not None:
                compare(args.compare, stage, tensor.float(), compare_results)

    all_degraded = {}
    for source in (load_degraded, degraded):
        for key, reason in source.items():
            all_degraded[f"layer0.{key}"] = reason

    essential = ["layer0.in_1", "dense_attn.in_hidden_states", "layer0.out_1"]
    ready = all(norms.get(stage) is not None for stage in essential)

    result = {
        "state": "GOLDEN_READY" if ready else "GOLDEN_DEGRADED",
        "model": root,
        "architecture": conf(config, "architectures", ["?"])[0],
        "model_type": conf(config, "model_type", "?"),
        "layer": 0,
        "prefix": prefix,
        "kv_shared_from_layer": kv_shared_from,
        "embed": {"key": embed_key, "note": embed_note, "vocab": vocab},
        "tokens": token_ids,
        "num_tokens": len(token_ids),
        "prompt": prompt if not args.prompt_file else f"(file:{args.prompt_file})",
        "config": cfg,
        "mla": {
            "formulation": "full non-absorbed (per-head nope/rope dot, MQA kv_b expansion)",
            "scale": cfg["scale"],
            "scale_basis": "1/sqrt(qk_nope_head_dim=192)",
            "rope": {
                "convention": "is_neox_style=False -> interleaved even/odd pairs",
                "theta": cfg["theta"],
                "rotary_dim": cfg["qk_rope"],
                "positions": "arange(seq_len)",
            },
            "mlp": "plain SwiGLU silu(gate)*up (dense layer 0, no alpha/beta)",
        },
        "out_dir": os.path.abspath(args.out_dir),
        "norms": norms,
        "degraded": all_degraded,
        "control": control,
        "emulate_w8a8": bool(args.emulate_w8a8),
    }
    if compare_results:
        result["compare"] = compare_results
    print(json.dumps(result))


if __name__ == "__main__":
    main()
