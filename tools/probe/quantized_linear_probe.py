"""In-pod W8A8 single-linear-layer probe. Prints JSON to stdout.

Exercises the quantized path at the smallest unit that still carries the real
convention risk: one per-channel int8 weight from a real checkpoint, through the
same two operators `KunlunScaledMMLinearKernel.apply_weights` uses, compared
against a float32 dequantized reference.

The specific thing under test is a units mismatch. Upstream compressed-tensors
stores a per-channel *scale*; the Kunlun `matmul` kernel wants a per-channel
*max*, so `quantization/kernels/scale_mm.py` does `w_s.mul_(127.0)` at load time.
Both spellings run, both produce fluent text, and only one is numerically right —
which is why this probe also runs the omission as a negative control. A pass that
cannot fail proves nothing, so the control must miss the threshold that the
candidate meets.

The control is also what settled the choice of metric. Dropping the 127 factor
scales every output channel by the same constant, so cosine similarity — the
threshold used elsewhere in this repo for kernel equivalence — stays at 1.0 and
reports the wrong answer as correct. The gate here is therefore the relative L2
norm, with cosine kept only as a direction check.

Activation scaling is measured, not assumed: `scaled_int8_quant` returns a scale
for static quantization and a per-token max for dynamic, and scale_mm reflects
that by passing `x_s * 127.0 if static else x_s`. The probe recovers whichever
convention actually holds by dequantizing both ways and keeping the one that
reproduces the input.
"""

from __future__ import annotations

import argparse
import json


def weight_map(model_path: str) -> dict:
    import os

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path, encoding="utf-8") as handle:
        return json.load(handle)["weight_map"]


def resolve_tensor(mapping: dict, requested: str) -> str:
    """Pick a quantized linear to exercise, by name or by discovery.

    `auto` exists because the name is model-specific in a way that fails silently:
    MiniMax-M2.5 stores `model.layers.0.self_attn.q_proj`, while the M3 checkpoint
    nests the language model and the same layer is
    `language_model.model.layers.0.self_attn.q_proj`. Discovery takes the first
    layer-0 scale in sorted order, so it stays deterministic, and skips experts —
    those are the MoE dimension's business.
    """
    if requested and requested != "auto":
        return requested
    candidates = sorted(key[: -len(".weight_scale")] for key in mapping
                        if key.endswith(".weight_scale") and ".layers.0." in key
                        and "expert" not in key)
    if not candidates:
        raise SystemExit("no quantized layer-0 linear in the checkpoint index")
    return candidates[0]


def precision_map(mapping: dict) -> dict:
    """Which top-level module prefixes are quantized and which are not.

    A W8A8 checkpoint is not uniformly int8: M3's quantization `ignore` list excludes
    the vision tower, the projector, the patch-merge MLP and the MoE gate, so those
    stay bfloat16. Recording the split makes the mixed-precision boundary a measured
    fact rather than an assumption — and that boundary is what a loader gets wrong.
    """
    prefixes: dict[str, dict] = {}
    for key in mapping:
        entry = prefixes.setdefault(key.split(".")[0], {"tensors": 0, "scales": 0})
        entry["tensors"] += 1
        if key.endswith(".weight_scale"):
            entry["scales"] += 1
    for entry in prefixes.values():
        entry["quantized"] = entry["scales"] > 0
    return prefixes


def load_pair(model_path: str, tensor: str, mapping: dict) -> tuple:
    """Read one quantized weight and its scale, without loading a whole shard."""
    import os

    from safetensors import safe_open

    weight_key, scale_key = f"{tensor}.weight", f"{tensor}.weight_scale"
    for key in (weight_key, scale_key):
        if key not in mapping:
            raise SystemExit(f"{key} is not in the checkpoint index")
    if mapping[weight_key] != mapping[scale_key]:
        raise SystemExit(f"{weight_key} and its scale live in different shards")
    shard = os.path.join(model_path, mapping[weight_key])
    with safe_open(shard, framework="pt") as handle:
        return handle.get_tensor(weight_key), handle.get_tensor(scale_key)


def activation_convention(x, x_q, x_s):
    """Return ("max"|"scale", dequantized activations) by checking which reproduces x."""
    import torch

    x32 = x.to(torch.float32)
    candidates = {
        "max": x_q.to(torch.float32) * (x_s.to(torch.float32) / 127.0),
        "scale": x_q.to(torch.float32) * x_s.to(torch.float32),
    }
    best = min(candidates, key=lambda name: float((candidates[name] - x32).abs().max()))
    return best, candidates[best]


def cosine(a, b) -> float:
    import torch

    a32, b32 = a.to(torch.float32).flatten(), b.to(torch.float32).flatten()
    return float(torch.dot(a32, b32) / (a32.norm() * b32.norm() + 1e-30))


def relative_l2(candidate, reference) -> float:
    import torch

    diff = (candidate.to(torch.float32) - reference.to(torch.float32)).norm()
    return float(diff / (reference.to(torch.float32).norm() + 1e-30))


def apply_kernel(x, w_q_layer, w_pc_max):
    """The two operators scale_mm.py uses, in the order it uses them.

    `w_q_layer` is in the layout the layer holds after loading — [in, out], as the
    upstream cutlass kernel leaves it — so the transpose below is the same one
    scale_mm.py writes.
    """
    import torch

    # scale=None selects the dynamic per-token branch, which is what a
    # W8A8-INT8-Dynamic checkpoint runs. Passing an empty tensor instead takes the
    # static branch and fails inside `static_scaled_int8_quant`.
    x_q, x_s, x_zp, static = torch.ops._C.scaled_int8_quant(
        x=x.contiguous(), scale=None, azp=None, symmetric=True
    )
    out = torch.ops._C.matmul(
        x=x_q,
        w=w_q_layer.transpose(0, 1),
        out_dtype=x.dtype,
        x_pc_max=x_s * 127.0 if static else x_s,
        w_pc_max=w_pc_max,
        bias=None,
    )
    return out, x_q, x_s, bool(static)


def main() -> int:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tensor", default="auto",
                        help="a quantized linear, or 'auto' to discover the first layer-0 one")
    parser.add_argument("--tokens", type=int, default=17)
    parser.add_argument("--device", default="cuda", help="the plugin claims to be CUDA")
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--cosine-floor", type=float, default=0.9999)
    parser.add_argument("--max-relative-l2", type=float, default=0.01,
                        help="magnitude-sensitive gate; cosine alone cannot see a scale error")
    args = parser.parse_args()

    import vllm  # noqa: F401  - registers the _C namespace and activates the plugin
    import vllm._custom_ops  # noqa: F401

    mapping = weight_map(args.model_path)
    tensor = resolve_tensor(mapping, args.tensor)
    precision = precision_map(mapping)
    w_q, w_s = load_pair(args.model_path, tensor, mapping)
    out_features, in_features = w_q.shape
    # The kernel wants a max; the checkpoint stores a scale. This is the line under
    # test, reproduced here rather than imported so the control can omit it.
    w_pc_max = w_s.to(torch.float32) * 127.0

    torch.manual_seed(args.seed)
    x = (torch.randn(args.tokens, in_features) * 0.02).to(torch.bfloat16)

    device = torch.device(args.device)
    x_d = x.to(device)
    # The checkpoint stores [out, in]; the cutlass path the Kunlun kernel inherits
    # holds [in, out] after loading. Emulating that here keeps the call identical to
    # scale_mm.py's, which is the point of the probe.
    w_q_layer = w_q.transpose(0, 1).contiguous().to(device)

    result: dict = {
        "dimension": "quantization",
        "tensor": tensor,
        "tensor_selection": "declared" if args.tensor and args.tensor != "auto"
                            else "discovered_from_the_index",
        # A W8A8 checkpoint is not uniformly int8, and the boundary is where a loader
        # goes wrong: M3 leaves the vision tower, projector, patch-merge MLP and MoE
        # gate in bfloat16 via the quantization `ignore` list.
        "precision_by_prefix": precision,
        "mixed_precision": len({entry["quantized"] for entry in precision.values()}) > 1,
        "shape": {"in_features": int(in_features), "out_features": int(out_features),
                  "tokens": int(args.tokens)},
        "weight": {"dtype": str(w_q.dtype), "scale_dtype": str(w_s.dtype),
                   "scale_shape": list(w_s.shape)},
        "operators": ["_C::scaled_int8_quant", "_C::matmul"],
        "thresholds": {"min_cosine": args.cosine_floor,
                       "max_relative_l2": args.max_relative_l2},
        # Measured, not asserted: omitting the 127 factor scales every output
        # channel by the same constant, and cosine similarity is invariant to that.
        # The relative L2 is what makes the control fail, so it is the gate.
        "gate": "relative_l2 on kernel_vs_dequantized_reference, with cosine as a "
                "direction check only",
        "cases": [],
    }

    try:
        candidate, x_q, x_s, static = apply_kernel(x_d, w_q_layer, w_pc_max.to(device))
    except Exception as error:  # the operator itself is what may be missing
        result["state"] = "EVALUATION_ERROR"
        result["error"] = f"{type(error).__name__}: {error}"
        print(json.dumps(result))
        return 1

    convention, x_deq = activation_convention(x, x_q.cpu(), x_s.cpu())
    result["activation"] = {"static": static, "convention": convention,
                            "scale_shape": list(x_s.shape)}

    # Reference on the CPU in float32. Two of them, because they answer different
    # questions: `matched` reuses the candidate's own x_q so any disagreement is
    # the kernel's arithmetic, while `exact` keeps float32 activations and shows
    # the error quantization costs regardless of which kernel runs.
    w_deq = w_q.to(torch.float32) * w_s.to(torch.float32).reshape(-1, 1)
    reference_matched = x_deq @ w_deq.transpose(0, 1)
    reference_exact = x.to(torch.float32) @ w_deq.transpose(0, 1)
    got = candidate.cpu().to(torch.float32)

    result["cases"].append({
        "case": "kernel_vs_dequantized_reference",
        "reference": "float32 dequantized matmul reusing the candidate's quantized activations",
        "cosine": cosine(got, reference_matched),
        "relative_l2": relative_l2(got, reference_matched),
    })
    result["cases"].append({
        "case": "kernel_vs_exact_reference",
        "reference": "float32 matmul on unquantized activations (quantization cost, not a defect)",
        "cosine": cosine(got, reference_exact),
        "relative_l2": relative_l2(got, reference_exact),
    })

    # Negative control: hand the kernel the raw scale, i.e. omit `w_s.mul_(127.0)`.
    # If this still clears the gate, the probe is blind to the convention it claims
    # to check and no pass may be reported.
    try:
        control, _, _, _ = apply_kernel(x_d, w_q_layer, w_s.to(torch.float32).to(device))
        control_out = control.cpu().to(torch.float32)
        control_metrics = {"cosine": cosine(control_out, reference_matched),
                           "relative_l2": relative_l2(control_out, reference_matched)}
        control_error: str | None = None
    except Exception as error:
        control_metrics, control_error = {}, f"{type(error).__name__}: {error}"
    result["control"] = {
        "case": "omit_scale_to_max_conversion",
        "description": "same call with w_pc_max=w_s, i.e. without scale_mm.py's mul_(127.0)",
        **control_metrics,
        "error": control_error,
        "discriminates": control_error is not None
        or control_metrics.get("relative_l2", 0.0) > args.max_relative_l2,
        # Recorded because it is the reason this probe does not gate on cosine: the
        # omission is a uniform per-channel factor, which cosine cannot see.
        "cosine_is_blind_to_this": control_metrics.get("cosine") is not None
        and control_metrics["cosine"] >= args.cosine_floor,
    }

    primary = result["cases"][0]
    passed = (primary["relative_l2"] <= args.max_relative_l2
              and primary["cosine"] >= args.cosine_floor)
    if not result["control"]["discriminates"]:
        result["state"] = "EVALUATION_INCONCLUSIVE"
    else:
        result["state"] = "EXERCISED_PASS" if passed else "EXERCISED_FAIL"
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
