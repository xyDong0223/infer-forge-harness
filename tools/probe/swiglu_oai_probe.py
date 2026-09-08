"""SwiGLU-OAI (`swigluoai`) probe. Prints JSON to stdout.

MiniMax-M3's MLP is `SiluAndMulWithClamp(limit=7.0, alpha=1.702, beta=1.0)`, and
`kunlun_ops` has no `swigluoai_and_mul` — only plain swiglu variants. That looked like a
gap to be filled by composing the clamp with the existing kernel.

Measured instead: `SiluAndMulWithClamp.__init__` binds `torch.ops._C.silu_and_mul_with_clamp`
only under `is_cuda_alike()`, and vLLM-Kunlun answers False to every predicate it tests,
so dispatch falls through to `CustomOp.forward_oot`, which resolves to the native torch
path — and that path *is* upstream's own reference formula. So there is nothing to
compose for correctness; the only open question is whether the native path is fast
enough, which is a performance item and not this dimension's business.

What is left worth checking is that the fallthrough is numerically what upstream
documents, so this grades the layer as it will actually run against a float32 reference
computed off the accelerator. The control drops the clamp: with inputs scaled so a real
fraction of them exceed the limit, a comparison that cannot see the clamp cannot see the
difference between swigluoai and plain swiglu either.
"""

from __future__ import annotations

import argparse
import json


def metrics(candidate, reference) -> dict:
    import torch

    a = candidate.detach().cpu().to(torch.float32).flatten()
    b = reference.detach().cpu().to(torch.float32).flatten()
    return {
        "cosine": float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-30)),
        "relative_l2": float((a - b).norm() / (b.norm() + 1e-30)),
    }


def cpu_reference(x, limit: float, alpha: float, beta: float, clamp: bool):
    """The formula from vllm/model_executor/layers/activation.py:190, in float32.

    Note the asymmetry, which is easy to get wrong by symmetry: the gate is clamped
    from above only, the up projection from both sides.
    """
    import torch

    f = x.detach().cpu().to(torch.float32)
    d = f.shape[-1] // 2
    gate, up = f[..., :d], f[..., d:]
    if clamp:
        gate = torch.clamp(gate, max=limit)
        up = torch.clamp(up, min=-limit, max=limit)
    return gate * torch.sigmoid(alpha * gate) * (up + beta)


def main() -> int:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--limit", type=float, default=7.0)
    parser.add_argument("--alpha", type=float, default=1.702)
    parser.add_argument("--beta", type=float, default=1.0)
    # Scaled so a real fraction of the inputs exceed the limit. With plain randn
    # almost nothing clamps and the control cannot discriminate.
    parser.add_argument("--input-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--cosine-floor", type=float, default=0.9999)
    parser.add_argument("--max-relative-l2", type=float, default=0.01)
    args = parser.parse_args()

    import vllm_kunlun  # noqa: F401 - installs the platform plugin
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp
    from vllm.platforms import current_platform

    result: dict = {
        "dimension": "swiglu_oai",
        "operators": ["vllm.model_executor.layers.activation.SiluAndMulWithClamp"],
        "geometry": {"tokens": args.tokens, "intermediate": args.intermediate,
                     "limit": args.limit, "alpha": args.alpha, "beta": args.beta,
                     "input_scale": args.input_scale},
        "thresholds": {"min_cosine": args.cosine_floor,
                       "max_relative_l2": args.max_relative_l2},
        "gate": "relative_l2 of the layer as it dispatches here against a float32 reference",
        "cases": [],
    }

    torch.manual_seed(args.seed)
    x = torch.randn((args.tokens, 2 * args.intermediate),
                    dtype=torch.bfloat16, device="cuda") * args.input_scale
    try:
        with set_current_vllm_config(VllmConfig()):
            layer = SiluAndMulWithClamp(swiglu_limit=args.limit, alpha=args.alpha,
                                        beta=args.beta)
        out = layer(x)
        torch.cuda.synchronize()
    except Exception as error:
        result["state"] = "EVALUATION_ERROR"
        result["error"] = f"{type(error).__name__}: {error}"
        print(json.dumps(result))
        return 1

    # Which path a Kunlun launch actually takes, recorded rather than assumed: the
    # layer binds the custom op only under is_cuda_alike(), which is False here.
    result["dispatch"] = {
        "bound_method": getattr(getattr(layer, "_forward_method", None), "__qualname__", None),
        "custom_op_bound": hasattr(layer, "op"),
        "platform_predicates": {name: getattr(current_platform, name)()
                                for name in ("is_cuda_alike", "is_xpu", "is_cpu",
                                             "is_rocm", "is_out_of_tree")},
    }

    d = args.intermediate
    clamped = float(((x[..., :d].float() > args.limit)
                     | (x[..., d:].float().abs() > args.limit)).float().mean())
    reference = cpu_reference(x, args.limit, args.alpha, args.beta, clamp=True)
    result["cases"].append({
        "case": "layer_vs_cpu_reference",
        "reference": "gate * sigmoid(alpha * gate) * (up + beta), both halves clamped, float32",
        "fraction_of_lanes_clamped": clamped,
        **metrics(out, reference),
    })

    control = metrics(out, cpu_reference(x, args.limit, args.alpha, args.beta, clamp=False))
    result["control"] = {
        "case": "reference_without_the_clamp",
        "description": "the same output graded against plain swiglu, which is what this "
                       "model would be if the limit were ignored",
        **control,
        "discriminates": clamped > 0.01 and control["relative_l2"] > args.max_relative_l2,
    }

    primary = result["cases"][0]
    passed = (primary["relative_l2"] <= args.max_relative_l2
              and primary["cosine"] >= args.cosine_floor)
    if not result["control"]["discriminates"]:
        result["state"] = "EVALUATION_INCONCLUSIVE"
        result["error"] = (
            f"only {clamped:.3%} of lanes clamped, so the geometry does not distinguish "
            "swigluoai from plain swiglu"
        )
    else:
        result["state"] = "EXERCISED_PASS" if passed else "EXERCISED_FAIL"
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

