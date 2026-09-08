"""In-pod MoE layer probe (TP path). Prints JSON to stdout.

Exercises one MoE layer forward — `KunlunOps.fused_moe`, the monolithic entry the
Kunlun `UnquantizedFusedMoEMethod` uses — against a float32 CPU reference that does
the same thing the slow way: softmax the router, take top-k, renormalise, run
SwiGLU per expert, and weight the sum.

The geometry is chosen for one reason. `fused_moe` switches preprocessing at
`M * top_k > 768`: above it `_C.moe_pre_sorted`, below it
`xspeedgate_ops.moe_pre_small`. Two implementations of the same step means two
chances to disagree, and nothing else in this repo checks that they agree, so the
probe runs a batch on each side of the threshold and reports which path each took.

Expert parallel is out of scope here: `fused_moe_ep` is a different call and the
confirmed plan is TP first.
"""

from __future__ import annotations

import argparse
import json

THRESHOLD = 768  # M * top_k above which fused_moe switches preprocessing


def torch_reference(x, w13, w2, router_logits, top_k: int, renormalize: bool,
                    select: str = "top"):
    """Float32 MoE forward on the CPU. `select="bottom"` is the negative control.

    Also returns a mask of tokens whose routing is decided by an exact tie between
    the k-th and (k+1)-th expert. Those tokens are not evidence about arithmetic: the
    kernel and `torch.topk` are both correct and simply break the tie differently,
    and one such token was enough to move the batch's relative L2 from 0.009 to 0.031.
    """
    import torch

    x = x.cpu().to(torch.float32)
    w13 = w13.cpu().to(torch.float32)
    w2 = w2.cpu().to(torch.float32)
    scores = torch.softmax(router_logits.cpu().to(torch.float32), dim=-1)
    ordered, _ = torch.sort(scores, dim=-1, descending=(select == "top"))
    tied = ordered[:, top_k - 1] == ordered[:, top_k] if scores.shape[1] > top_k \
        else torch.zeros(scores.shape[0], dtype=torch.bool)
    weights, ids = torch.topk(scores, top_k, dim=-1, largest=(select == "top"))
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)

    intermediate = w13.shape[1] // 2
    out = torch.zeros(x.shape[0], w2.shape[1], dtype=torch.float32)
    for token in range(x.shape[0]):
        for slot in range(top_k):
            expert = int(ids[token, slot])
            fused = x[token] @ w13[expert].transpose(0, 1)
            gate, up = fused[:intermediate], fused[intermediate:]
            activated = torch.nn.functional.silu(gate) * up
            out[token] += float(weights[token, slot]) * (activated @ w2[expert].transpose(0, 1))
    return out, tied


def metrics(candidate, reference, keep=None) -> dict:
    import torch

    a = candidate.cpu().to(torch.float32)
    b = reference.to(torch.float32)
    if keep is not None:
        a, b = a[keep], b[keep]
    flat_a, flat_b = a.flatten(), b.flatten()
    per_token = (a - b).norm(dim=-1) / (b.norm(dim=-1) + 1e-30)
    return {
        "cosine": float(torch.dot(flat_a, flat_b) / (flat_a.norm() * flat_b.norm() + 1e-30)),
        "relative_l2": float((a - b).norm() / (flat_b.norm() + 1e-30)),
        "per_token_relative_l2_max": float(per_token.max()),
        "per_token_relative_l2_median": float(per_token.median()),
    }


def build(experts: int, hidden: int, intermediate: int, tokens: int, seed: int, dtype):
    import torch

    torch.manual_seed(seed)
    scale = hidden**-0.5
    return {
        "x": (torch.randn(tokens, hidden) * scale).to(dtype).cuda(),
        # w13 is [gate; up] stacked, which is the order `_C.silu_and_mul` expects:
        # silu(first half) * second half.
        "w13": (torch.randn(experts, 2 * intermediate, hidden) * scale).to(dtype).cuda(),
        "w2": (torch.randn(experts, hidden, intermediate) * scale).to(dtype).cuda(),
        # Router logits stay float32. In bfloat16 an eight-expert row collides often
        # enough that a single tied token showed up in a 384-token batch, and
        # `fused_moe` casts the logits to float anyway.
        "router_logits": torch.randn(tokens, experts).cuda(),
    }


def main() -> int:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--intermediate", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--tokens-below", type=int, default=16, help="M with M*top_k <= 768")
    parser.add_argument("--tokens-above", type=int, default=512, help="M with M*top_k > 768")
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--device", default="cuda", help="the plugin claims to be CUDA")
    parser.add_argument("--cosine-floor", type=float, default=0.9999)
    parser.add_argument("--max-relative-l2", type=float, default=0.02)
    args = parser.parse_args()

    import vllm  # noqa: F401 - registers the _C namespace
    import vllm._custom_ops  # noqa: F401
    import vllm_kunlun  # noqa: F401

    from vllm_kunlun.ops._kunlun_ops import KunlunOps

    # fused_moe takes its temporaries from the worker's workspace manager, which is
    # normally initialised in `vllm/v1/worker/xpu_worker.py`. Outside a worker it has
    # to be initialised here or the first call asserts.
    from vllm.v1.worker.workspace import init_workspace_manager

    init_workspace_manager(torch.device(args.device), 1)

    dtype = torch.bfloat16
    result: dict = {
        "dimension": "moe",
        "operators": ["_C::moe_softmax_topk_norm", "_C::moe_fc", "_C::silu_and_mul",
                      "_C::moe_post", "_C::moe_pre_sorted",
                      "xspeedgate_ops::moe_pre_small"],
        "geometry": {"experts": args.experts, "hidden": args.hidden,
                     "intermediate": args.intermediate, "top_k": args.top_k,
                     "tokens_below": args.tokens_below, "tokens_above": args.tokens_above},
        "preprocessing_threshold": THRESHOLD,
        "parallelism": "tensor_parallel_path_only",
        "thresholds": {"min_cosine": args.cosine_floor,
                       "max_relative_l2": args.max_relative_l2},
        "gate": "relative_l2 against a float32 CPU MoE forward, on both sides of the "
                "M*top_k=768 preprocessing switch",
        "cases": [],
    }
    for label, tokens in (("below", args.tokens_below), ("above", args.tokens_above)):
        expected = tokens * args.top_k > THRESHOLD
        if expected != (label == "above"):
            result["state"] = "EVALUATION_INCONCLUSIVE"
            result["error"] = (
                f"tokens_{label}={tokens} with top_k={args.top_k} does not land on the "
                f"{label} side of M*top_k={THRESHOLD}, so the two cases would exercise the "
                "same preprocessing path"
            )
            print(json.dumps(result))
            return 1

    references = {}
    for label, tokens in (("below", args.tokens_below), ("above", args.tokens_above)):
        case = build(args.experts, args.hidden, args.intermediate, tokens, args.seed, dtype)
        reference, tied = torch_reference(case["x"], case["w13"], case["w2"],
                                         case["router_logits"], args.top_k, True)
        references[label] = (case, reference, tied)
        try:
            output = KunlunOps.fused_moe(
                case["x"], case["w13"], case["w2"], case["router_logits"],
                0, args.top_k, renormalize=True, inplace=False, use_grouped_topk=False,
                scoring_func="softmax",
            )
            torch.cuda.synchronize()
        except Exception as error:
            result["state"] = "EVALUATION_ERROR"
            result["error"] = f"M={tokens}: {type(error).__name__}: {error}"
            print(json.dumps(result))
            return 1
        result["cases"].append({
            "case": f"fused_moe_vs_torch_reference_M{tokens}",
            "reference": "float32 CPU MoE forward: softmax router, top-k, renormalise, "
                         "per-expert SwiGLU",
            "preprocessing": "_C::moe_pre_sorted" if label == "above"
                             else "xspeedgate_ops::moe_pre_small",
            "router_ties_excluded": int(tied.sum()),
            **metrics(output, reference, keep=~tied),
        })

    # Control: route to the bottom-k experts instead. If that also matches, expert
    # selection does not affect the output at this geometry and the comparison is
    # blind to routing.
    case, _, tied = references["below"]
    control_reference, _ = torch_reference(case["x"], case["w13"], case["w2"],
                                          case["router_logits"], args.top_k, True,
                                          select="bottom")
    output = KunlunOps.fused_moe(
        case["x"], case["w13"], case["w2"], case["router_logits"],
        0, args.top_k, renormalize=True, inplace=False, use_grouped_topk=False,
        scoring_func="softmax",
    )
    torch.cuda.synchronize()
    control_metrics = metrics(output, control_reference, keep=~tied)
    result["control"] = {
        "case": "route_to_the_bottom_k_experts",
        "description": "the same call compared against a reference that selects the "
                       "least-scored experts",
        **control_metrics,
        "error": None,
        "discriminates": control_metrics["relative_l2"] > args.max_relative_l2,
    }

    worst = max(case["relative_l2"] for case in result["cases"])
    weakest = min(case["cosine"] for case in result["cases"])
    passed = worst <= args.max_relative_l2 and weakest >= args.cosine_floor
    # Recorded on purpose: the two preprocessing paths agreeing with the reference is
    # a stronger statement than either one alone.
    result["paths_agree_with_reference"] = passed
    if not result["control"]["discriminates"]:
        result["state"] = "EVALUATION_INCONCLUSIVE"
    else:
        result["state"] = "EXERCISED_PASS" if passed else "EXERCISED_FAIL"
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
