"""Torch stand-ins for the two MoE defects that silently corrupt MiniMax-M3 on P800.

Both defects live in one function --
``KunlunCompressedTensorsW8A8Int8MoEMethod.apply_monolithic``
(``vllm_kunlun/quantization/compressed_tensors/compressed_tensors_moe.py:149``) --
and neither raises, which is why the service answered HTTP 200 with fluent
nonsense instead of crashing.

1. The routing arguments never arrive.  Upstream calls the monolithic method with
   four arguments only::

       # vllm/model_executor/layers/fused_moe/routed_experts.py:1209
       return self.quant_method.apply_monolithic(
           layer=self, x=x, router_logits=router_logits, input_ids=input_ids)

   so ``scoring_func`` keeps its default ``"softmax"`` and
   ``e_score_correction_bias`` stays ``None``.  MiniMax-M3 declares
   ``scoring_func="sigmoid"`` with ``use_routing_bias=True``
   (``config.json``), so every token was routed to softmax top-4 experts with no
   bias correction -- wrong experts, wrong weights, no error.

2. The expert activation is hard-coded to plain SwiGLU.  ``apply_monolithic``
   always calls ``torch.ops._C.silu_and_mul``, but M3 asks for
   ``activation="swigluoai_uninterleave"`` with ``swiglu_limit=7.0``,
   ``swiglu_alpha=1.702``, ``swiglu_beta=1.0``
   (``vllm/models/minimax_m3/nvidia/model.py:257``), which upstream maps to
   ``silu_and_mul_with_clamp``::

       gate = clamp(x[..., :d], max=limit)
       up   = clamp(x[..., d:], -limit, limit)
       out  = gate * sigmoid(alpha * gate) * (up + beta)

Measured on the way, and reported to the vendor separately: the vendor kernel's
signature is ``moe_sigmoid_group_topk_norm(x, topk_index, norm_score,
block_statistic, bias, scale, n_group, topk_group)``, while
``vllm_kunlun/ops/_custom_ops.py:1362`` passes ``block_static=``.  The sigmoid
branch therefore could not have run even if it had been selected -- so the
routing here is done in torch rather than by re-routing to that kernel.

``routed_scaling_factor`` is deliberately left at 1.0: M3 sets
``apply_routed_scale_to_output=True``, so ``MoERunner`` applies the 2.0 to the
combined output (``fused_moe/layer.py:411``) and applying it to the weights too
would double-count it.

This is a stand-in, not a fix: the durable fix belongs in vllm_kunlun.
"""

import torch

_PATCHED = "_m3_moe_routing_activation_patched"


def sigmoid_bias_topk(router_logits, bias, top_k, renormalize=True, scale=1.0):
    """M3 routing: select on sigmoid(logits)+bias, weight with sigmoid(logits)."""
    scores = router_logits.float().sigmoid()
    selection = scores if bias is None else scores + bias.float().reshape(1, -1)
    ids = torch.topk(selection, top_k, dim=-1).indices
    weights = scores.gather(-1, ids)
    if renormalize:
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-20)
    if scale != 1.0:
        weights = weights * scale
    return weights, ids


def swigluoai_uninterleave(x, out, limit, alpha, beta):
    """gate = first half, up = second half (packed w13), computed in fp32."""
    d = x.shape[-1] // 2
    gate = x[..., :d].float().clamp(max=limit)
    up = x[..., d:].float().clamp(min=-limit, max=limit)
    out.copy_((gate * torch.sigmoid(alpha * gate) * (up + beta)).to(out.dtype))


def _sigmoid_routing_shim(
    x=None,
    topk_index=None,
    norm_score=None,
    block_statistic=None,
    block_static=None,
    bias=None,
    scale=1.0,
    n_group=1,
    topk_group=1,
    **_,
):
    """Stands in for kunlun_ops.moe_sigmoid_group_topk_norm.

    n_group/topk_group are ignored: M3's config declares no expert groups, so
    grouped selection degenerates to plain top-k over all experts.
    """
    weights, ids = sigmoid_bias_topk(
        x, bias, topk_index.shape[-1], renormalize=True, scale=float(scale)
    )
    topk_index.copy_(ids.to(topk_index.dtype))
    norm_score.copy_(weights.to(norm_score.dtype))


def _activation_name(layer):
    activation = getattr(layer, "activation", "silu")
    return str(getattr(activation, "value", activation))


def patch():
    import kunlun_ops
    from vllm_kunlun.quantization.compressed_tensors import compressed_tensors_moe

    cls = compressed_tensors_moe.KunlunCompressedTensorsW8A8Int8MoEMethod
    if getattr(cls, _PATCHED, False):
        return

    kunlun_ops.moe_sigmoid_group_topk_norm = _sigmoid_routing_shim
    original = cls.apply_monolithic

    def apply_monolithic(self, layer, x, router_logits, input_ids=None, **_):
        activation = _activation_name(layer)
        saved_swiglu = kunlun_ops.swiglu
        if activation == "swigluoai_uninterleave":
            limit = getattr(layer, "swiglu_limit", None)
            if limit is None:
                raise ValueError("swigluoai_uninterleave requires swiglu_limit")
            alpha = float(getattr(layer, "swiglu_alpha", None) or 1.0)
            beta = float(getattr(layer, "swiglu_beta", None) or 0.0)

            def swiglu(x=None, y=None, **__):
                swigluoai_uninterleave(x, y, float(limit), alpha, beta)

            kunlun_ops.swiglu = swiglu
        elif activation not in ("silu", "silu_and_mul"):
            raise NotImplementedError(
                f"apply_monolithic hard-codes SwiGLU; activation={activation!r} "
                "would be computed wrong and silently"
            )
        try:
            return original(
                self,
                layer=layer,
                x=x,
                router_logits=router_logits,
                input_ids=input_ids,
                global_num_experts=getattr(layer, "global_num_experts", -1),
                scoring_func=getattr(layer, "scoring_func", "softmax"),
                e_score_correction_bias=getattr(layer, "e_score_correction_bias", None),
                num_expert_group=getattr(layer, "num_expert_group", None) or 1,
                topk_group=getattr(layer, "topk_group", None) or 1,
                routed_scaling_factor=1.0,
            )
        finally:
            kunlun_ops.swiglu = saved_swiglu

    cls.apply_monolithic = apply_monolithic
    setattr(cls, _PATCHED, True)
