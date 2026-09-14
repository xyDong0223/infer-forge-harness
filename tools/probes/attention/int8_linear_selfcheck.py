"""Differential self-check for the int8 dense linear path on P800.

The layer trace showed MiniMax-M3's layer1 dense MLP saturating its SwiGLU clamp
identically for unrelated prompts, with the MLP *input* already reaching ~31 after
a GemmaRMSNorm. If the int8 linear's scale convention is off, everything downstream
inherits the wrong magnitude and the clamp turns it into prompt-independent mush.

The suspicious line is the activation scale in
``vllm_kunlun/quantization/kernels/scale_mm.py:87``::

    x_pc_max=x_s * 127.0 if static else x_s

The kernel takes an amax ("max") convention -- which is why the *weight* scale is
multiplied by 127 at load time (``:50``). For static per-tensor quant the activation
scale is likewise promoted to amax. For **dynamic per-token** quant -- which is what
this checkpoint uses (``input_activations.strategy == "token"``, dynamic) -- ``x_s``
is passed through unpromoted. Whether that is correct depends entirely on what
``scaled_int8_quant`` returns for the dynamic case, so measure it rather than argue.

Reference: dequantize the stored int8 weight with its own per-channel scale and do
the matmul in fp32. Relative L2 is the metric -- cosine is blind to exactly the kind
of uniform scale error under suspicion here.
"""

import os

import torch

ENABLED = os.environ.get("M3_LINEAR_CHECK") == "1"
MAX_CHECKS = int(os.environ.get("M3_LINEAR_CHECKS", "2"))

_state = {"checks": 0}


def _rel_l2(actual, reference):
    actual = actual.detach().float()
    reference = reference.detach().float()
    denominator = reference.norm()
    if denominator == 0:
        return float("nan")
    return float((actual - reference).norm() / denominator)


def _report_quant(x):
    """What scaled_int8_quant actually hands back for a dynamic per-token quant."""
    x_q, x_s, x_zp, static = torch.ops._C.scaled_int8_quant(
        x=x.contiguous(), scale=None, azp=None, symmetric=True
    )
    amax = x.detach().float().abs().amax(dim=-1, keepdim=True)
    ratio = (x_s.detach().float() / amax.clamp_min(1e-12)).reshape(-1)
    print(
        f"[M3_LINEAR_CHECK] scaled_int8_quant static={static} "
        f"x_s/amax min={float(ratio.min()):.6g} max={float(ratio.max()):.6g} "
        f"(1.0 => returns amax, ~0.00787 => returns amax/127)",
        flush=True,
    )
    return x_q, x_s


def patch():
    if not ENABLED:
        return
    from vllm.models.minimax_m3.nvidia import model as m3

    original = m3.MiniMaxM3MLP.forward

    def forward(self, x):
        check = _state["checks"] < MAX_CHECKS
        if check:
            _state["checks"] += 1
            try:
                linear = self.gate_up_proj
                weight = linear.weight
                scale = getattr(linear, "weight_scale", None)
                print(
                    f"[M3_LINEAR_CHECK] weight {tuple(weight.shape)} {weight.dtype} "
                    f"scale {None if scale is None else tuple(scale.shape)} "
                    f"x {tuple(x.shape)} {x.dtype}",
                    flush=True,
                )
                _report_quant(x)
                actual, _ = linear(x)
                if scale is not None and weight.dtype == torch.int8:
                    # Measured: the weight is stored [in_features, out_features]
                    # (6144, 3072) with a per-output-channel scale (3072, 1) -- which is
                    # why apply_weights hands the kernel w_q.transpose(0, 1). And
                    # weight_scale carries amax, process_weights_after_loading having
                    # multiplied the checkpoint's scale by 127.
                    dequantized = weight.float() * (
                        scale.float().reshape(1, -1) / 127.0
                    )
                    reference = x.detach().float() @ dequantized
                    print(
                        f"[M3_LINEAR_CHECK] gate_up relL2={_rel_l2(actual, reference):.6g} "
                        f"actual_norm={float(actual.detach().float().norm()):.6g} "
                        f"reference_norm={float(reference.norm()):.6g}",
                        flush=True,
                    )
                    _check_activation_and_down(self, actual)
            except Exception as error:
                print(f"[M3_LINEAR_CHECK] failed: {error!r}", flush=True)
        return original(self, x)

    m3.MiniMaxM3MLP.forward = forward


def _check_activation_and_down(mlp, gate_up):
    """Grade the clamped SwiGLU and the down_proj, the two steps the gate_up check left.

    The golden reference (tools/probe/m3_layer0_golden.py) put layer 0's MLP output at
    0.74x of what the service produces and layer 1's at 2.2x, while the extreme values
    matched -- so the divergence is inside the MLP, after gate_up. down_proj's input is
    the activated tensor, whose few huge lanes (up to limit*(limit+beta) ~ 56) set the
    per-token int8 step, so the same measurement is taken twice: once against an fp32
    activation (what the golden used) and once against the int8-quantized activation the
    kernel actually consumes. If only the fp32 one is off, the gap is inherent to W8A8;
    if both are off, the kernel is wrong.
    """
    activated = mlp.act_fn(gate_up)
    d = gate_up.shape[-1] // 2
    gate = gate_up.detach().float()[..., :d].clamp(max=mlp.act_fn.swiglu_limit)
    up = gate_up.detach().float()[..., d:].clamp(
        min=-mlp.act_fn.swiglu_limit, max=mlp.act_fn.swiglu_limit
    )
    act_reference = gate * torch.sigmoid(mlp.act_fn.alpha * gate) * (up + mlp.act_fn.beta)
    print(
        f"[M3_LINEAR_CHECK] act relL2={_rel_l2(activated, act_reference):.6g} "
        f"clamped_lanes={float((gate_up.detach().float()[..., :d] > mlp.act_fn.swiglu_limit).float().mean()):.4g} "
        f"act_max={float(act_reference.max()):.6g}",
        flush=True,
    )

    down = mlp.down_proj
    weight, scale = down.weight, getattr(down, "weight_scale", None)
    if scale is None or weight.dtype != torch.int8:
        return
    dequantized = weight.float() * (scale.float().reshape(1, -1) / 127.0)
    actual, _ = down(activated)
    fp32_reference = activated.detach().float() @ dequantized
    amax = activated.detach().float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    requantized = torch.round(activated.detach().float() / amax * 127.0).clamp(-127, 127)
    int8_reference = (requantized * (amax / 127.0)) @ dequantized
    print(
        f"[M3_LINEAR_CHECK] down relL2_fp32={_rel_l2(actual, fp32_reference):.6g} "
        f"relL2_int8act={_rel_l2(actual, int8_reference):.6g} "
        f"actual_norm={float(actual.detach().float().norm()):.6g} "
        f"fp32_norm={float(fp32_reference.norm()):.6g} "
        f"int8_norm={float(int8_reference.norm()):.6g}",
        flush=True,
    )
