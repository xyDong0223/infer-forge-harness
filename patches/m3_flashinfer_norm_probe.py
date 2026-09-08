"""Probe-only torch stand-in for the two flashinfer norms MiniMax-M3 imports.

Not a port. It exists to answer one question: what breaks *after* the flashinfer
import at `vllm/models/minimax_m3/nvidia/model.py:134`, which is the first wall a
MiniMax-M3 launch hits on P800 —

    ModuleNotFoundError: No module named 'flashinfer'

Semantics are taken from the AMD triton implementation
(`vllm/models/minimax_m3/amd/ops/gemma_rmsnorm.py`): `out = x * rstd * (1.0 + w)` in
float32, and the fused variant writes `x + residual` back into `residual` before
normalising. Gemma's `1 + w` is why the weight tensor is initialised to zeros.

Installed by copying this file to `/tmp/m3_shim/flashinfer/norm.py` in the pod and
launching with `PYTHONPATH=/tmp/m3_shim`, never into site-packages, so removing the
path removes it. `patches/m3_fused_qknorm_rope_probe.py` is registered from the bottom
of the installed copy, because flashinfer.norm is imported at the first layer's norm —
before the first attention call — and nothing else would import it in time.
"""

from __future__ import annotations

import torch


def gemma_rmsnorm(x, weight, eps=1e-6, out=None):
    s = x.float()
    rstd = torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps)
    result = (s * rstd * (1.0 + weight.float())).to(x.dtype)
    if out is not None:
        out.copy_(result)
        return out
    return result


def gemma_fused_add_rmsnorm(x, residual, weight, eps=1e-6):
    """In place, as the caller expects: it reuses both tensors afterwards."""
    s = x.float() + residual.float()
    residual.copy_(s.to(residual.dtype))
    rstd = torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps)
    x.copy_((s * rstd * (1.0 + weight.float())).to(x.dtype))
    return x, residual
