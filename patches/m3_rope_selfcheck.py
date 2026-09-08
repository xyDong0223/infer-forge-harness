"""Differential self-check for the qk-norm + partial-RoPE stand-in, run inside the
serving process against vLLM's own rotary module.

`patches/m3_fused_qknorm_rope_probe.py` stands in for
`_C::fused_minimax_m3_qknorm_rope_kv_insert`, and its offline probe graded it
against a reference *I wrote*.  That leaves the convention itself unverified: if
the cos/sin cache halves, the NeoX rotate-halves split, or the partial-rotary
boundary were read the wrong way, both the stand-in and its reference would agree
and still be wrong, and the model would emit fluent nonsense -- which is the
symptom under investigation.

This compares the stand-in's in-place q/k against an independent reference built
from objects the model already owns: ``self.q_norm`` / ``self.k_norm`` (the model's
own GemmaRMSNorm) and ``self.rotary_emb`` (vLLM's ``RotaryEmbedding``, whose
forward is a different implementation of the same convention).  Gated by
M3_ROPE_CHECK and by forward count so it cannot flood the log; it recomputes
``qkv_proj`` on a copy, so it does not perturb the real forward.
"""

import os

import torch

ENABLED = os.environ.get("M3_ROPE_CHECK") == "1"
MAX_CHECKS = int(os.environ.get("M3_ROPE_CHECKS", "3"))

_state = {"checks": 0}


def _rel_l2(actual, reference):
    actual = actual.detach().float()
    reference = reference.detach().float()
    denominator = reference.norm()
    if denominator == 0:
        return float("nan")
    return float((actual - reference).norm() / denominator)


def patch():
    if not ENABLED:
        return
    from vllm.models.minimax_m3.nvidia import model as m3

    original = m3.MiniMaxM3Attention.forward

    def forward(self, positions, hidden_states):
        check = _state["checks"] < MAX_CHECKS
        reference = None
        if check:
            try:
                qkv, _ = self.qkv_proj(hidden_states)
                q, k, _ = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
                q = q.reshape(-1, self.num_heads, self.head_dim).clone()
                k = k.reshape(-1, self.num_kv_heads, self.head_dim).clone()
                # The model's own norm, applied over the head dimension.
                q = self.q_norm(q.reshape(-1, self.head_dim)).reshape(q.shape)
                k = self.k_norm(k.reshape(-1, self.head_dim)).reshape(k.shape)
                # vLLM's rotary module: an independent implementation of the same
                # partial-NeoX convention the stand-in reimplements.
                q_flat = q.reshape(q.shape[0], -1).contiguous()
                k_flat = k.reshape(k.shape[0], -1).contiguous()
                q_ref, k_ref = self.rotary_emb(positions, q_flat, k_flat)
                reference = (q_ref.clone(), k_ref.clone())
            except Exception as error:  # a failed check must not break serving
                print(f"[M3_ROPE_CHECK] reference failed: {error!r}", flush=True)
                reference = None

        result = original(self, positions, hidden_states)

        if check and reference is not None:
            _state["checks"] += 1
            # Re-derive what the stand-in produced: run qkv_proj again and apply
            # the registered op, exactly as the real forward just did.
            qkv, _ = self.qkv_proj(hidden_states)
            # The registered schema has no defaults -- torch custom ops require every
            # argument -- so spell out the optional tail the model's Python wrapper fills.
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
                None,  # index_q_norm_weight
                None,  # index_k_norm_weight
                0,  # num_index_heads
                None,  # slot_mapping
                None,  # index_slot_mapping
                None,  # kv_cache
                None,  # index_cache
                0,  # block_size
                None,  # q_out
                None,  # index_q_out
                "auto",  # kv_cache_dtype
            )
            q_got, k_got, _ = qkv.split(
                [self.q_size, self.kv_size, self.kv_size], dim=-1
            )
            q_ref, k_ref = reference
            print(
                f"[M3_ROPE_CHECK] rotary_dim={self.rotary_emb.rotary_dim} "
                f"head_dim={self.head_dim} tokens={hidden_states.shape[0]} "
                f"q_relL2={_rel_l2(q_got, q_ref.reshape(q_got.shape)):.4g} "
                f"k_relL2={_rel_l2(k_got, k_ref.reshape(k_got.shape)):.4g}",
                flush=True,
            )
        return result

    m3.MiniMaxM3Attention.forward = forward
