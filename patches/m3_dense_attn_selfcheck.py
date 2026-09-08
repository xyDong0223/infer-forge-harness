"""Differential self-check for the platform's dense attention, with no golden weights and
no tensor-parallel assumptions.

The from-checkpoint golden (`tools/probe/m3_layer0_golden.py`) puts layer 0's attention
output 24.6% off element-wise (cos 0.970) while its *input* matches at 0.0017, and every
structural explanation has been swept and eliminated: qk-norm/RoPE order, causality, the
softmax scale, the kv-head-to-rank mapping, the q-head block, attending past the sequence
bound, sliding windows, V being normed or roped by mistake, per-token int8 activation
quantisation, and an fp8 paged cache. All of them sit at 0.245 +/- 0.002.

Two explanations survive: the platform's dense attention kernel is wrong, or the golden is
(a sharding or convention assumption I cannot see from a scalar). This check settles it
without the golden: during the prefill of a fresh sequence, causal attention over the
q/k/v that were just handed to `Attention.forward` is the answer, computed here in fp32
with no cache, no block table and no sharding arithmetic -- the same instrument that put
the torch block-sparse attend at 1-2% (`patches/m3_sparse_attend_selfcheck.py`).

If the dense kernel lands at 1e-2 like the sparse stand-in did, the golden is at fault and
attention is fine. If it lands near 0.25, the kernel is.
"""

import os

import torch

ENABLED = os.environ.get("M3_DENSE_ATTN_CHECK") == "1"
MAX_CHECKS = int(os.environ.get("M3_DENSE_ATTN_CHECKS", "6"))
ONLY_TOKENS = int(os.environ.get("M3_DENSE_ATTN_TOKENS", "0"))

_state = {"checks": 0}


def _rel_l2(actual, reference):
    actual = actual.detach().float()
    reference = reference.detach().float()
    denominator = reference.norm()
    if denominator == 0:
        return float("nan")
    return float((actual - reference).norm() / denominator)


def _causal(query, key, value, num_heads, num_kv_heads, head_dim, scale):
    tokens = query.shape[0]
    q = query.reshape(tokens, num_heads, head_dim).float()
    k = key.reshape(tokens, num_kv_heads, head_dim).float()
    v = value.reshape(tokens, num_kv_heads, head_dim).float()
    group = num_heads // num_kv_heads
    mask = torch.triu(torch.full((tokens, tokens), float("-inf"), device=q.device), diagonal=1)
    out = torch.empty_like(q)
    for head in range(num_kv_heads):
        queries = q[:, head * group : (head + 1) * group]
        logits = torch.einsum("tgd,sd->gts", queries, k[:, head]) * scale
        weights = torch.softmax(logits + mask, dim=-1)
        out[:, head * group : (head + 1) * group] = torch.einsum(
            "gts,sd->tgd", weights, v[:, head]
        )
    return out.reshape(tokens, num_heads * head_dim)


def patch():
    if not ENABLED:
        return
    from vllm.models.minimax_m3.nvidia import model as m3

    original = m3.MiniMaxM3Attention.forward

    def forward(self, positions, hidden_states):
        tokens = hidden_states.shape[0]
        if not ENABLED or (ONLY_TOKENS and tokens != ONLY_TOKENS):
            return original(self, positions, hidden_states)
        if _state["checks"] >= MAX_CHECKS:
            return original(self, positions, hidden_states)
        _state["checks"] += 1

        qkv, _ = self.qkv_proj(hidden_states)
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
            None, None, 0, None, None, None, None, 0, None, None, "auto",
        )
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        try:
            reference = _causal(
                q, k, v, self.num_heads, self.num_kv_heads, self.head_dim, self.scaling
            )
            served = self.attn(q, k, v)
            print(
                f"[M3_DENSE_ATTN_CHECK] tokens={tokens} heads={self.num_heads}/"
                f"{self.num_kv_heads} scale={self.scaling:.6g} "
                f"relL2={_rel_l2(served, reference.reshape(served.shape)):.6g} "
                f"served_norm={float(served.detach().float().norm()):.6g} "
                f"ref_norm={float(reference.norm()):.6g}",
                flush=True,
            )
        except Exception as error:
            print(f"[M3_DENSE_ATTN_CHECK] failed: {error!r}", flush=True)
        return original(self, positions, hidden_states)

    m3.MiniMaxM3Attention.forward = forward
