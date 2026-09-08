"""Differential self-check for the torch block-sparse attend, against a reference
that never touches the paged cache.

Why this check exists: 60 of MiniMax-M3's 63 layers run through
``patches/m3_torch_index_topk.py`` + ``patches/m3_torch_sparse_attn.py``, and the
only differential run so far (M3_FORCE_DENSE_ATTEND) compared my sparse path against
my own dense path -- same ``_attend_row``, same ``_split_kv``, same layout
assumption. A bug in the shared part is invisible to it.

This reference is independent of all of that: during the *prefill of a fresh
sequence* the whole visible context is exactly the tokens in this call, so causal
full attention can be computed straight from the q/k/v of this forward, with no
block table, no slot mapping and no cache read. For a prompt shorter than
``sparse_block_size * sparse_topk_blocks`` (128*16 = 2048) the block selection is a
no-op, so the sparse result must equal full attention -- any disagreement is a bug
in the selection, the cache read, the layout, or the GQA grouping.

k/v are captured by wrapping the stand-in's ``_insert`` (the values it scatters into
the main 5-D cache are exactly the normed+roped k/v of these tokens), so the capture
costs nothing and cannot disturb the real forward.
"""

import os

import torch

ENABLED = os.environ.get("M3_ATTEND_CHECK") == "1"
MAX_CHECKS = int(os.environ.get("M3_ATTEND_CHECKS", "3"))
# Only check calls with this many query tokens. The startup warmups run with dummy
# shapes and an unpopulated cache, so they burn the check budget and report NaN on both
# sides; set this to the real prompt's token count to get a meaningful comparison.
ONLY_TOKENS = int(os.environ.get("M3_ATTEND_TOKENS", "0"))

_state = {"checks": 0}
_stash = {}


def _rel_l2(actual, reference):
    actual = actual.detach().float()
    reference = reference.detach().float()
    denominator = reference.norm()
    if denominator == 0:
        return float("nan")
    return float((actual - reference).norm() / denominator)


def _full_causal_attention(query, key, value, num_heads, num_kv_heads, head_dim, scale):
    """[T, num_heads*head_dim] out, computed in fp32 with a plain causal mask."""
    tokens = query.shape[0]
    q = query.reshape(tokens, num_heads, head_dim).float()
    k = key.reshape(tokens, num_kv_heads, head_dim).float()
    v = value.reshape(tokens, num_kv_heads, head_dim).float()
    group = num_heads // num_kv_heads
    mask = torch.full((tokens, tokens), float("-inf"), device=q.device)
    mask = torch.triu(mask, diagonal=1)
    out = torch.empty_like(q)
    for head in range(num_kv_heads):
        keys = k[:, head]
        values = v[:, head]
        queries = q[:, head * group : (head + 1) * group]
        logits = torch.einsum("tgd,sd->gts", queries, keys) * scale
        weights = torch.softmax(logits + mask, dim=-1)
        out[:, head * group : (head + 1) * group] = torch.einsum(
            "gts,sd->tgd", weights, values
        )
    return out.reshape(tokens, num_heads * head_dim)


def patch():
    if not ENABLED:
        return
    import importlib

    from vllm.models.minimax_m3.nvidia import model as m3

    # The stand-in is deployed as m3_probe_ops in the pod and lives as
    # m3_fused_qknorm_rope_probe in the repo; accept either.
    probe = None
    for name in ("m3_probe_ops", "m3_fused_qknorm_rope_probe"):
        try:
            probe = importlib.import_module(name)
            break
        except ImportError:
            continue
    if probe is None or not hasattr(probe, "_insert"):
        print("[M3_ATTEND_CHECK] disabled: cannot find the insert stand-in", flush=True)
        return

    original_insert = probe._insert

    def _insert(cache, which, values, slot_mapping, block_size):
        if cache.dim() == 5:  # main kv cache, not the 3-D index cache
            _stash["k" if which == 0 else "v"] = values.detach().clone()
        return original_insert(cache, which, values, slot_mapping, block_size)

    probe._insert = _insert

    original_run = m3.MiniMaxM3SparseAttention._run_attention

    def _run_attention(self, query, index_query, output):
        result = original_run(self, query, index_query, output)
        tokens = query.shape[0]
        if ONLY_TOKENS and tokens != ONLY_TOKENS:
            return result
        if _state["checks"] < MAX_CHECKS and "k" in _stash and "v" in _stash:
            _state["checks"] += 1
            try:
                key, value = _stash["k"], _stash["v"]

                def nans(tensor):
                    return int(torch.isnan(tensor.detach().float()).sum())

                print(
                    f"[M3_ATTEND_CHECK] tokens={tokens} nan q={nans(query)} "
                    f"k={nans(key)} v={nans(value)} out={nans(result)}",
                    flush=True,
                )
                if key.shape[0] != tokens:
                    print(
                        f"[M3_ATTEND_CHECK] skipped: captured {key.shape[0]} keys for "
                        f"{tokens} queries -- not a fresh-sequence prefill",
                        flush=True,
                    )
                else:
                    reference = _full_causal_attention(
                        query,
                        key,
                        value,
                        self.num_heads,
                        self.num_kv_heads,
                        self.head_dim,
                        self.scaling,
                    )
                    got = result.reshape(tokens, -1)
                    print(
                        f"[M3_ATTEND_CHECK] tokens={tokens} heads={self.num_heads}/"
                        f"{self.num_kv_heads} relL2="
                        f"{_rel_l2(got, reference.reshape(got.shape)):.6g} "
                        f"got_norm={float(got.detach().float().norm()):.6g} "
                        f"ref_norm={float(reference.norm()):.6g}",
                        flush=True,
                    )
            except Exception as error:
                print(f"[M3_ATTEND_CHECK] failed: {error!r}", flush=True)
        return result

    m3.MiniMaxM3SparseAttention._run_attention = _run_attention
