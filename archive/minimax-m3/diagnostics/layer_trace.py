"""First-forward layer trace: find where MiniMax-M3's hidden state stops depending
on the prompt.

The symptom that motivated this: two unrelated prompts produced nearly identical
token streams, which is not what wrong-expert routing looks like -- it looks like
the hidden state being overwritten or blown up somewhere and the prompt's
contribution being lost.

Deliberately generic: it logs fp32 norm/min/max/nan of whatever tensors go in and
come out of each patched module rather than assuming any signature, so it cannot
itself break a forward it does not understand. Rank-gated and forward-count-gated
so it cannot flood 8 ranks' logs, and off unless M3_TRACE is set.

The natural bisection point is layer index 3: MiniMax-M3's first three layers are
dense attention on the platform's own path, layers 3..62 are sparse and run through
the torch MSA stand-ins in m3_torch_index_topk.py / m3_torch_sparse_attn.py.
"""

import os

import torch

ENABLED = os.environ.get("M3_TRACE") == "1"
MAX_FORWARDS = int(os.environ.get("M3_TRACE_FORWARDS", "2"))
# When set, also write every logged tensor to this directory as a .pt, for an
# element-wise comparison against tools/probe/m3_layer0_golden.py. Norm-only comparison
# ran out of resolution: the golden and the service agreed on extreme values but differed
# by ~2x in norm from layer 0's MLP onwards, which a scalar cannot explain.
DUMP_DIR = os.environ.get("M3_TRACE_DUMP", "")
# Only trace forwards with this many tokens, so the startup warmups (8192 dummy tokens)
# do not consume the budget before the real request arrives.
ONLY_TOKENS = int(os.environ.get("M3_TRACE_TOKENS", "0"))

_state = {"forwards": 0, "rank": None, "dumped": 0}


def _rank():
    if _state["rank"] is None:
        try:
            from vllm.distributed import get_tensor_model_parallel_rank

            _state["rank"] = get_tensor_model_parallel_rank()
        except Exception:
            _state["rank"] = 0
    return _state["rank"]


def _describe(value):
    if not isinstance(value, torch.Tensor):
        return None
    flat = value.detach().float().reshape(-1)
    if flat.numel() == 0:
        return "empty"
    nan = int(torch.isnan(flat).sum())
    inf = int(torch.isinf(flat).sum())
    finite = flat[torch.isfinite(flat)]
    if finite.numel() == 0:
        return f"shape={tuple(value.shape)} all-nonfinite nan={nan} inf={inf}"
    return (
        f"shape={tuple(value.shape)} norm={float(finite.norm()):.4g} "
        f"min={float(finite.min()):.4g} max={float(finite.max()):.4g} "
        f"nan={nan} inf={inf}"
    )


def _log(tag, values):
    for name, value in values:
        text = _describe(value)
        if text is not None:
            print(f"[M3_TRACE] {tag} {name}: {text}", flush=True)
            if DUMP_DIR:
                _dump(tag, name, value)


def _dump(tag, name, value):
    import os.path

    safe = f"{tag}.{name}".replace("[", "_").replace("]", "").replace("/", "_")
    path = os.path.join(DUMP_DIR, f"{safe}.pt")
    if os.path.exists(path):
        return
    try:
        os.makedirs(DUMP_DIR, exist_ok=True)
        torch.save(value.detach().float().cpu(), path)
        _state["dumped"] += 1
    except Exception as error:
        print(f"[M3_TRACE] dump {safe} failed: {error!r}", flush=True)


def _wrap(cls, tag_of):
    original = cls.forward

    def forward(self, *args, **kwargs):
        trace = ENABLED and _rank() == 0 and _state["forwards"] < MAX_FORWARDS
        if trace and ONLY_TOKENS:
            # Some call sites pass everything by keyword (the decoder layer calls
            # self_attn(positions=..., hidden_states=...)), so look at both.
            candidates = list(args) + list(kwargs.values())
            tokens = next(
                (a.shape[0] for a in candidates if isinstance(a, torch.Tensor) and a.dim() >= 1),
                None,
            )
            if tokens is not None and tokens != ONLY_TOKENS:
                trace = False
        if trace:
            tag = tag_of(self)
            inputs = [(f"in[{i}]", a) for i, a in enumerate(args)]
            inputs += [(f"in[{k}]", v) for k, v in kwargs.items()]
            _log(tag, inputs)
        result = original(self, *args, **kwargs)
        if trace:
            outputs = (
                [(f"out[{i}]", r) for i, r in enumerate(result)]
                if isinstance(result, tuple)
                else [("out", result)]
            )
            _log(tag, outputs)
        return result

    cls.forward = forward


def patch():
    if not ENABLED:
        return
    from vllm.models.minimax_m3.nvidia import model as m3

    _wrap(m3.MiniMaxM3DecoderLayer, lambda s: f"layer{getattr(s, '_m3_index', '?')}")
    _wrap(m3.MiniMaxM3Attention, lambda s: "dense_attn")
    _wrap(m3.MiniMaxM3SparseAttention, lambda s: "sparse_attn")
    _wrap(m3.MiniMaxM3MoE, lambda s: "moe")
    _wrap(m3.MiniMaxM3MLP, lambda s: "mlp")

    # Layer identity: the decoder layers carry no index of their own, so stamp one
    # from the model's layer list the first time the model runs.
    model_original = m3.MiniMaxM3Model.forward

    def model_forward(self, *args, **kwargs):
        if ENABLED and not getattr(self, "_m3_indexed", False):
            for index, layer in enumerate(getattr(self, "layers", [])):
                layer._m3_index = index
            self._m3_indexed = True
        result = model_original(self, *args, **kwargs)
        # Only spend the forward budget on forwards that were actually traced, otherwise
        # the startup profiling run exhausts it before the first real request.
        candidates = list(args) + list(kwargs.values())
        tokens = next(
            (a.shape[0] for a in candidates if isinstance(a, torch.Tensor) and a.dim() >= 1),
            None,
        )
        if not ONLY_TOKENS or tokens is None or tokens == ONLY_TOKENS:
            _state["forwards"] += 1
        return result

    m3.MiniMaxM3Model.forward = model_forward
