"""Probe-only patch: let an out-of-tree platform through `bind_kv_cache`.

MiniMax-M3's sparse layers own two caches in one decoder layer — the main paged K/V
and the lightning indexer's index cache — so `bind_kv_cache` takes its
"several attention layers share a layer_index" branch. Upstream allows that branch only
for platforms it has checked:

    if (current_platform.is_cuda_alike()
        or current_platform.is_xpu()
        or current_platform.is_cpu()):
        pass
    else:
        raise NotImplementedError

vLLM-Kunlun answers False to all three — measured: `device_type` is "cuda" but
`is_cuda_alike()`, `is_cuda()`, `is_rocm()`, `is_xpu()` and `is_cpu()` are all False,
while `is_out_of_tree()` is True. So the model dies during KV cache initialisation with
a bare NotImplementedError, after weights load and the dummy run both succeed.

The gap is a platform predicate, not a kernel, and it is not MiniMax-M3 specific: any
model with two attention-like caches in one layer hits it — encoder-decoder models
(upstream's own comment names bart) and MLA-plus-indexer models included.

This replacement is a faithful copy of the upstream function with `is_out_of_tree()`
added to the same whitelist. It is a probe, not a fix: the real fix belongs upstream or
in the Kunlun plugin's platform class.
"""

from __future__ import annotations

from collections import defaultdict


def _bind_kv_cache(kv_caches, forward_context, runner_kv_caches, num_attn_module=1):
    from vllm.platforms import current_platform
    from vllm.v1.worker.utils import extract_layer_index

    assert len(runner_kv_caches) == 0

    index2name = defaultdict(list)
    for layer_name in kv_caches:
        index2name[extract_layer_index(layer_name, num_attn_module)].append(layer_name)

    for layer_index in sorted(index2name.keys()):
        layer_names = index2name[layer_index]
        if len(layer_names) > 1 and not (
            current_platform.is_cuda_alike()
            or current_platform.is_xpu()
            or current_platform.is_cpu()
            # The one added line. An out-of-tree platform running the GPU model
            # runner is in the same position as the platforms above.
            or current_platform.is_out_of_tree()
        ):
            raise NotImplementedError
        for layer_name in layer_names:
            runner_kv_caches.append(kv_caches[layer_name])

    for layer_name, kv_cache in kv_caches.items():
        forward_context[layer_name].kv_cache = kv_cache


def patch() -> None:
    """Rebind in both places: the model runner imported the name directly."""
    import vllm.v1.worker.gpu_model_runner as runner
    import vllm.v1.worker.utils as utils

    utils.bind_kv_cache = _bind_kv_cache
    if hasattr(runner, "bind_kv_cache"):
        runner.bind_kv_cache = _bind_kv_cache
