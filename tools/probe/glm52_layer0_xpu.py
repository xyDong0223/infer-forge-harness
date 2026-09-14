"""Layer-0 XPU capture v3: on-disk model with REAL layer-0 + embed + dummy MoE layers 1-4.

Layer 0 is dense (first_k_dense_replace=3 in the real config) with real weights;
layers 1..N are MoE layers with dummy experts so the plugin's model init finds a
MoE layer (it refuses a fully dense model). Layer-0's hook in/out is unaffected
by the dummy layers. Same 6 tokens as glm52_layer0_golden.py.
Usage: python3 glm52_layer0_xpu.py <model_root> <out_dir>
"""
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

MODEL_ROOT = sys.argv[1]
OUT_DIR = sys.argv[2]
TOKENS = [785, 3974, 13867, 38627, 34041, 916]
WORK = "/tmp/glm52_layer0_model"
DUMMY_LAYERS = [1, 2, 3, 4]  # MoE layers, dummy experts


def build(model_root):
    cfg = json.load(open(os.path.join(model_root, "config.json")))
    real_layers = cfg["num_hidden_layers"]
    cfg["num_hidden_layers"] = 1 + len(DUMMY_LAYERS)
    cfg["first_k_dense_replace"] = 1  # only layer 0 is dense
    cfg["n_routed_experts"] = 8
    cfg.pop("num_nextn_predict_layers", None)
    shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(WORK)
    json.dump(cfg, open(os.path.join(WORK, "config.json"), "w"), indent=1)
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
                 "chat_template.jinja"):
        src = os.path.join(model_root, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(WORK, name))

    index = json.load(open(os.path.join(model_root, "model.safetensors.index.json")))["weight_map"]
    handles = {}

    def real(key):
        shard = os.path.join(model_root, index[key])
        if shard not in handles:
            handles[shard] = safe_open(shard, framework="pt")
        return handles[shard].get_tensor(key)

    tensors = {}
    # real: embedding, final norm, lm_head, full layer 0
    for key in index:
        if (key.startswith("model.embed_tokens") or key.startswith("model.norm.")
                or key.startswith("lm_head") or key.startswith("model.layers.0.")):
            tensors[key] = real(key)

    # dummy MoE layers: attention copied from REAL layer 0 (valid shapes/values),
    # MLP with 8 dummy experts + shared expert + router.
    hidden = cfg["hidden_size"]
    e_inter = cfg["moe_intermediate_size"]
    shared_inter = e_inter * cfg["n_shared_experts"]
    n_experts = cfg["n_routed_experts"]
    g = torch.Generator().manual_seed(7)
    for layer in DUMMY_LAYERS:
        prefix = f"model.layers.{layer}."
        for key in index:
            if key.startswith("model.layers.0."):
                tensors[prefix + key[len("model.layers.0."):]] = real(key).clone()
        tensors[prefix + "mlp.gate.weight"] = torch.randn(n_experts, hidden, generator=g) * 0.02
        for name, out_f in (("gate_proj", e_inter), ("up_proj", e_inter), ("down_proj", hidden)):
            for expert in range(n_experts):
                w = torch.randint(-32, 32, (out_f, hidden), generator=g, dtype=torch.int8)
                tensors[prefix + f"mlp.experts.{expert}.{name}.weight"] = w
                tensors[prefix + f"mlp.experts.{expert}.{name}.weight_scale"] = torch.full((out_f,), 1e-3)
        for name, out_f in (("gate_proj", shared_inter), ("up_proj", shared_inter),
                            ("down_proj", hidden)):
            w = torch.randint(-32, 32, (out_f, hidden), generator=g, dtype=torch.int8)
            tensors[prefix + f"mlp.shared_experts.{name}.weight"] = w
            tensors[prefix + f"mlp.shared_experts.{name}.weight_scale"] = torch.full((out_f,), 1e-3)
        tensors[prefix + "mlp.gate.e_score_correction_bias"] = torch.zeros(n_experts)
    save_file(tensors, os.path.join(WORK, "model.safetensors"))
    json.dump({"metadata": {"total_size": 0},
               "weight_map": {k: "model.safetensors" for k in tensors}},
              open(os.path.join(WORK, "model.safetensors.index.json"), "w"))
    return {"real_layers": real_layers, "n_tensors": len(tensors)}


if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    info = build(MODEL_ROOT)
    print(f"built: {info['n_tensors']} tensors (real depth {info['real_layers']})")

    llm = LLM(model=WORK, tensor_parallel_size=1, max_model_len=64, dtype="bfloat16",
              enforce_eager=True, gpu_memory_utilization=0.3, trust_remote_code=False)

    core = llm.llm_engine.engine_core.engine_core  # InprocClient -> EngineCore
    runner = core.model_executor.driver_worker.model_runner
    captured = {}
    layer0 = runner.model.model.layers[0]

    def layer_hook(module, args, output):
        # DeepseekV2DecoderLayer.forward(positions, hidden_states, residual):
        # args[0] is positions; the hidden states we want are args[1].
        # The engine uses the deferred-residual protocol: the layer returns
        # (hidden, residual) and the effective output is hidden + residual.
        captured["in"] = args[1].detach().float().cpu()
        captured["out"] = output[0].detach().float().cpu()
        captured["residual"] = output[1].detach().float().cpu() if (
            isinstance(output, tuple) and len(output) > 1 and torch.is_tensor(output[1])
        ) else None

    layer0.register_forward_hook(layer_hook)

    # Bisect: submodule outputs for stage-level comparison against the golden.
    def _post(module, args, output):
        o = output[0] if isinstance(output, tuple) else output
        return o.detach().float().cpu() if torch.is_tensor(o) else None

    layer0.input_layernorm.register_forward_hook(
        lambda m, a, o: captured.__setitem__("post_ln", _post(m, a, o)))
    layer0.self_attn.register_forward_hook(
        lambda m, a, o: captured.__setitem__("attn_out", _post(m, a, o)))
    layer0.mlp.register_forward_hook(
        lambda m, a, o: captured.__setitem__("mlp_out", _post(m, a, o)))

    # W8A8 kernel discrimination: capture gate_proj in/out and the module's
    # loaded weight/scale so the kernel's output can be re-derived in pure
    # torch from the very same quantized values.
    gate = layer0.mlp.gate_up_proj  # MergedColumnParallelLinear: [gate | up]
    def gate_hook(module, args, output):
        x_in = args[0].detach()
        o = output[0] if isinstance(output, tuple) else output
        captured["gate_in"] = x_in.float().cpu()
        captured["gate_out"] = o.detach().float().cpu()
        w = getattr(module, "weight", None)
        if w is not None:
            captured["gate_w"] = w.detach().cpu()
        s = getattr(module, "weight_scale", None)
        if s is not None:
            captured["gate_scale"] = s.detach().cpu()
        s2 = getattr(module, "weight_scale_inv", None)
        if s2 is not None:
            captured["gate_scale_inv"] = s2.detach().cpu()
    gate.register_forward_hook(gate_hook)
    out = llm.generate([{"prompt_token_ids": TOKENS}],
                       SamplingParams(max_tokens=1, temperature=0.0))
    torch.save(captured, os.path.join(OUT_DIR, "xpu_layer0.pt"))
    print("CAPTURED", {k: tuple(v.shape) for k, v in captured.items()})
    print("HOOK_IN_NORM", float(captured["in"].norm()))
    print("HOOK_OUT_NORM", float(captured["out"].norm()))
    print("NEXT_TOKEN", out[0].outputs[0].token_ids)
