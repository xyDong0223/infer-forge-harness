"""Layer-swap diagnosis, standardized: one real layer, everything else cheap.

When a 707 GiB model replies garbage, the question is which layer broke —
and the answer cannot be bought with full-model restarts. The method that
worked on GLM5.2 (run glm52-int-w8a8-p800-001, 2026-09-14) was a hand-built
probe: a tiny on-disk model with the real layer-0 plus dummy MoE layers
(the plugin's init refuses an all-dense model), run through in-process vLLM
with forward hooks, compared stage-by-stage against a CPU reference built
from the same checkpoint weights. That method localized the W8A8 kernel
acquittal and the layer-0 acquittal without a single full-model reload.

This tool is that method as one command, so the next model does not get a
hand-written probe per question. Three subcommands, each independently
testable:

  build    --model-path P --real-layers 0 --depth 5 --out-dir W
            Weight-name-driven, HF-layout checkpoints: real embed/norm/
            lm_head/layer-N, donor-cloned attention in the dummy layers,
            fabricated int8 MoE experts there (the plugin needs the
            structure, not the values). config surgery is explicit and
            recorded in the manifest.

  capture  --swap-dir W --real-layers 0 --out O
            In-process vLLM (VLLM_ENABLE_V1_MULTIPROCESSING=0), forward
            hooks on the real layers. The deferred-residual protocol is
            encoded, not rediscovered: args[0] is positions, hidden states
            are args[1], the layer returns (hidden, residual) and the
            EFFECTIVE output is hidden + residual — the 2026-09-14 session
            burned a round on a false 112% error by misreading exactly
            this, so the capture records out, residual AND effective_out.

  compare  --capture O1 --reference O2 --out report.json
            relL2 + norms per stage, verdict per tolerance, and protocol
            guards: both legs must carry the same tokens and manifest —
            comparing captures taken on different inputs is the classic
            silent-corruption failure.

The reference leg can be anything that produces the same tensor-dict
format: the CPU golden (e.g. layer0_golden.py), another build,
another platform, another engine version.

Limits, honestly: the conventions cover the DeepSeek/GLM MoE family the
harness serves (mlp.experts.{e}.{gate,up,down}_proj + weight_scale,
mlp.shared_experts, first_k_dense_replace); an exotic architecture needs
its own dummy-fabrication rules before this builds a valid swap.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "generation_config.json",
    "chat_template.jinja",
)
# Config keys that must not survive into a swap model: MTP modules are extra
# depth the swap does not reproduce, and their absence is recorded, not
# silent.
DROP_CONFIG_KEYS = ("num_nextn_predict_layers",)

# In-process V1: LLM -> llm_engine -> engine_core (InprocClient) ->
# engine_core (EngineCore) -> model_executor.driver_worker.model_runner
ENGINE_MODEL_PATHS = (
    ("llm_engine", "engine_core", "engine_core", "model_executor",
     "driver_worker", "model_runner", "model"),
)


def parse_layers(spec: str) -> list[int]:
    return [int(part) for part in spec.split(",") if part.strip()]


def load_index(model_path: Path) -> dict[str, str]:
    index_file = model_path / "model.safetensors.index.json"
    if index_file.exists():
        return json.loads(index_file.read_text(encoding="utf-8"))["weight_map"]
    single = model_path / "model.safetensors"
    if single.exists():
        with safe_open(single, framework="pt") as handle:
            return {key: "model.safetensors" for key in handle.keys()}
    raise SystemExit(f"no safetensors weights found under {model_path}")


class Checkpoint:
    """Lazy shard reader: one open handle per shard, tensors on demand."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.index = load_index(root)
        self._handles: dict[str, object] = {}

    def get(self, key: str) -> torch.Tensor:
        shard = self.root / self.index[key]
        path = str(shard)
        if path not in self._handles:
            self._handles[path] = safe_open(path, framework="pt")
        return self._handles[path].get_tensor(key)


def layer_index(key: str) -> int | None:
    parts = key.split(".")
    if len(parts) > 2 and parts[0] == "model" and parts[1] == "layers":
        try:
            return int(parts[2])
        except ValueError:
            return None
    return None


def build_swap(model_path: Path, real_layers: list[int], depth: int,
               out_dir: Path, dummy_experts: int = 8,
               expert_scale: float = 1e-3, seed: int = 7) -> dict:
    """Write the swap model; return its manifest.

    Real: embed_tokens, final norm, lm_head, and every key of the chosen
    layers. Dummy layers clone a donor layer's weights (valid shapes for
    attention and norms) and replace the MoE block with fabricated int8
    experts — the plugin needs the structure, and scale 1e-3 keeps the
    dummies numerically inert so they cannot contaminate the target layer's
    captured input beyond one layer of propagation.
    """
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    real_depth = int(config["num_hidden_layers"])
    donor = min(real_layers)
    if max(real_layers) >= real_depth:
        raise SystemExit(f"real layer {max(real_layers)} beyond depth {real_depth}")
    dummy_layers = [i for i in range(depth) if i not in real_layers]

    surgery = {
        "num_hidden_layers": depth,
        "first_k_dense_replace": 1,
        "n_routed_experts": dummy_experts,
    }
    dropped = [key for key in DROP_CONFIG_KEYS if config.pop(key, None) is not None]
    config.update(surgery)
    (out_dir).mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(config, indent=1), encoding="utf-8")
    for name in TOKENIZER_FILES:
        src = model_path / name
        if src.exists():
            shutil.copy(src, out_dir / name)

    checkpoint = Checkpoint(model_path)
    tensors: dict[str, torch.Tensor] = {}

    def is_real(key: str) -> bool:
        if key.startswith(("model.embed_tokens", "model.norm.", "lm_head")):
            return True
        return layer_index(key) in real_layers

    for key in checkpoint.index:
        if is_real(key):
            tensors[key] = checkpoint.get(key)

    # Dummy layers: donor clone + fabricated experts. The donor's own MoE
    # keys (if it has any) are dropped so only one fabrication recipe exists.
    # The clone keeps scale keys alongside their int8 weights: a W8A8
    # checkpoint carries weight+weight_scale pairs, and cloning one without
    # the other hands the loader a weight it cannot dequantize.
    hidden = int(config["hidden_size"])
    generator = torch.Generator().manual_seed(seed)
    moe_keys = ("mlp.experts.", "mlp.shared_experts.", "mlp.gate.")
    for layer in dummy_layers:
        prefix = f"model.layers.{layer}."
        donor_prefix = f"model.layers.{donor}."
        for key in checkpoint.index:
            if key.startswith(donor_prefix):
                name = key[len(donor_prefix):]
                if any(marker in key for marker in moe_keys):
                    continue  # replaced by fabrication below
                tensors[prefix + name] = checkpoint.get(key).clone()
        fabricate_moe(tensors, prefix, config, dummy_experts,
                      expert_scale, generator)

    save_file(tensors, str(out_dir / "model.safetensors"))
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0},
                    "weight_map": {k: "model.safetensors" for k in tensors}}),
        encoding="utf-8")
    return {
        "source": str(model_path), "real_layers": real_layers,
        "depth": depth, "dummy_layers": dummy_layers,
        "donor_layer": donor, "dummy_experts": dummy_experts,
        "config_surgery": surgery, "config_dropped": dropped,
        "tensors": len(tensors),
    }


def fabricate_moe(tensors: dict, prefix: str, config: dict,
                  n_experts: int, scale: float,
                  generator: torch.Generator) -> None:
    """DeepSeek/GLM-family MoE keys with inert int8 values."""
    hidden = int(config["hidden_size"])
    inter = int(config["moe_intermediate_size"])
    shared_inter = inter * int(config.get("n_shared_experts", 1))
    tensors[prefix + "mlp.gate.weight"] = (
        torch.randn(n_experts, hidden, generator=generator) * 0.02)
    tensors[prefix + "mlp.gate.e_score_correction_bias"] = torch.zeros(n_experts)

    def int8(out_features: int) -> tuple[torch.Tensor, torch.Tensor]:
        weight = torch.randint(-32, 32, (out_features, hidden),
                               generator=generator, dtype=torch.int8)
        weight_scale = torch.full((out_features,), scale)
        return weight, weight_scale

    for expert in range(n_experts):
        for name, out_f in (("gate_proj", inter), ("up_proj", inter),
                            ("down_proj", hidden)):
            weight, weight_scale = int8(out_f)
            tensors[prefix + f"mlp.experts.{expert}.{name}.weight"] = weight
            tensors[prefix + f"mlp.experts.{expert}.{name}.weight_scale"] = weight_scale
    for name, out_f in (("gate_proj", shared_inter), ("up_proj", shared_inter),
                        ("down_proj", hidden)):
        weight, weight_scale = int8(out_f)
        tensors[prefix + f"mlp.shared_experts.{name}.weight"] = weight
        tensors[prefix + f"mlp.shared_experts.{name}.weight_scale"] = weight_scale


def reach_model(llm) -> object:
    """The in-process model object, defensively; versions rename the path."""
    for path in ENGINE_MODEL_PATHS:
        obj = llm
        try:
            for attr in path:
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        return obj
    raise RuntimeError(
        "cannot reach the in-process model — VLLM_ENABLE_V1_MULTIPROCESSING=0 "
        "must be set before importing vllm, and the LLM must be V1 in-process"
    )


def capture_stages(model: object, real_layers: list[int]) -> dict:
    """Register forward hooks; return the filled capture dict (post-run).

    Protocol (recorded, not rediscovered): the decoder layer's forward is
    (positions, hidden_states, residual) — args[0] is POSITIONS, the hidden
    states are args[1] — and it returns (hidden, residual) under the
    deferred-residual protocol, so the effective output is hidden +
    residual. The 2026-09-14 session produced a false 112% layer error by
    getting exactly this wrong once.
    """
    captured: dict = {}
    layers = model.model.layers

    def tensor_of(value):
        if isinstance(value, tuple) and value and torch.is_tensor(value[0]):
            return value[0]
        return value if torch.is_tensor(value) else None

    for index in real_layers:
        layer = layers[index]
        target = f"layers.{index}"

        def layer_hook(module, args, output, target=target):
            captured[f"{target}.in"] = args[1].detach().float().cpu()
            hidden = tensor_of(output)
            residual = (output[1] if isinstance(output, tuple) and len(output) > 1
                        and torch.is_tensor(output[1]) else None)
            if hidden is not None:
                captured[f"{target}.out"] = hidden.detach().float().cpu()
            if residual is not None:
                captured[f"{target}.residual"] = residual.detach().float().cpu()
                # The effective output the next layer actually sees.
                captured[f"{target}.effective_out"] = (
                    hidden.detach().float().cpu() + residual.detach().float().cpu())

        layer.register_forward_hook(layer_hook)

        for stage, module_name in (
            ("post_input_ln", "input_layernorm"),
            ("post_attn", "self_attn"),
            ("post_attn_ln", "post_attention_layernorm"),
            ("post_mlp", "mlp"),
        ):
            module = getattr(layer, module_name, None)
            if module is None:
                captured[f"{target}.{stage}"] = None  # absent, recorded
                continue

            def stage_hook(module, args, output, key=f"{target}.{stage}"):
                value = tensor_of(output)
                if value is not None:
                    captured[key] = value.detach().float().cpu()

            module.register_forward_hook(stage_hook)
    return captured


def run_capture(swap_dir: Path, real_layers: list[int], out_dir: Path,
                tokens: list[int], dtype: str = "bfloat16",
                max_model_len: int = 64,
                gpu_memory_utilization: float = 0.3) -> dict:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from vllm import LLM, SamplingParams  # noqa: PLC0415 - in-process only

    out_dir.mkdir(parents=True, exist_ok=True)
    llm = LLM(model=str(swap_dir), tensor_parallel_size=1,
              max_model_len=max_model_len, dtype=dtype, enforce_eager=True,
              gpu_memory_utilization=gpu_memory_utilization,
              trust_remote_code=False)
    model = reach_model(llm)
    captured = capture_stages(model, real_layers)
    outputs = llm.generate([{"prompt_token_ids": tokens}],
                           SamplingParams(max_tokens=1, temperature=0.0))
    next_token = int(outputs[0].outputs[0].token_ids[0])

    torch.save(captured, out_dir / "capture.pt")
    manifest = {
        "swap_dir": str(swap_dir), "real_layers": real_layers,
        "tokens": tokens, "dtype": dtype, "next_token": next_token,
        "stages": {key: (list(value.shape) if torch.is_tensor(value) else None)
                   for key, value in captured.items()},
        "stage_norms": {key: (float(value.float().norm())
                              if torch.is_tensor(value) else None)
                        for key, value in captured.items()},
    }
    (out_dir / "capture_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def compare_captures(capture_dir: Path, reference_dir: Path,
                     tolerance: float = 0.05) -> dict:
    """relL2 per stage with protocol guards; the verdict, not just numbers.

    Guards: both legs must record the same tokens (comparing captures taken
    on different inputs is a silent-corruption failure, not a finding) and
    share at least one stage.
    """
    capture = torch.load(capture_dir / "capture.pt")
    reference = torch.load(reference_dir / "capture.pt")
    capture_manifest = json.loads(
        (capture_dir / "capture_manifest.json").read_text(encoding="utf-8"))
    reference_manifest = json.loads(
        (reference_dir / "capture_manifest.json").read_text(encoding="utf-8"))
    if capture_manifest.get("tokens") != reference_manifest.get("tokens"):
        return {
            "state": "INCOMPARABLE",
            "reason": "captures used different tokens: "
                      f"{capture_manifest.get('tokens')} vs "
                      f"{reference_manifest.get('tokens')}",
        }

    stages = []
    for key in sorted(set(capture) | set(reference)):
        got, want = capture.get(key), reference.get(key)
        if not (torch.is_tensor(got) and torch.is_tensor(want)):
            stages.append({"stage": key, "verdict": "SKIPPED",
                           "detail": "absent or non-tensor on one side"})
            continue
        if got.shape != want.shape:
            stages.append({"stage": key, "verdict": "INCOMPARABLE",
                           "detail": f"shape {list(got.shape)} vs "
                                     f"{list(want.shape)}"})
            continue
        rel = float((got.float() - want.float()).norm() / want.float().norm())
        stages.append({
            "stage": key, "rel_l2": rel,
            "norm_capture": float(got.float().norm()),
            "norm_reference": float(want.float().norm()),
            "verdict": "PASS" if rel <= tolerance else "FAIL",
        })
    compared = [s for s in stages if "rel_l2" in s]
    if not compared:
        return {"state": "INCOMPARABLE",
                "reason": "no shared tensor stages between the two legs",
                "stages": stages}
    worst = max(compared, key=lambda s: s["rel_l2"])
    return {
        "state": "PASS" if all(s["verdict"] == "PASS" for s in compared) else "FAIL",
        "tolerance": tolerance,
        "worst_stage": worst["stage"], "worst_rel_l2": worst["rel_l2"],
        "stages": stages,
        "capture": {"dir": str(capture_dir),
                    "next_token": capture_manifest.get("next_token")},
        "reference": {"dir": str(reference_dir),
                      "next_token": reference_manifest.get("next_token")},
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="layer-swap diagnosis: build / capture / compare")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="write the swap model")
    build.add_argument("--model-path", type=Path, required=True)
    build.add_argument("--real-layers", default="0",
                       help="comma-separated real layer indices")
    build.add_argument("--depth", type=int, default=None,
                       help="swap depth; default max(real)+1")
    build.add_argument("--out-dir", type=Path, required=True)
    build.add_argument("--dummy-experts", type=int, default=8)
    build.add_argument("--manifest", type=Path, default=None,
                       help="where to write build_manifest.json (default out-dir)")

    capture = sub.add_parser("capture", help="run in-process vLLM with hooks")
    capture.add_argument("--swap-dir", type=Path, required=True)
    capture.add_argument("--real-layers", default="0")
    capture.add_argument("--out", type=Path, required=True)
    capture.add_argument("--tokens", default="785,3974,13867,38627,34041,916",
                         help="fixed prompt token ids; both legs must match")
    capture.add_argument("--dtype", default="bfloat16")
    capture.add_argument("--max-model-len", type=int, default=64)
    capture.add_argument("--gpu-memory-utilization", type=float, default=0.3)

    compare = sub.add_parser("compare", help="two captures -> verdict report")
    compare.add_argument("--capture", type=Path, required=True)
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--tolerance", type=float, default=0.05)
    compare.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "build":
        real_layers = parse_layers(args.real_layers)
        depth = args.depth or (max(real_layers) + 1)
        manifest = build_swap(args.model_path, real_layers, depth,
                              args.out_dir, dummy_experts=args.dummy_experts)
        path = args.manifest or (args.out_dir / "build_manifest.json")
        Path(path).write_text(json.dumps(manifest, indent=2),
                              encoding="utf-8")
        print(json.dumps(manifest))
        return 0
    if args.command == "capture":
        manifest = run_capture(
            args.swap_dir, parse_layers(args.real_layers), args.out,
            [int(t) for t in args.tokens.split(",")], dtype=args.dtype,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization)
        print(json.dumps({"state": "CAPTURED", "stages": len(manifest["stages"]),
                          "next_token": manifest["next_token"]}))
        return 0
    report = compare_captures(args.capture, args.reference, args.tolerance)
    args.out.write_text(json.dumps(report, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps({"state": report["state"],
                      "worst_stage": report.get("worst_stage"),
                      "worst_rel_l2": report.get("worst_rel_l2")}))
    return 0 if report["state"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
