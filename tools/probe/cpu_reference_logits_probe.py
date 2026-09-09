"""In-pod CPU reference: next-token distributions from `transformers`.

Runs the same checkpoint on CPU so the comparison has an authority that does not
share the accelerator's kernels. Slow by nature — one forward pass per prompt, no
generation — which is why it compares distributions instead of generated text.

Accelerator visibility is cleared by the caller; this refuses to run if torch can
still see a device, because a reference that quietly used the accelerator would
agree with the candidate for the wrong reason.
"""

from __future__ import annotations

import json
import sys


def _visible_accelerator(torch) -> str | None:
    """Return the first visible accelerator backend, if any.

    CUDA visibility is insufficient on Kunlun: torch exposes the vendor device
    through ``torch.xpu``.  Keep this probe conservative so a reference run can
    never silently share the candidate's device.
    """
    checks = (
        ("cuda", "cuda"),
        ("xpu", "xpu"),
        ("mlu", "mlu"),
        ("npu", "npu"),
        ("mps", "mps"),
    )
    for backend, attribute in checks:
        module = getattr(torch, attribute, None)
        available = getattr(module, "is_available", None)
        try:
            if callable(available) and available():
                return backend
        except Exception:
            # A partially installed backend is not usable as a reference. It
            # is safer to report it as visible than to load the model anyway.
            return backend
    return None


def _dequantize_int8_weights(model, model_path: str) -> dict:
    """Apply per-channel scales that transformers silently discarded.

    transformers 5.5.x does not build quantized modules for the compressed-tensors
    ``int-quantized`` format on CPU: the int8 ``weight`` is loaded as-is and the
    paired ``weight_scale`` is dropped as an unexpected key, so the reference
    computes with raw int8 magnitudes and produces NaN after the first layer.
    The dequantization itself is a per-output-channel symmetric multiply —
    checkpoint metadata only, no accelerator involvement, so the reference stays
    independent.
    """
    import os

    import torch
    from safetensors import safe_open

    weight_map = {}
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as handle:
            weight_map = json.load(handle).get("weight_map", {})
    else:
        for name in os.listdir(model_path):
            if name.endswith(".safetensors"):
                weight_map[name] = name

    # Group scale keys by shard so each file is opened once.
    shards: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        if key.endswith("weight_scale"):
            shards.setdefault(shard, []).append(key)

    patched = 0
    missing_scale = []
    modules = dict(model.named_modules())
    for shard, keys in shards.items():
        with safe_open(os.path.join(model_path, shard), framework="pt", device="cpu") as handle:
            for key in keys:
                scale = handle.get_tensor(key)
                weight_key = key[: -len("weight_scale")] + "weight"
                module = modules.get(weight_key[: -len(".weight")])
                if module is None and ".language_model." in weight_key:
                    # transformers flattens the multimodal wrapper: checkpoint keys
                    # say model.language_model.layers.N, parameters say model.layers.N.
                    alt = weight_key.replace(".language_model.", ".")
                    module = modules.get(alt[: -len(".weight")])
                if module is None or not hasattr(module, "weight"):
                    missing_scale.append(key)
                    continue
                weight = module.weight.detach()
                if weight.shape != scale.shape:
                    # Per-channel scales broadcast over the input dim.
                    scale = scale.reshape(weight.shape[0], -1)
                with torch.no_grad():
                    module.weight.copy_(weight.float() * scale.float())
                patched += 1
    return {"patched": patched, "missing_scale": missing_scale[:5]}


def main(model_path: str, top_k: int, prompts: list[str]) -> int:
    if top_k < 1:
        print(json.dumps({"state": "CONTRACT_INVALID", "reason": "top_k must be >= 1"}))
        return 0
    if not prompts:
        print(json.dumps({"state": "CONTRACT_INVALID", "reason": "at least one prompt is required"}))
        return 0

    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as error:
        print(json.dumps({"state": "REFERENCE_UNAVAILABLE", "reason": str(error)}))
        return 0

    accelerator = _visible_accelerator(torch)
    if accelerator:
        print(
            json.dumps(
                {
                    "state": "REFERENCE_UNAVAILABLE",
                    "reason": f"{accelerator} is still visible; the reference must run on CPU",
                }
            )
        )
        return 0

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # No device_map: it would pull in `accelerate`, and the point is to stay on
        # plain CPU torch with as little machinery as possible between the weights
        # and the logits.
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.float32, trust_remote_code=True
        )
        model.eval()

        dequant = _dequantize_int8_weights(model, model_path)
        if dequant["patched"] == 0:
            print(
                json.dumps(
                    {
                        "state": "REFERENCE_FAILED",
                        "reason": "no weight_scale tensors applied; checkpoint may not be quantized",
                    }
                )
            )
            return 0

        results = {}
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits[0, -1].float()
            values, indices = torch.topk(torch.log_softmax(logits, dim=-1), k=top_k)
            results[prompt] = [
                {"token": tokenizer.decode([int(index)]), "logprob": float(value)}
                for value, index in zip(values, indices)
            ]
        print(
            json.dumps(
                {
                    "state": "REFERENCE_READY",
                    # float32 on purpose: the reference should not inherit the
                    # candidate's reduced precision.
                    "dtype": "float32",
                    "transformers_version": transformers.__version__,
                    "dequantized": dequant,
                    "results": results,
                }
            )
        )
    except Exception as error:
        print(json.dumps({"state": "REFERENCE_FAILED", "reason": f"{type(error).__name__}: {error}"}))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(json.dumps({"state": "CONTRACT_INVALID", "reason": "usage: probe <path> <top_k> <prompt>..."}))
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], int(sys.argv[2]), sys.argv[3:]))
