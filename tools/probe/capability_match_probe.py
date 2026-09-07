"""In-pod capability match probe. Prints JSON to stdout.

Compares what the model's configuration demands against what the *installed*
vLLM-Kunlun actually provides, and grades the evidence: a name in an importable
registry is stronger than a module that merely exists on disk, which is stronger
than a sentence in documentation.

This is a static match. It cannot conclude runtime support, and the known
counter-example is Qwen3-8B on P800: every axis below matches and the server
still dies in an attention kernel during decode warmup. The output is therefore
a list of candidate requirements, never a support statement.
"""

from __future__ import annotations

import json
import os
import sys

SITE = "vllm_kunlun"


def load_config(path: str) -> dict:
    with open(os.path.join(path, "config.json"), encoding="utf-8") as handle:
        return json.load(handle)


def required_capabilities(config: dict) -> dict:
    text = json.dumps(config)
    quant = (config.get("quantization_config") or {}).get("quant_method")
    experts = (
        config.get("num_experts")
        or config.get("n_routed_experts")
        or config.get("num_local_experts")
    )
    # MLA is identified by the latent rank, the way vLLM-Kunlun itself keys on
    # config attributes rather than on the architecture name.
    if config.get("kv_lora_rank"):
        attention = "mla"
    elif "linear_attn" in text or "mamba" in text or config.get("layer_types"):
        attention = "hybrid_linear"
    elif config.get("sliding_window"):
        attention = "sliding_window"
    elif config.get("num_key_value_heads") and config.get("num_attention_heads"):
        attention = (
            "gqa"
            if config["num_key_value_heads"] < config["num_attention_heads"]
            else "mha"
        )
    else:
        attention = "unknown"
    return {
        "attention_variant": attention,
        "quantization": quant,
        "moe_experts": experts,
        "multimodal": bool(config.get("vision_config") or config.get("audio_config")),
        "speculative_layers": config.get("num_nextn_predict_layers"),
    }


def module_dir() -> str:
    import vllm_kunlun

    return os.path.dirname(vllm_kunlun.__file__)


def provided_capabilities() -> dict:
    """What the installation provides, with the evidence that says so."""
    import vllm_kunlun  # noqa: F401 - activates the plugin

    root = module_dir()

    def exists(*parts: str) -> bool:
        return os.path.exists(os.path.join(root, *parts))

    try:
        from vllm_kunlun.quantization import QUANTIZATION_METHODS

        quant_methods = sorted(QUANTIZATION_METHODS)
        quant_evidence = "REGISTRY"
    except Exception as error:
        quant_methods, quant_evidence = [], f"UNKNOWN: {error}"

    attention = {
        "gqa": ("MODULE" if exists("v1", "attention", "backends", "kunlun_attn.py") else "ABSENT"),
        "mha": ("MODULE" if exists("v1", "attention", "backends", "kunlun_attn.py") else "ABSENT"),
        "sliding_window": (
            "MODULE" if exists("v1", "attention", "backends", "kunlun_attn.py") else "ABSENT"
        ),
        "mla": ("MODULE" if exists("v1", "attention", "backends", "mla") else "ABSENT"),
        "hybrid_linear": (
            "MODULE" if exists("v1", "attention", "backends", "gdn_attn.py") else "ABSENT"
        ),
    }
    return {
        "attention_backends": attention,
        "quantization_methods": quant_methods,
        "quantization_evidence": quant_evidence,
        "moe": "MODULE" if exists("quantization", "moe_wna16.py") else "ABSENT",
        "lora": "MODULE" if exists("lora", "punica_wrapper") else "ABSENT",
        "reasoning_parsers": "MODULE" if exists("reasoning", "__init__.py") else "ABSENT",
        "tool_parsers": "MODULE" if exists("tool_parsers", "__init__.py") else "ABSENT",
        # Exhaustively searched and absent upstream in this plugin; recorded so a
        # later Task does not rediscover it.
        "pd_disaggregation": "ABSENT",
    }


def match(required: dict, provided: dict) -> list[dict]:
    axes: list[dict] = []

    variant = required["attention_variant"]
    evidence = provided["attention_backends"].get(variant, "ABSENT")
    axes.append(
        {
            "axis": "attention",
            "required": variant,
            "evidence": evidence,
            "verdict": "PROVIDED_MODULE_ONLY" if evidence == "MODULE" else "NOT_PROVIDED",
        }
    )

    quant = required["quantization"]
    if not quant:
        axes.append({"axis": "quantization", "required": None, "evidence": "N/A",
                     "verdict": "NOT_REQUIRED"})
    else:
        known = quant in provided["quantization_methods"]
        axes.append(
            {
                "axis": "quantization",
                "required": quant,
                "evidence": provided["quantization_evidence"] if known else "ABSENT",
                "verdict": "PROVIDED" if known else "NOT_PROVIDED",
            }
        )

    experts = required["moe_experts"]
    axes.append(
        {
            "axis": "moe",
            "required": experts,
            "evidence": provided["moe"] if experts else "N/A",
            "verdict": "NOT_REQUIRED"
            if not experts
            else ("PROVIDED_MODULE_ONLY" if provided["moe"] == "MODULE" else "NOT_PROVIDED"),
        }
    )

    axes.append(
        {
            "axis": "multimodal",
            "required": required["multimodal"],
            "evidence": "N/A",
            # No multimodal surface is introspected here, so claiming either way
            # would be invention.
            "verdict": "NOT_REQUIRED" if not required["multimodal"] else "UNKNOWN",
        }
    )

    layers = required["speculative_layers"]
    axes.append(
        {
            "axis": "speculative_decode",
            "required": layers,
            "evidence": "N/A",
            "verdict": "NOT_REQUIRED" if not layers else "UNKNOWN",
        }
    )
    return axes


def main(model_path: str) -> int:
    try:
        config = load_config(model_path)
        required = required_capabilities(config)
        provided = provided_capabilities()
        axes = match(required, provided)
    except Exception as error:
        print(json.dumps({"state": "MATCH_FAILED", "reason": f"{type(error).__name__}: {error}"}))
        return 0

    blocking = [axis for axis in axes if axis["verdict"] == "NOT_PROVIDED"]
    unknown = [axis for axis in axes if axis["verdict"] == "UNKNOWN"]
    print(
        json.dumps(
            {
                "state": "MATCH_READY",
                # Stated in the payload so no consumer can read this as support.
                "runtime_verified": False,
                "model_path": model_path,
                "required": required,
                "provided": provided,
                "axes": axes,
                "verdict": "MISMATCH"
                if blocking
                else ("MATCHED_WITH_UNKNOWNS" if unknown else "MATCHED"),
                "blocking_axes": [axis["axis"] for axis in blocking],
                "unknown_axes": [axis["axis"] for axis in unknown],
            }
        )
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({"state": "CONTRACT_INVALID", "reason": "usage: probe <model-path>"}))
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
