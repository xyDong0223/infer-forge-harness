"""In-pod probe: bring the architecture up at toy scale with dummy weights.

Everything between "the modules import" and "the server answers" is a contract:
abstract methods the engine now expects, a factory whose return shape changed, a
KV-cache tensor whose rank the layer no longer slices the same way. None of it
depends on the real weights, and all of it used to be discovered by loading them
-- 707 GiB per attempt for GLM-5.2, once per bug.

This derives a few-layer copy of the real config, points the engine at dummy
weights, and runs prefill and one decode step. Same code path, seconds instead of
minutes, and no disk read at all.

Emits one JSON object on the last stdout line, always, including on failure:

    {"stage": "...", "stages_passed": [...], "config": {...}, "error": {...}}
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import traceback

STAGES = ["CONFIG_DERIVED", "ENGINE_CONSTRUCTED", "PREFILL_OK", "DECODE_OK"]

# Copied rather than symlinked: the engine reads these through the tokenizer, which
# does not follow every path the way plain file IO does.
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
)


def derive_config(source: str, target: str, layers: int, experts: int) -> dict:
    """Write a shrunken config.json next to copied tokenizer files.

    Two rules do the real work. Keep every dimension the weights and kernels care
    about -- head dims, hidden size, lora ranks -- because shrinking those changes
    which kernel is selected and stops the run from being the same code path. And
    truncate any per-layer list to the new depth: a config that says 78 layer types
    for a 5-layer model fails in the config, long before anything interesting.
    """
    with open(os.path.join(source, "config.json"), "rb") as handle:
        config = json.load(handle)

    original_layers = int(config.get("num_hidden_layers") or layers)
    original_experts = config.get("n_routed_experts")
    # Keep at least one layer past first_k_dense_replace so both the dense prefix
    # and one MoE layer are exercised; the MoE layer is where most contracts live.
    dense_prefix = int(config.get("first_k_dense_replace") or 0)
    new_layers = min(original_layers, max(layers, dense_prefix + 2))

    config["num_hidden_layers"] = new_layers
    for key, value in list(config.items()):
        if isinstance(value, list) and len(value) == original_layers:
            config[key] = value[:new_layers]

    if config.get("n_routed_experts"):
        top_k = int(config.get("num_experts_per_tok") or 1)
        groups = int(config.get("n_group") or 1)
        # Enough experts to keep grouped top-k meaningful, few enough to stay cheap.
        config["n_routed_experts"] = max(experts, top_k * groups, groups)
    # MTP is a separate module with its own bring-up; excluding it keeps one
    # failure per report.
    if config.get("num_nextn_predict_layers"):
        config["num_nextn_predict_layers"] = 0

    os.makedirs(target, exist_ok=True)
    with open(os.path.join(target, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    for name in TOKENIZER_FILES:
        candidate = os.path.join(source, name)
        if os.path.exists(candidate):
            shutil.copy2(candidate, os.path.join(target, name))

    return {
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "num_hidden_layers": {"real": original_layers, "toy": new_layers},
        "n_routed_experts": {"real": original_experts, "toy": config.get("n_routed_experts")},
        "quantization": (config.get("quantization_config") or {}).get("quant_method"),
        "kept_dimensions": {
            key: config.get(key)
            for key in ("hidden_size", "num_attention_heads", "qk_nope_head_dim",
                        "qk_rope_head_dim", "v_head_dim", "kv_lora_rank", "q_lora_rank",
                        "index_topk", "index_head_dim", "index_n_heads")
            if config.get(key) is not None
        },
    }


def main(argv: list[str]) -> int:
    source = argv[1]
    layers = int(argv[2]) if len(argv) > 2 else 4
    experts = int(argv[3]) if len(argv) > 3 else 8
    tp_size = int(argv[4]) if len(argv) > 4 else 1
    max_len = int(argv[5]) if len(argv) > 5 else 256

    passed: list[str] = []
    report: dict = {"stage": None, "stages_passed": passed, "config": None, "error": None}
    workdir = tempfile.mkdtemp(prefix="toy_bringup_")
    try:
        report["config"] = derive_config(source, workdir, layers, experts)
        passed.append("CONFIG_DERIVED")

        from vllm import LLM, SamplingParams

        llm = LLM(
            model=workdir,
            load_format="dummy",
            tensor_parallel_size=tp_size,
            dtype="bfloat16",
            max_model_len=max_len,
            max_num_seqs=1,
            max_num_batched_tokens=max_len,
            block_size=64,
            gpu_memory_utilization=0.9,
            enforce_eager=True,
            trust_remote_code=True,
            enable_prefix_caching=False,
        )
        passed.append("ENGINE_CONSTRUCTED")

        prompt = "hello"
        llm.generate([prompt], SamplingParams(max_tokens=1, temperature=0.0))
        passed.append("PREFILL_OK")

        outputs = llm.generate([prompt], SamplingParams(max_tokens=8, temperature=0.0))
        token_ids = list(outputs[0].outputs[0].token_ids)
        report["decode"] = {"token_ids": token_ids, "count": len(token_ids)}
        if len(token_ids) < 2:
            raise RuntimeError(f"decode produced {len(token_ids)} tokens, expected more than one")
        passed.append("DECODE_OK")
    except BaseException as error:  # noqa: BLE001 - a probe reports, it does not judge
        frames = traceback.format_exc().strip().splitlines()
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error)[:600],
            "traceback_tail": frames[-12:],
        }
    finally:
        report["stage"] = passed[-1] if passed else "NOTHING_RAN"
        report["complete"] = passed == STAGES
        report["workdir"] = workdir
        print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
