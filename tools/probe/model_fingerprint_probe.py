"""In-pod model fingerprint probe. Runs inside a container, prints JSON to stdout.

Kept dependency-free (stdlib only) because it must run in the serving image
without installing anything, and read-only because it runs against a shared PVC.

The digest deliberately does not hash whole weight files: the MiniMax checkpoint
is 214 GiB across 125 shards, so a full hash costs tens of minutes of shared-PVC
IO. Structure files are hashed in full, weight files by size plus the first and
last 1 MiB — enough to catch a re-quantised or truncated shard, not enough to
catch a deliberate mid-file forgery, which is not the threat model here.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

SAMPLE_BYTES = 1 << 20
STRUCTURE_NAMES = (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "chat_template.jinja",
)
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".gguf")
# Megatron/mcore training checkpoints share model names with servable weights on
# this PVC; they must be rejected rather than fingerprinted.
MCORE_MARKERS = ("latest_checkpointed_iteration.txt", "mp_rank_00", "iter_0000001")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_file(path: str, size: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        digest.update(handle.read(SAMPLE_BYTES))
        if size > SAMPLE_BYTES:
            handle.seek(max(size - SAMPLE_BYTES, SAMPLE_BYTES))
            digest.update(handle.read(SAMPLE_BYTES))
    return digest.hexdigest()


def identity_from_config(config: dict) -> dict:
    return {
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        # transformers >= 2.56 renamed torch_dtype to dtype; recent checkpoints
        # (e.g. Qwen3.8) only carry the new key, so both must be read.
        "torch_dtype": config.get("torch_dtype") or config.get("dtype"),
        "max_position_embeddings": config.get("max_position_embeddings"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "num_attention_heads": config.get("num_attention_heads"),
        "num_key_value_heads": config.get("num_key_value_heads"),
        "head_dim": config.get("head_dim"),
        "vocab_size": config.get("vocab_size"),
        "rope_scaling": config.get("rope_scaling"),
        "quantization_config": config.get("quantization_config"),
        # auto_map means the checkpoint carries its own modelling code, so
        # --trust-remote-code is mandatory and the code itself is part of the
        # revision.
        "remote_code": sorted((config.get("auto_map") or {}).values()),
    }


def main(path: str) -> int:
    result: dict = {"path": path}
    if not os.path.isdir(path):
        result["state"] = "MODEL_UNAVAILABLE"
        result["reason"] = "path is not a directory"
        print(json.dumps(result))
        return 0

    entries = sorted(os.listdir(path))
    marker = next((name for name in entries if name in MCORE_MARKERS), None)
    if marker:
        result["state"] = "UNSUPPORTED_FORMAT"
        result["reason"] = f"looks like an mcore training checkpoint: found {marker}"
        print(json.dumps(result))
        return 0
    if "config.json" not in entries:
        result["state"] = "UNSUPPORTED_FORMAT"
        result["reason"] = "no config.json: not a servable HuggingFace checkpoint"
        result["entries_sample"] = entries[:20]
        print(json.dumps(result))
        return 0

    with open(os.path.join(path, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)

    structure = {}
    for name in entries:
        if name in STRUCTURE_NAMES or name.endswith(".index.json") or name.endswith(".py"):
            full = os.path.join(path, name)
            if os.path.isfile(full):
                structure[name] = sha256_file(full)

    weights = []
    total = 0
    for name in entries:
        if not name.endswith(WEIGHT_SUFFIXES):
            continue
        full = os.path.join(path, name)
        if not os.path.isfile(full):
            continue
        size = os.path.getsize(full)
        total += size
        weights.append({"name": name, "size": size, "sample_sha256": sample_file(full, size)})

    if not weights:
        result["state"] = "UNSUPPORTED_FORMAT"
        result["reason"] = "config.json present but no weight shard found"
        print(json.dumps(result))
        return 0

    canonical = json.dumps(
        {"structure": structure, "weights": weights, "sample_bytes": SAMPLE_BYTES},
        sort_keys=True,
        separators=(",", ":"),
    )
    result.update(
        state="INTAKE_READY",
        revision=hashlib.sha256(canonical.encode()).hexdigest(),
        revision_method=f"structure-sha256 + head/tail {SAMPLE_BYTES}B per shard",
        shard_count=len(weights),
        total_weight_bytes=total,
        structure=structure,
        weights=weights,
        identity=identity_from_config(config),
        trust_remote_code_required=bool(config.get("auto_map")),
    )
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({"state": "CONTRACT_INVALID", "reason": "usage: probe <model-path>"}))
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
