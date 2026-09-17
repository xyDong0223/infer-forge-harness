"""MAT-005 Resource & Deployment Plan: derive launch parameters, with sources.

Every parameter carries where it came from. That is the whole point: the launch
parameters were the one thing in this repository still being typed by hand, and a
typed parameter is indistinguishable from a measured one once it lands in a
contract.

Derivations use facts already established upstream — the checkpoint's own dtype
and position limit from MAT-001, weight bytes from the fingerprint, device memory
from the catalog, and the patch's limitations from MAT-007 (a shape-dynamic
fallback forces eager execution). Where no fact supports a value, the parameter is
marked `assumption` so a reader can tell the difference.
"""

from __future__ import annotations

import json
import hashlib
import math
import shlex
import sys
from pathlib import Path

from core.paths import REPO_ROOT

from operations.deployment.memory_budget import load_device_spec
from core.storage import default_state_root, ensure_external, safe_component  # noqa: E402
from validators.plan_validator import validate_deployment_plan  # noqa: E402

CONTRACT = REPO_ROOT / "tasks" / "mat-005-deployment-plan" / "task.yaml"
# vLLM-Kunlun forces 64 for MLA and defaults to 16 otherwise
# (openwiki/vllm-kunlun/platform-contract.md).
BLOCK_SIZE = {"mla": 64, "default": 16}
# Weights plus runtime must leave room for a KV pool worth having.
WEIGHT_HEADROOM = 0.55


def value(name: str, val: object, source: str, assumption: bool = False) -> dict:
    return {"parameter": name, "value": val, "source": source, "assumption": assumption}


def kv_floor_mib(identity: dict, max_model_len: int) -> float:
    """Minimum KV-pool size one max-length sequence needs, per rank.

    The GLM-5.2 run (glm52-int-w8a8-p800-001, 2026-09-14) launched a 707 GiB
    model with the identity-derived 1M context and paid a full weight load to
    learn that the KV allocator needs at least max_model_len tokens of room:
    with 88.4 of 96 GiB per card in weights, the remainder could not hold it
    and the server died with zero cache blocks. This estimate runs before the
    load instead. It is deliberately conservative arithmetic, not a
    measurement: MLA stores (kv_lora_rank + qk_rope_head_dim) elements per
    token per layer (replicated across TP ranks), assumed 2 bytes in bf16
    caches; regular attention stores 2 x num_kv_heads x head_size. MEM-001
    replaces the estimate with the measured budget once the service is up.
    """
    layers = int(identity.get("num_hidden_layers") or 1)
    if identity.get("kv_lora_rank"):
        per_token = int(identity["kv_lora_rank"]) + int(
            identity.get("qk_rope_head_dim") or 0
        )
    else:
        per_token = 2 * int(identity.get("num_key_value_heads") or 1) * int(
            identity.get("head_dim") or identity.get("qk_nope_head_dim") or 128
        )
    return per_token * layers * max_model_len * 2 / (1 << 20)


def cap_model_len_to_budget(identity: dict, position_limit: int,
                             weight_mib: float, hbm_mib: int, tp: int,
                             utilization: float) -> tuple[int, str]:
    """Fit max_model_len into the post-weight remainder, or report why not.

    Returns (model_len, source). The halving search keeps the largest
    context that fits with margin for activations and the sampler; the
    identity limit is returned untouched when it already fits.
    """
    per_card_weights = weight_mib / tp
    budget = hbm_mib * utilization
    remainder = budget - per_card_weights
    if kv_floor_mib(identity, position_limit) <= remainder * 0.5:
        return position_limit, "mat-001 identity.max_position_embeddings"
    # 0.5: the KV floor is a minimum for one sequence; the pool needs room
    # beyond it, and activations and the sampler warmup share the remainder.
    model_len = position_limit
    while model_len > 1024 and kv_floor_mib(identity, model_len) > remainder * 0.5:
        model_len //= 2
    if model_len == position_limit:
        return position_limit, "mat-001 identity.max_position_embeddings"
    return model_len, (
        f"pre-flight budget cap: {per_card_weights:.0f} MiB/card of weights "
        f"against a {budget:.0f} MiB/card budget leaves {remainder:.0f} MiB for "
        f"KV and runtime, but one {position_limit}-token sequence needs "
        f"{kv_floor_mib(identity, position_limit):.0f} MiB (conservative "
        f"bf16 estimate); capped to {model_len}. MEM-001 measures the real "
        f"budget and the length can be revisited."
    )


def plan(request: dict, classification: dict, spec: dict, patch: dict | None) -> dict:
    identity = request.get("identity") or {}
    model = request.get("model") or {}
    hbm_mib = int(spec["hbm_mib"])
    weight_mib = int(model.get("total_weight_bytes", 0)) / (1 << 20)

    # Smallest TP whose per-card weight share leaves the headroom a KV pool needs,
    # then rounded up to a power of two that divides the attention heads.
    heads = int(identity.get("num_attention_heads") or 1)
    minimum = max(1, math.ceil(weight_mib / (hbm_mib * WEIGHT_HEADROOM)))
    candidates = [size for size in (1, 2, 4, 8) if size >= minimum and heads % size == 0]
    tensor_parallel = candidates[0] if candidates else 8

    attention = "mla" if identity.get("kv_lora_rank") else "default"
    position_limit = int(identity.get("max_position_embeddings") or 32768)
    model_len, model_len_source = cap_model_len_to_budget(
        identity, position_limit, weight_mib, hbm_mib, tensor_parallel, 0.9
    )
    parameters = [
        value("dtype", identity.get("torch_dtype"), "mat-001 identity.torch_dtype"),
        value(
            "tensor_parallel_size",
            tensor_parallel,
            f"{weight_mib:.0f} MiB of weights against {hbm_mib} MiB per card with "
            f"{WEIGHT_HEADROOM:.0%} left for KV and runtime; heads {heads} divisible",
        ),
        value("max_model_len", model_len, model_len_source),
        value("block_size", BLOCK_SIZE[attention],
              f"vLLM-Kunlun default for {attention} attention (openwiki/vllm-kunlun/platform-contract.md)"),
        value("trust_remote_code", bool(model.get("trust_remote_code_required")),
              "mat-001 model.trust_remote_code_required"),
        value("gpu_memory_utilization", 0.9,
              "no measurement supports a specific fraction yet; MEM-001 can refine it",
              assumption=True),
    ]

    limitations = (patch or {}).get("limitations") or []
    eager_reason = next((text for text in limitations if "eager" in text.lower()), None)
    parameters.append(
        value("enforce_eager", bool(eager_reason),
              f"mat-007 patch limitation: {eager_reason}" if eager_reason
              else "no placed patch requires eager execution")
    )

    return {
        "state": "PLAN_READY",
        "subject": model.get("id"),
        "revision": model.get("revision"),
        "hardware": (request.get("target") or {}).get("hardware"),
        "stack_commit": (request.get("target") or {}).get("vllm_kunlun_commit"),
        "classification": classification.get("classification"),
        "parameters": parameters,
        "device": {"id": "p800", "hbm_mib": hbm_mib},
        "assumptions": [item["parameter"] for item in parameters if item["assumption"]],
    }


def _cluster_profile() -> dict:
    """Namespace, queue, setup flags: deployment identity the plan does not derive.

    Launch parameters come from the model and device facts; where the service
    runs comes from the cluster profile. Mixing those sources silently is how
    a plan ends up targeting a namespace the adapter refuses.
    """
    import yaml

    profile = yaml.safe_load(
        (REPO_ROOT / "config" / "clusters" / "p800-cluster.yaml").read_text(encoding="utf-8")
    ) or {}
    return {
        "namespace": (profile.get("cluster") or {}).get("namespace", "pd-test"),
        "setup": (profile.get("runtime") or {}).get("common_setup", []),
    }


def render_instance(
    report: dict, request: dict, runtime_artifact_root: str | Path | None = None,
    user_id: str | None = None,
) -> str:
    from core.user_identity import resolve_user_id

    import yaml

    values = {item["parameter"]: item["value"] for item in report["parameters"]}
    model = request.get("model") or {}
    profile = _cluster_profile()
    user_id = resolve_user_id(user_id)
    subject = str(report["subject"])
    resource_subject = safe_component(subject.lower()).replace("_", "-")
    artifact_root = ensure_external(
        runtime_artifact_root if runtime_artifact_root is not None
        else default_state_root() / "runs" / safe_component(f"kdp-001-{subject.lower()}")
    )
    # Weight-proportional patience: 215 GiB measured ~45 min wall-to-health on
    # MiniMax, so give ~5 s/GiB with a floor. A timeout before that is a
    # finding, not a property of the model.
    weight_gib = int(model.get("total_weight_bytes", 0)) >> 30
    timeout = max(900, weight_gib * 5)
    serve = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--host", "0.0.0.0", "--port", "8356",
        "--model", str(model.get('source')),
    ]
    if values.get("trust_remote_code"):
        serve.append("--trust-remote-code")
    serve += [
        "--max-model-len", str(values['max_model_len']),
        "--tensor-parallel-size", str(values['tensor_parallel_size']),
        "--dtype", str(values['dtype']),
        "--served-model-name", subject,
        "--block-size", str(values['block_size']),
        "--gpu-memory-utilization", str(values['gpu_memory_utilization']),
    ]
    if values.get("enforce_eager"):
        serve.append("--enforce-eager")
    instance = {
        "api_version": "infer.kunlun/v1alpha1",
        "kind": "Task",
        "metadata": {
            "name": f"kdp-001-{resource_subject}",
            "task_type": "service_proof",
            "version": "0.1.0",
            "generated_by": "mat-005-deployment-plan",
        },
        "context": {
            "runtime": {"engine": "vllm", "backend": "kunlun", "plugin": "vllm-kunlun",
                        "revisions": {"plugin": report["stack_commit"]} if report.get("stack_commit") else {}},
            "model": {"name": report["subject"], "path": model.get("source"),
                      "pvc": model.get("pvc"), "revision": model.get("revision")},
            "target": {"hardware": report["hardware"],
                       "device_count": values["tensor_parallel_size"]},
            "server": {"host": "0.0.0.0", "port": 8356, "served_model_name": subject,
                       "tensor_parallel_size": values["tensor_parallel_size"],
                       **{k: values[k] for k in ("dtype", "max_model_len", "block_size",
                                                  "gpu_memory_utilization")}},
        },
        "plan_sources": {item["parameter"]: item["source"] for item in report["parameters"]},
        # The sections below are what makes the instance a runnable contract:
        # without them kdp-001b validates CONTRACT_INVALID before touching the
        # cluster, which is how the graph used to stop here.
        "actions": ["toy_bringup_before_load", "start_server", "poll_health", "run_chat_smoke", "verify_backend", "collect_artifacts"],
        "acceptance": {
            "pod_ready": True,
            "health_check": 200,
            "chat_completion": "non_empty",
            "expected_backend": "kunlun",
            "unexpected_fallback": False,
            "startup_timeout_seconds": timeout,
        },
        "execution": {
            **({"user_id": user_id} if user_id else {}),
            "mode": "execute",
            "namespace": profile["namespace"],
            "resource_name": f"{user_id}-kdp001-{resource_subject}" if user_id
                             else f"kdp001-{resource_subject}",
            "startup_timeout_seconds": timeout,
            "health_interval_seconds": 15,
            "health_successes_required": 3,
            "retain_on_failure": True,
            "server_log": f"/workspace/server_{resource_subject}.log",
            "commands": {
                "setup": profile["setup"],
                "serve": [shlex.join(serve)],
            },
        },
        "checks": {
            "health": {"path": "/health", "expected_status": 200},
            "chat": {
                "path": "/v1/chat/completions", "method": "POST",
                "expected_non_empty_text": True,
                "payload": {"model": subject,
                            "messages": [{"role": "user",
                                          "content": "Say hello in one short sentence."}],
                            "max_tokens": 32},
            },
            "backend": {"expected": "kunlun", "reject_unexpected_fallback": True},
        },
        "artifacts": {
            "directory": str(artifact_root),
            "collect": ["deployment_manifest", "task_contract", "pod_spec", "server_log",
                        "health_result", "chat_result", "reproduce_command"],
        },
        "exit_states": {
            "pass": "DEPLOYMENT_READY",
            "startup_failed": "SERVER_START_FAILED",
            "health_timeout": "READINESS_TIMEOUT",
            "api_failed": "API_SMOKE_FAILED",
            "fallback_detected": "UNEXPECTED_FALLBACK",
            "invalid": "CONTRACT_INVALID",
            "needs_human": "NEEDS_HUMAN",
        },
    }
    return yaml.safe_dump(instance, sort_keys=False, allow_unicode=True)


def execute(args) -> int:
    import yaml

    request = yaml.safe_load(Path(args.model_request).read_text(encoding="utf-8"))
    classification = json.loads(Path(args.classification).read_text(encoding="utf-8"))
    if classification.get("state") != "CLASSIFICATION_READY":
        print(f"NEEDS_HUMAN: classification is {classification.get('state')}", file=sys.stderr)
        return 1
    patch = json.loads(Path(args.placed_patch).read_text(encoding="utf-8")) if args.placed_patch else None

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = plan(request, classification, load_device_spec(args.device), patch)
    (out / "deployment_plan.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "kdp_instance.yaml").write_text(
        render_instance(report, request, runtime_artifact_root=args.runtime_artifact_root,
                        user_id=getattr(args, "user_id", None)),
        encoding="utf-8",
    )

    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    gate = validate_deployment_plan(report, contract, request)
    (out / "plan_status.json").write_text(
        json.dumps({"state": report["state"],
                    "instance_sha256": hashlib.sha256((out / "kdp_instance.yaml").read_bytes()).hexdigest(),
                    "validator": {"passed": not gate, "errors": gate}}, indent=2),
        encoding="utf-8",
    )
    if gate:
        print("CONTRACT_INVALID: " + "; ".join(gate), file=sys.stderr)
        return 1
    for item in report["parameters"]:
        mark = "assumption" if item["assumption"] else "derived"
        print(f"  {item['parameter']:24} {str(item['value']):12} [{mark}] {item['source'][:70]}")
    print(f"instance: {out}/kdp_instance.yaml")
    return 0
