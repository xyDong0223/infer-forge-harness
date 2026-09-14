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

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.memory_budget import load_device_spec  # noqa: E402
from validators.plan_validator import validate_deployment_plan  # noqa: E402

CONTRACT = REPO_ROOT / "tasks" / "mat-005-deployment-plan" / "task.yaml"
INSTANCE_TEMPLATE = REPO_ROOT / "tasks" / "kdp-001-deployment-proof" / "instances"
# vLLM-Kunlun forces 64 for MLA and defaults to 16 otherwise
# (openwiki/vllm-kunlun/platform-contract.md).
BLOCK_SIZE = {"mla": 64, "default": 16}
# Weights plus runtime must leave room for a KV pool worth having.
WEIGHT_HEADROOM = 0.55


def value(name: str, val: object, source: str, assumption: bool = False) -> dict:
    return {"parameter": name, "value": val, "source": source, "assumption": assumption}


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
    parameters = [
        value("dtype", identity.get("torch_dtype"), "mat-001 identity.torch_dtype"),
        value(
            "tensor_parallel_size",
            tensor_parallel,
            f"{weight_mib:.0f} MiB of weights against {hbm_mib} MiB per card with "
            f"{WEIGHT_HEADROOM:.0%} left for KV and runtime; heads {heads} divisible",
        ),
        value("max_model_len", position_limit, "mat-001 identity.max_position_embeddings"),
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


def render_instance(report: dict, request: dict) -> str:
    import os

    import yaml

    values = {item["parameter"]: item["value"] for item in report["parameters"]}
    model = request.get("model") or {}
    profile = _cluster_profile()
    user_id = os.environ.get("USER_ID", "").strip()
    subject = str(report["subject"])
    # Weight-proportional patience: 215 GiB measured ~45 min wall-to-health on
    # MiniMax, so give ~5 s/GiB with a floor. A timeout before that is a
    # finding, not a property of the model.
    weight_gib = int(model.get("total_weight_bytes", 0)) >> 30
    timeout = max(900, weight_gib * 5)
    serve = [
        "python -m vllm.entrypoints.openai.api_server",
        "--host 0.0.0.0",
        "--port 8356",
        f"--model {model.get('source')}",
    ]
    if values.get("trust_remote_code"):
        serve.append("--trust-remote-code")
    serve += [
        f"--max-model-len {values['max_model_len']}",
        f"--tensor-parallel-size {values['tensor_parallel_size']}",
        f"--dtype {values['dtype']}",
        f"--served-model-name {subject}",
        f"--block-size {values['block_size']}",
        f"--gpu-memory-utilization {values['gpu_memory_utilization']}",
    ]
    if values.get("enforce_eager"):
        serve.append("--enforce-eager")
    instance = {
        "api_version": "infer.kunlun/v1alpha1",
        "kind": "Task",
        "metadata": {
            "name": f"kdp-001-{subject.lower()}",
            "task_type": "service_proof",
            "version": "0.1.0",
            "generated_by": "mat-005-deployment-plan",
        },
        "context": {
            "model": {"name": report["subject"], "path": model.get("source"),
                      "pvc": model.get("pvc"), "revision": model.get("revision")},
            "target": {"hardware": report["hardware"],
                       "device_count": values["tensor_parallel_size"]},
            "server": {"host": "0.0.0.0", "port": 8356, "served_model_name": subject,
                       **{k: values[k] for k in ("dtype", "max_model_len", "block_size",
                                                  "gpu_memory_utilization")}},
        },
        "plan_sources": {item["parameter"]: item["source"] for item in report["parameters"]},
        # The sections below are what makes the instance a runnable contract:
        # without them kdp-001b validates CONTRACT_INVALID before touching the
        # cluster, which is how the graph used to stop here.
        "actions": ["start_server", "poll_health", "run_chat_smoke", "collect_artifacts"],
        "acceptance": {
            "pod_ready": True,
            "health_check": 200,
            "chat_completion": "non_empty",
            "expected_backend": "kunlun",
            "unexpected_fallback": False,
            "startup_timeout_seconds": timeout,
        },
        "execution": {
            "mode": "execute",
            "namespace": profile["namespace"],
            "resource_name": f"{user_id}-kdp001-{subject.lower()}" if user_id
                             else f"kdp001-{subject.lower()}",
            "startup_timeout_seconds": timeout,
            "health_interval_seconds": 15,
            "health_successes_required": 3,
            "retain_on_failure": True,
            "server_log": f"/workspace/server_{subject.lower()}.log",
            "commands": {
                "setup": profile["setup"],
                "serve": [" ".join(serve)],
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
            "directory": str(Path(REPO_ROOT) / "artifacts" / f"kdp-001-{subject.lower()}"),
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


def main() -> int:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-request", required=True)
    parser.add_argument("--classification", required=True, help="mat-004 gap_classification.json")
    parser.add_argument("--placed-patch", help="mat-007 placement report, if one exists")
    parser.add_argument("--device", default="p800")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

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
    (out / "kdp_instance.yaml").write_text(render_instance(report, request), encoding="utf-8")

    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    gate = validate_deployment_plan(report, contract, request)
    (out / "plan_status.json").write_text(
        json.dumps({"state": report["state"], "validator": {"passed": not gate, "errors": gate}}, indent=2),
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


if __name__ == "__main__":
    raise SystemExit(main())
