"""Independent MAT-005 acceptance checks.

The parameters this Task produces used to be typed by hand, and a typed parameter
is indistinguishable from a measured one once it reaches a contract. So the rule
here is provenance: every parameter carries a source, anything not backed by an
upstream fact is flagged as an assumption rather than dressed up as a derivation,
and the values that can be checked against those facts are checked.
"""

from __future__ import annotations

from typing import Any

REQUIRED = {
    "dtype",
    "tensor_parallel_size",
    "max_model_len",
    "block_size",
    "trust_remote_code",
    "gpu_memory_utilization",
    "enforce_eager",
}
VALID_TP = {1, 2, 4, 8}


def validate_deployment_plan(
    report: dict[str, Any], contract: dict[str, Any], request: dict[str, Any] | None = None
) -> list[str]:
    errors: list[str] = []
    acceptance = contract.get("acceptance") or {}
    checks = contract.get("checks") or {}

    if report.get("state") != "PLAN_READY":
        errors.append(f"state is {report.get('state')!r}, not PLAN_READY")

    parameters = {item["parameter"]: item for item in report.get("parameters") or []}
    missing = sorted(REQUIRED - set(parameters))
    if missing:
        errors.append(f"the plan does not decide {missing}")
    for name, item in parameters.items():
        if not item.get("source"):
            errors.append(f"{name} has no source: an unsourced parameter is a guess in disguise")
        if item.get("value") is None:
            errors.append(f"{name} has no value")
    declared = set(report.get("assumptions") or [])
    actual = {name for name, item in parameters.items() if item.get("assumption")}
    if declared != actual:
        errors.append("assumptions disagree with the per-parameter flags")

    if acceptance.get("require_stack_pin") and not report.get("stack_commit"):
        errors.append("stack_commit must be recorded: a plan is only valid for one stack")
    if acceptance.get("require_revision") and not report.get("revision"):
        errors.append("revision must be recorded: a plan is only valid for one checkpoint")

    identity = ((request or {}).get("identity")) or {}
    model = ((request or {}).get("model")) or {}
    if identity:
        dtype = parameters.get("dtype", {}).get("value")
        if dtype and identity.get("torch_dtype") and dtype != identity["torch_dtype"]:
            errors.append(
                f"dtype {dtype!r} contradicts the checkpoint's own {identity['torch_dtype']!r}"
            )
        limit = identity.get("max_position_embeddings")
        length = parameters.get("max_model_len", {}).get("value")
        if limit and length and int(length) > int(limit):
            errors.append(f"max_model_len {length} exceeds the checkpoint's limit {limit}")
        heads = identity.get("num_attention_heads")
        tensor_parallel = parameters.get("tensor_parallel_size", {}).get("value")
        if tensor_parallel not in VALID_TP:
            errors.append(f"tensor_parallel_size {tensor_parallel!r} is not one of {sorted(VALID_TP)}")
        elif heads and int(heads) % int(tensor_parallel):
            errors.append(f"tensor_parallel_size {tensor_parallel} does not divide {heads} heads")
    if model:
        trust = parameters.get("trust_remote_code", {}).get("value")
        if bool(model.get("trust_remote_code_required")) and not trust:
            errors.append(
                "the checkpoint carries its own modelling code, so trust_remote_code must be true"
            )

    if checks.get("memory_arithmetic_required"):
        device = report.get("device") or {}
        weight_bytes = int(model.get("total_weight_bytes") or 0)
        tensor_parallel = parameters.get("tensor_parallel_size", {}).get("value") or 1
        if device.get("hbm_mib") and weight_bytes:
            share = weight_bytes / (1 << 20) / int(tensor_parallel)
            if share > int(device["hbm_mib"]):
                errors.append(
                    f"{share:.0f} MiB of weights per card does not fit in {device['hbm_mib']} MiB "
                    f"at tensor_parallel_size {tensor_parallel}"
                )
    return errors
