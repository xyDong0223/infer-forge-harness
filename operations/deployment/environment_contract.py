"""Generate the environment proof from the harness profile and task contract."""

from copy import deepcopy
import hashlib

import yaml

from core.paths import REPO_ROOT
from core.facade import resolve_adapters
from core.target import contract_target
from core.user_identity import resolve_user_id


def build_environment_contract(
    user_id: str | None = None, *, previous: dict | None = None,
    evidence_mode: str | None = None, health_interval_seconds: float | None = None,
    profile: dict | None = None,
) -> dict:
    """Only explicit operational overrides survive; target launch settings do not."""
    task_path = REPO_ROOT / "tasks/kdp-001a-environment-proof/task.yaml"
    profile_path = REPO_ROOT / "config/clusters/p800-cluster.yaml"
    task = yaml.safe_load(task_path.read_text())
    profile = profile if profile is not None else yaml.safe_load(profile_path.read_text())
    base = profile.get("validation", {}).get("base_model", {})
    if not base.get("required") or not base.get("path"):
        raise ValueError("environment base model is not configured in the cluster profile")
    deployment, cluster = profile["deployment"], profile["cluster"]
    previous = previous or {}
    execution = previous.get("execution", {})
    owner = resolve_user_id(user_id, execution.get("user_id"))
    mode = evidence_mode or previous.get("metadata", {}).get("evidence_mode", "real")
    if mode not in ("real", "simulation"):
        raise ValueError("evidence_mode must be real or simulation")
    interval = (health_interval_seconds if health_interval_seconds is not None
                else execution.get("health_interval_seconds", 10))
    if interval < 0:
        raise ValueError("health_interval_seconds must be nonnegative")
    contract = {
        "api_version": task["api_version"], "kind": "Task",
        "metadata": {
            "name": task["metadata"]["name"], "task_type": "environment_proof",
            "version": task["metadata"]["version"], "evidence_mode": mode,
            "generated_by": "harness.environment_contract",
            "sources": {str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in (task_path, profile_path)},
        },
        "context": {
            "model": {"name": base["name"], "path": base["path"], "pvc": deployment["model_pvc"]},
            "target": {"hardware": "Kunlunxin-3-P800", "device_count": deployment["xpu_count"],
                       "namespace": cluster["namespace"], "volcano_queue": deployment["queue"],
                       "dedicated_pool": deployment["node_pool"]},
            "runtime": {"engine": "vllm", "backend": "kunlun", "plugin": "vllm-kunlun"},
            "software": {"image": deployment["image"]},
            "server": {"host": "0.0.0.0", "port": 8356, **deepcopy(base)},
        },
        **{field: deepcopy(task[field]) for field in ("actions", "acceptance", "exit_states")},
        "execution": {
            "mode": "execute", "namespace": cluster["namespace"], "user_id": owner,
            "resource_name": f"{owner}-environment-base" if owner else "",
            "manifest": deployment["base_manifest"], "startup_timeout_seconds": 1800,
            "health_interval_seconds": interval,
            "health_successes_required": execution.get("health_successes_required", 3),
            "retain_on_failure": True,
            "commands": {"install": ["bash /workspace/install_vllm_kunlun.sh"],
                         "setup": deepcopy(profile.get("runtime", {}).get("common_setup", []))},
        },
        "artifacts": {"collect": list(dict.fromkeys([*task["artifacts"], "task_contract", "reproduce_command"]))},
        "checks": {
            "health": {"path": "/health", "expected_status": 200},
            "chat": {"path": "/v1/chat/completions", "method": "POST",
                     "expected_non_empty_text": True,
                     "payload": {"model": base["served_model_name"],
                                 "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
                                 "max_tokens": 16}},
            "backend": {"expected": "kunlun", "reject_unexpected_fallback": True},
        },
    }
    if execution.get("server_log"):
        contract["execution"]["server_log"] = execution["server_log"]
    runtime = resolve_adapters(contract_target(contract), require_supported=True).runtime
    contract["execution"]["commands"]["serve"] = [runtime.build_serve_command(contract["context"]["server"])]
    return contract
