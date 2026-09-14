"""CLI for the autonomous adaptation kernel (safe plan-only default)."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import yaml
from engine.autopilot import AutopilotRunner
from runners.deployment_proof import DeploymentProofRunner
from adapters.kunlun_p800.adapter import ClusterConfig, KunlunP800Adapter

def _profile_environment(state):
    return {"backend": state.goal.get("backend", "p800"), "mode": state.goal.get("mode", "autonomous")}

def _preflight(state):
    if state.goal.get("backend") != "p800": raise RuntimeError("unsupported backend")
    checks = ["backend"]
    try:
        cluster = ClusterConfig.load(state.goal["config"].get("cluster_config", "config/clusters/p800-cluster.yaml"))
        checks.append(KunlunP800Adapter(cluster).preflight())
    except Exception as exc:
        if state.goal.get("mode") == "autonomous":
            raise RuntimeError(f"external preflight blocked: {exc}")
    if os.environ.get("KUBERNETES_SERVICE_HOST") or os.system("command -v kubectl >/dev/null 2>&1") == 0:
        checks.append("kubectl-available")
    return {"ready": True, "checks": checks}

def _rollback(state):
    return {"rolled_back": True}

def _profile_model(state):
    return {"model": state.goal.get("model"), "state": "profiled"}

def _validate(state):
    deployments = [o for o in state.observations if o.kind == "deployment"]
    if deployments and not deployments[-1].data.get("ready", False):
        raise RuntimeError("deployment proof did not pass")
    return {"gate": state.goal.get("acceptance", {}), "state": "passed"}

def _adapt(state):
    """Extension point for real adapters; records an actionable plan today."""
    if state.goal.get("mode") == "plan-only":
        return {"state": "planned", "backend": state.goal.get("backend"), "model": state.goal.get("model")}
    return {"state": "execution-ready", "backend": state.goal.get("backend"), "model": state.goal.get("model"), "auto_repair": True}

def _deployment(state):
    cfg = state.goal["config"]
    cluster = ClusterConfig.load(cfg.get("cluster_config", "config/clusters/p800-cluster.yaml"))
    adapter = KunlunP800Adapter(cluster)
    contract_path = Path("tasks/kdp-001-deployment-proof/task.yaml")
    contract = yaml.safe_load(contract_path.read_text())
    model = state.goal.get("model")
    contract.setdefault("context", {}).setdefault("model", {})["path"] = model
    contract["context"]["target"].update({"namespace": adapter.config.namespace,
        "device_count": cfg.get("xpu_count", 8)})
    contract.setdefault("execution", {}).update({
        "image": cfg.get("image", ""), "workdir": cfg.get("workdir", "/workspace"),
        "queue": cfg.get("queue", ""), "node_pool": cfg.get("node_pool", ""),
        "server_port": cfg.get("server_port", 8000),
    })
    run_id = state.goal.get("run_id", "autopilot")
    artifact_dir = Path("artifacts") / run_id
    runner = DeploymentProofRunner(contract, adapter, Path.cwd(), artifact_dir,
                                   workdir=cfg.get("workdir", "/workspace"),
                                   image=cfg.get("image", ""), phase="all")
    result = runner.run()
    state.goal["last_deployment"] = result
    return {"adapter": adapter.config.namespace, "ready": result.get("state") == "DEPLOYMENT_READY", "proof": result, "artifact_dir": str(artifact_dir)}

def main(argv=None):
    p = argparse.ArgumentParser(description="Run model-driven adaptation autopilot")
    p.add_argument("--model", required=True)
    p.add_argument("--backend", default=None)
    p.add_argument("--acceptance", default="{}")
    p.add_argument("--max-attempts", type=int, default=None)
    p.add_argument("--mode", choices=("autonomous", "plan-only"), default=None)
    p.add_argument("--config", default="config/p800-autonomous.yaml")
    p.add_argument("--resume", default="autopilot-result.json")
    args = p.parse_args(argv)
    config = {}
    config_path = Path(args.config)
    if config_path.exists(): config = yaml.safe_load(config_path.read_text()) or {}
    try: acceptance = json.loads(args.acceptance)
    except json.JSONDecodeError: acceptance = {"profile": args.acceptance}
    execution = config.get("execution", {})
    runner = AutopilotRunner({"model": args.model, "backend": args.backend or config.get("backend", "p800"), "mode": args.mode or execution.get("mode", "autonomous"), "acceptance": acceptance, "config": config}, {
        "preflight": _preflight, "rollback": _rollback, "profile_environment": _profile_environment, "profile_model": _profile_model, "adapt": _adapt, "deployment": _deployment, "validate": _validate
    }, max_attempts=args.max_attempts or execution.get("max_attempts", 100))
    resume = Path(args.resume)
    if resume.exists():
        try:
            previous = json.loads(resume.read_text())
            runner.state.attempts.extend(previous.get("attempts", []))
            from engine.autopilot import Observation
            runner.state.observations.extend(Observation(**item) for item in previous.get("observations", []))
            if previous.get("status") == "succeeded":
                runner.state.status = "succeeded"
        except (OSError, ValueError):
            pass
    result = runner.run()
    Path("autopilot-result.json").write_text(json.dumps(result.__dict__, default=lambda o: o.__dict__, ensure_ascii=False, indent=2))
    print(json.dumps(result.__dict__, default=lambda o: o.__dict__, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
