"""MAT-028 Toy Bring-up: exercise the code path before paying for the weights.

Between "the modules import" (MAT-027) and "the server answers" (KDP-001b) sits a
band of pure contract: abstract methods the engine now requires, a factory whose
return shape changed, a KV-cache tensor whose rank the layer slices wrongly, a MoE
mapping helper that moved to module scope. None of it depends on the real weights.

GLM-5.2 found five such defects, each one at the end of a full 707 GiB load. A
few-layer copy of the same config with dummy weights reaches all five in under a
minute, on the same kernels, because the dimensions that select kernels are kept and
only depth and expert count are shrunk.

Consumes the ModelRequest for the config to copy and the pod from the environment
proof. Produces a position -- which stage was reached -- not an opinion.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from core.paths import REPO_ROOT

from adapters import get_hardware, push_snippet  # noqa: E402
from runtimes import default_runtime  # noqa: E402
KunlunP800Adapter = get_hardware()
from core.errors import ToolFailed
from validators.bringup_validator import validate_bringup_report  # noqa: E402

PROBE = REPO_ROOT / "tools" / "probe" / "toy_bringup_probe.py"
CONTRACT = REPO_ROOT / "tasks" / "mat-028-toy-bringup" / "task.yaml"


class BringupFailed(ToolFailed):
    """Task-specific name for the shared (state, reason) failure."""


def run_probe(adapter: KunlunP800Adapter, pod: str, model_path: str, layers: int, experts: int,
              tp_size: int, max_len: int, timeout: int, *,
              runtime=None, setup: list[str] | None = None, workdir: str | None = None) -> dict:
    push = push_snippet(PROBE, '/tmp/mat028_probe.py')
    context = [f"cd {shlex.quote(workdir)}"] if workdir else []
    context += [(runtime or default_runtime()).env_prefix(), *(setup or [])]
    script = (
        " && ".join(context) + " && "
        f"{push} && "
        f"python3 /tmp/mat028_probe.py {shlex.quote(model_path)} {layers} {experts} "
        f"{tp_size} {max_len}"
    )
    result = adapter.exec(pod, script, timeout=timeout)
    for line in reversed(result.stdout.strip().splitlines()):
        if line.startswith("{"):
            try:
                report = json.loads(line)
            except ValueError as error:
                raise BringupFailed("NEEDS_HUMAN", "probe returned invalid JSON: " + line[-800:]) from error
            if not isinstance(report, dict) or result.returncode:
                raise BringupFailed("NEEDS_HUMAN", f"probe exited {result.returncode}: {result.stderr[-800:]}")
            return report
    raise BringupFailed(
        "NEEDS_HUMAN",
        "probe produced no JSON, which usually means the process died rather than raised: "
        f"{(result.stdout + result.stderr)[-800:]}",
    )


def render_card(report: dict, model_id: str, pod: str) -> str:
    config = report.get("config") or {}
    depth = config.get("num_hidden_layers") or {}
    experts = config.get("n_routed_experts") or {}
    stages = ["CONFIG_DERIVED", "ENGINE_CONSTRUCTED", "PREFILL_OK", "DECODE_OK"]
    passed = set(report.get("stages_passed") or [])
    lines = [
        f"# Toy bring-up — {model_id}",
        "",
        f"- brought up in: `{pod}` with dummy weights (no checkpoint read)",
        f"- architecture: `{(config.get('architectures') or ['?'])[0]}` "
        f"(`{config.get('model_type')}`)",
        f"- depth: {depth.get('toy')} of {depth.get('real')} layers; "
        f"experts: {experts.get('toy')} of {experts.get('real')}",
        f"- quantization kept: `{config.get('quantization')}`",
        f"- reached: **{report.get('stage')}**",
        "",
        "| Stage | Result |",
        "| --- | --- |",
    ]
    for stage in stages:
        lines.append(f"| {stage} | {'passed' if stage in passed else 'not reached'} |")

    if config.get("kept_dimensions"):
        lines += ["", "## Dimensions kept at real size", ""]
        lines += [f"- `{k}` = {v}" for k, v in sorted(config["kept_dimensions"].items())]

    error = report.get("error")
    if error:
        lines += [
            "",
            f"## What stopped it — {error['type']}",
            "",
            "```",
            error["message"],
            "",
            *error.get("traceback_tail", []),
            "```",
        ]

    lines += [
        "",
        "## What a pass here does and does not mean",
        "",
        "- Does mean: the architecture constructs, the KV cache binds, and prefill and decode",
        "  both run on the kernels the real model will use.",
        "- Does not mean the numbers are right. The weights are random, so this cannot detect a",
        "  wrong scale, a mis-split head dim, or a layer that should have been dense. That is",
        "  what MAT-008 and MAT-013 are for.",
        "- Does not cover MTP: num_nextn_predict_layers is zeroed so one report carries one",
        "  failure.",
        "- With tensor_parallel_size 1 it also does not cover collective paths — an all-reduce",
        "  or an expert-parallel dispatch can still be wrong. Rerun at the planned TP to cover",
        "  them, at the cost of the cards.",
    ]
    return "\n".join(lines) + "\n"


def execute(args) -> int:
    import yaml

    request = yaml.safe_load(Path(args.model_request).read_text(encoding="utf-8"))
    model = request.get("model") or {}
    model_path = model.get("source")
    if not model_path:
        raise BringupFailed("CONTRACT_INVALID", "the ModelRequest carries no model.source")

    pod = args.pod
    if not pod and args.env_status:
        status = json.loads(Path(args.env_status).read_text(encoding="utf-8"))
        if status.get("state") != "ENVIRONMENT_READY":
            raise BringupFailed(
                "NEEDS_HUMAN",
                f"the environment proof is {status.get('state')}: a bring-up in an unproven "
                "runtime cannot clear anything",
            )
        pod = status.get("pod")
    if not pod:
        raise BringupFailed("CONTRACT_INVALID", "no pod: pass --pod or --env-status")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    adapter = KunlunP800Adapter()
    adapter.assert_owned(pod)

    report = run_probe(adapter, pod, model_path, args.layers, args.experts, args.tp_size,
                       args.max_model_len, args.probe_timeout)
    report["brought_up_in"] = pod
    report["model"] = {"id": model.get("id"), "revision": model.get("revision"),
                       "source": model_path}
    report["state"] = "BRINGUP_PASS" if report.get("complete") else "BRINGUP_BLOCKED"

    (out / "toy_bringup.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "toy_bringup_card.md").write_text(
        render_card(report, model.get("id") or "unknown", pod), encoding="utf-8"
    )

    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    gate = validate_bringup_report(report, contract, (request.get("identity") or {}).get("config"))
    (out / "bringup_status.json").write_text(
        json.dumps(
            {"state": report["state"], "validator": {"passed": not gate, "errors": gate}}, indent=2
        ),
        encoding="utf-8",
    )
    if gate:
        raise BringupFailed("CONTRACT_INVALID", "; ".join(gate))

    print(f"{report['state']} reached={report['stage']}")
    if report.get("error"):
        print(f"  {report['error']['type']}: {report['error']['message'][:200]}")
    print(f"artifacts: {out}/toy_bringup_card.md")
    # A blocked bring-up must fail the node: exiting 0 let the graph continue
    # past BRINGUP_BLOCKED into the service proof, paying a full model load to
    # rediscover what the toy had already found.
    return 0 if report["state"] == "BRINGUP_PASS" else 1
