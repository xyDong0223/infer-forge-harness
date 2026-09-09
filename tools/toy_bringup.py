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

import argparse
import base64
import json
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.kunlun_p800.adapter import KunlunP800Adapter  # noqa: E402
from validators.bringup_validator import validate_bringup_report  # noqa: E402

PROBE = REPO_ROOT / "tools" / "probe" / "toy_bringup_probe.py"
CONTRACT = REPO_ROOT / "tasks" / "mat-028-toy-bringup" / "task.yaml"


class BringupFailed(RuntimeError):
    def __init__(self, state: str, reason: str) -> None:
        super().__init__(f"{state}: {reason}")
        self.state = state
        self.reason = reason


def run_probe(adapter: KunlunP800Adapter, pod: str, model_path: str, layers: int, experts: int,
              tp_size: int, max_len: int, timeout: int) -> dict:
    payload = base64.b64encode(PROBE.read_bytes()).decode()
    script = (
        "export VIRTUAL_ENV=/opt/vllm_kunlun PATH=/opt/vllm_kunlun/bin:$PATH; "
        f"echo {payload} | base64 -d > /tmp/mat028_probe.py && "
        f"python3 /tmp/mat028_probe.py {shlex.quote(model_path)} {layers} {experts} "
        f"{tp_size} {max_len} 2>/dev/null"
    )
    result = adapter.exec(pod, script, timeout=timeout)
    for line in reversed(result.stdout.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
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


def main() -> int:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-request", required=True, help="model_request.yaml from MAT-001")
    parser.add_argument("--pod", help="pod from the environment proof; overrides --env-status")
    parser.add_argument("--env-status", help="status.json of the environment proof")
    parser.add_argument("--layers", type=int, default=4, help="toy depth floor")
    parser.add_argument("--experts", type=int, default=8, help="toy routed expert floor")
    parser.add_argument("--tp-size", type=int, default=1,
                        help="1 keeps it cheap; the planned TP also covers collectives")
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--probe-timeout", type=int, default=1800)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

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
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BringupFailed as failure:
        print(f"{failure.state}: {failure.reason}", file=sys.stderr)
        raise SystemExit(1) from failure
