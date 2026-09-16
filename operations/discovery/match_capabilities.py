"""MAT-003 Capability Match: what the model demands vs what the installation provides.

Consumes the ModelRequest from MAT-001, the ModelSupportCard from MAT-002 and the
pod from the environment proof. Produces a per-axis match with graded evidence.

It deliberately cannot conclude support. Qwen3-8B matches on every axis and still
dies in an attention kernel during decode warmup, so the deliverable is a set of
candidate requirements for MAT-004 to classify — not a verdict on whether the
model runs.
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
from validators.capability_validator import validate_capability_match  # noqa: E402

PROBE = REPO_ROOT / "tools" / "probe" / "capability_match_probe.py"
CONTRACT = REPO_ROOT / "tasks" / "mat-003-capability-match" / "task.yaml"


class MatchFailed(ToolFailed):
    """Task-specific name for the shared (state, reason) failure."""


def run_probe(adapter: KunlunP800Adapter, pod: str, model_path: str) -> dict:
    push = push_snippet(PROBE, '/tmp/mat003_probe.py')
    script = (
        f"{default_runtime().env_prefix()}; "
        f"{push} && "
        f"python3 /tmp/mat003_probe.py {shlex.quote(model_path)} 2>/dev/null"
    )
    result = adapter.exec(pod, script, timeout=600)
    for line in reversed(result.stdout.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise MatchFailed("NEEDS_HUMAN", f"probe produced no JSON: {(result.stdout + result.stderr)[-500:]}")


def render_card(match: dict, request: dict, support: dict, pod: str) -> str:
    model = request.get("model", {})
    archs = [entry["architecture"] for entry in support.get("results", [])]
    verdicts = {entry["architecture"]: entry["verdict"] for entry in support.get("results", [])}
    lines = [
        f"# Capability match — {model.get('id')}",
        "",
        f"- revision: `{model.get('revision')}`",
        f"- matched in: `{pod}` (installed runtime)",
        f"- architectures: {', '.join(f'{a} ({verdicts[a]})' for a in archs) or 'unknown'}",
        f"- overall: **{match.get('verdict')}**, runtime_verified: {match.get('runtime_verified')}",
        "",
        "| Axis | Required | Evidence | Verdict |",
        "| --- | --- | --- | --- |",
    ]
    for axis in match.get("axes", []):
        lines.append(
            f"| {axis['axis']} | {axis['required']} | {axis['evidence']} | {axis['verdict']} |"
        )
    lines += [
        "",
        "Evidence grades, strongest first: `EXERCISED` (the capability ran and its numbers",
        "were checked — MAT-008, not this Task), `REGISTRY` (a name in an importable registry),",
        "`MODULE` (an implementation file exists), `ABSENT`. A module on disk does not mean the",
        "operators inside it work on this hardware, and neither does a registry entry.",
        "",
        "## Limitations",
        "",
        "- This is a static match and cannot conclude runtime support.",
        "- Known counter-example: Qwen3-8B matches every axis on P800 and still fails in",
        "  `kunlun_ops.speculative_attention` during decode warmup. A MATCHED verdict means",
        "  \"no requirement is known to be missing\", not \"it will run\".",
        "- Axes reported UNKNOWN were not introspected; they are open questions for MAT-004,",
        "  not silent passes.",
    ]
    return "\n".join(lines) + "\n"


def device_constraints(model_path: str) -> dict:
    """Numerical constraints from catalog/xpu_specs.yaml for the target device.

    The GLM-5.2 run (glm52-int-w8a8-p800-001, 2026-09-14) routed an MLA-sparse
    model onto the engine's generic fp8 path because nothing in the match
    recorded that this device has no fp8 compute; the catalog entry now does,
    and the match carries it so MAT-004 and the recovery Brain can see it.
    """
    import yaml

    from operations.deployment.memory_budget import load_device_spec

    spec = load_device_spec("p800")
    numerical = spec.get("numerical") or {}
    if not numerical:
        return {}
    constraints = dict(numerical)
    constraints.pop("evidence", None)
    constraints["source"] = "catalog/xpu_specs.yaml (observed on cluster)"
    return constraints


def execute(args) -> int:
    import yaml

    request = yaml.safe_load(Path(args.model_request).read_text(encoding="utf-8"))
    support = json.loads(Path(args.support_card).read_text(encoding="utf-8"))
    if support.get("state") != "SCAN_READY":
        raise MatchFailed(
            "NEEDS_HUMAN",
            f"the model scan is {support.get('state')}: matching capabilities before knowing which "
            "implementation runs would compare against the wrong code path",
        )

    pod = args.pod
    if not pod and args.env_status:
        status = json.loads(Path(args.env_status).read_text(encoding="utf-8"))
        if status.get("state") != "ENVIRONMENT_READY":
            raise MatchFailed("NEEDS_HUMAN", f"the environment proof is {status.get('state')}")
        pod = status.get("pod")
    if not pod:
        raise MatchFailed("CONTRACT_INVALID", "no pod: pass --pod or --env-status")

    model_path = (request.get("model") or {}).get("source")
    if not model_path:
        raise MatchFailed("CONTRACT_INVALID", "the ModelRequest carries no model.source")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    adapter = KunlunP800Adapter()
    adapter.assert_owned(pod)
    match = run_probe(adapter, pod, model_path)
    match["matched_in"] = pod
    match["model"] = {"id": (request.get("model") or {}).get("id"),
                      "revision": (request.get("model") or {}).get("revision")}
    constraints = device_constraints(model_path)
    if constraints:
        match["device_constraints"] = constraints
    (out / "capability_match.json").write_text(json.dumps(match, indent=2), encoding="utf-8")
    (out / "capability_match_card.md").write_text(
        render_card(match, request, support, pod), encoding="utf-8"
    )

    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    gate = validate_capability_match(match, contract)
    (out / "match_status.json").write_text(
        json.dumps({"state": match.get("state"), "validator": {"passed": not gate, "errors": gate}}, indent=2),
        encoding="utf-8",
    )
    if gate:
        raise MatchFailed("CONTRACT_INVALID", "; ".join(gate))
    print(f"{match['verdict']} (runtime_verified={match['runtime_verified']})")
    for axis in match["axes"]:
        print(f"  {axis['axis']:20} {axis['evidence']:9} {axis['verdict']}")
    print(f"artifacts: {out}/capability_match_card.md")
    return 0
