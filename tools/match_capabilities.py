"""MAT-003 Capability Match: what the model demands vs what the installation provides.

Consumes the ModelRequest from MAT-001, the ModelSupportCard from MAT-002 and the
pod from the environment proof. Produces a per-axis match with graded evidence.

It deliberately cannot conclude support. Qwen3-8B matches on every axis and still
dies in an attention kernel during decode warmup, so the deliverable is a set of
candidate requirements for MAT-004 to classify — not a verdict on whether the
model runs.
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
from validators.capability_validator import validate_capability_match  # noqa: E402

PROBE = REPO_ROOT / "tools" / "probe" / "capability_match_probe.py"
CONTRACT = REPO_ROOT / "tasks" / "mat-003-capability-match" / "task.yaml"


class MatchFailed(RuntimeError):
    def __init__(self, state: str, reason: str) -> None:
        super().__init__(f"{state}: {reason}")
        self.state = state
        self.reason = reason


def run_probe(adapter: KunlunP800Adapter, pod: str, model_path: str) -> dict:
    payload = base64.b64encode(PROBE.read_bytes()).decode()
    script = (
        "export VIRTUAL_ENV=/opt/vllm_kunlun PATH=/opt/vllm_kunlun/bin:$PATH; "
        f"echo {payload} | base64 -d > /tmp/mat003_probe.py && "
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


def main() -> int:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-request", required=True, help="model_request.yaml from MAT-001")
    parser.add_argument("--support-card", required=True, help="model_support.json from MAT-002")
    parser.add_argument("--pod")
    parser.add_argument("--env-status", help="status.json of the environment proof")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

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


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MatchFailed as failure:
        print(f"{failure.state}: {failure.reason}", file=sys.stderr)
        raise SystemExit(1) from failure
