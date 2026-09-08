"""MAT-009 API Conformance: the response shape, not the numbers.

A different failure mode from every other Task here. The model can compute correctly,
the text can be right, and the API response can still be wrong: `reasoning_content`
empty, `tool_calls` never parsed, `finish_reason` plain. No differential over logits
sees that, which is why this is its own family rather than another MAT-008 dimension.

Parsers are pure text transforms, so this needs a pod but not a running server. That
is deliberate: it can run before a deployment is configured with `--reasoning-parser`
and still say whether the pair the deployment is about to use actually works.

Samples live in the contract, not here. Adding a model means writing down its parser
pair and a sample of its output, which is reviewable — and the probe refuses the
sample unless the marker it uses appears in that model's own chat template.
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
from validators.conformance_validator import validate_conformance  # noqa: E402

CONTRACT = REPO_ROOT / "tasks" / "mat-009-api-conformance" / "task.yaml"
PROBE = REPO_ROOT / "tools" / "probe" / "parser_conformance_probe.py"


class ConformanceFailed(RuntimeError):
    def __init__(self, state: str, reason: str) -> None:
        super().__init__(f"{state}: {reason}")
        self.state = state
        self.reason = reason


def profile_for(contract: dict, model_id: str) -> dict:
    """Longest declared family prefix wins, so Qwen3-30B-A3B resolves to Qwen3."""
    families = (contract.get("checks") or {}).get("models") or {}
    matches = [name for name in families if model_id.startswith(name)]
    if not matches:
        raise ConformanceFailed(
            "NEEDS_HUMAN",
            f"the contract declares no parser profile for {model_id!r}; add one rather than "
            "letting the probe guess a marker the model may never emit",
        )
    return families[max(matches, key=len)]


def run_probe(adapter: KunlunP800Adapter, pod: str, argv: list[str]) -> dict:
    payload = base64.b64encode(PROBE.read_bytes()).decode()
    script = (
        "export VIRTUAL_ENV=/opt/vllm_kunlun PATH=/opt/vllm_kunlun/bin:$PATH; "
        f"echo {payload} | base64 -d > /tmp/mat009_probe.py && "
        f"python3 /tmp/mat009_probe.py {' '.join(shlex.quote(part) for part in argv)} 2>/dev/null"
    )
    result = adapter.exec(pod, script, timeout=1800)
    for line in reversed(result.stdout.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise ConformanceFailed(
        "NEEDS_HUMAN", f"probe produced no JSON: {(result.stdout + result.stderr)[-500:]}"
    )


def probe_argv(profile: dict, model_path: str) -> list[str]:
    argv = ["--model-path", model_path,
            "--reasoning-sample", profile["reasoning_sample"],
            "--reasoning-moved", profile["reasoning_moved"],
            "--reasoning-control", profile["reasoning_control"]]
    for flag, key in (("--reasoning-parser", "reasoning_parser"),
                      ("--tool-parser", "tool_parser"),
                      ("--tool-sample", "tool_sample"),
                      ("--tool-control", "tool_control"),
                      ("--expected-tool-name", "expected_tool_name")):
        if profile.get(key):
            argv += [flag, profile[key]]
    for marker in profile.get("markers", []):
        argv += ["--marker", marker]
    return argv


def render_card(report: dict) -> str:
    lines = [
        f"# API conformance — {report.get('subject')}",
        "",
        f"- checked in: `{report.get('checked_in')}`",
        f"- weights: `{report.get('model_path')}`",
        f"- state: **{report.get('state')}**",
        "",
        "| Check | Parser | Implementation | Control discriminates |",
        "| --- | --- | --- | --- |",
    ]
    for case in report.get("cases", []):
        lines.append(
            f"| {case['case']} | `{case['parser']}` | `{case['implementation']}` | "
            f"{case['control']['discriminates']} |"
        )
    registries = report.get("registries") or {}
    lines += [
        "",
        f"- reasoning parsers registered: {len(registries.get('reasoning', []))}",
        f"- tool parsers registered: {len(registries.get('tool', []))}",
        "- Kunlun out-of-tree parser registries: "
        f"reasoning {registries.get('kunlun_oot_reasoning')}, "
        f"tool {registries.get('kunlun_oot_tool')} — empty means upstream's parsers are "
        "the ones that run here.",
        "",
        "## What this does and does not say",
        "",
        "- It says the parser pair splits this model's output format correctly, checked",
        "  against a control that moves the marker.",
        "- It does not say a *served* response carries the fields: that needs a",
        "  deployment launched with the parser and is a separate check.",
    ]
    for case in report.get("cases", []):
        trap = (case.get("unmarked_output") or {}).get("note")
        if trap:
            lines.append(f"- {case['parser']}: {trap}")
    return "\n".join(lines) + "\n"


def main() -> int:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True, help="model id, e.g. Qwen3-30B-A3B")
    parser.add_argument("--model-path", help="weights; defaults to the ModelRequest's source")
    parser.add_argument("--model-request", help="model_request.yaml from MAT-001")
    parser.add_argument("--pod", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    profile = profile_for(contract, args.subject)

    model_path = args.model_path
    if not model_path and args.model_request:
        request = yaml.safe_load(Path(args.model_request).read_text(encoding="utf-8"))
        model_path = (request.get("model") or {}).get("source")
    if not model_path:
        raise ConformanceFailed("CONTRACT_INVALID", "no model path: pass --model-path or "
                                                    "--model-request")

    args.out.mkdir(parents=True, exist_ok=True)
    adapter = KunlunP800Adapter()
    adapter.assert_owned(args.pod)
    report = run_probe(adapter, args.pod, probe_argv(profile, model_path))
    report["subject"] = args.subject
    report["checked_in"] = args.pod
    report["profile_source"] = str(CONTRACT.relative_to(REPO_ROOT))

    (args.out / "api_conformance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out / "api_conformance_card.md").write_text(render_card(report), encoding="utf-8")
    gate = validate_conformance(report, contract)
    (args.out / "conformance_status.json").write_text(
        json.dumps({"state": report.get("state"),
                    "validator": {"passed": not gate, "errors": gate}}, indent=2),
        encoding="utf-8",
    )
    if gate:
        raise ConformanceFailed("CONTRACT_INVALID", "; ".join(gate))
    print(f"{args.subject}: {report['state']}")
    for case in report.get("cases", []):
        print(f"  {case['case']:42} parser={case['parser']} "
              f"control={case['control']['discriminates']}")
    return 0 if report["state"] == "CONFORMANT" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConformanceFailed as failure:
        print(f"{failure.state}: {failure.reason}", file=sys.stderr)
        raise SystemExit(1) from failure
