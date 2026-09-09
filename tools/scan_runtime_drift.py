"""MAT-027 Runtime Drift Scan: does the plugin still load against this engine.

The cheapest gate in the graph, and the one whose absence cost the most. A model
scan answers "is this architecture registered"; it does not answer "does the
registered code import here". When the installed engine has moved past the version
the plugin targets, every drifted symbol is a separate ImportError that surfaces
only when something imports that module -- and for an attention backend that means
after the weights are loaded. GLM-5.2 paid for six of them one full 707 GiB load at
a time before anyone ran this check.

Imports every module of the plugin package in one pass, and for each failure indexes
the engine's own source to say where the missing symbol lives now. Needs the pod
(the plugin is only installed there) and nothing else: no launch parameter, no
weights, no XPU.
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
from validators.drift_validator import validate_drift_report  # noqa: E402

PROBE = REPO_ROOT / "tools" / "probe" / "runtime_drift_probe.py"
CONTRACT = REPO_ROOT / "tasks" / "mat-027-runtime-drift" / "task.yaml"


class DriftScanFailed(RuntimeError):
    def __init__(self, state: str, reason: str) -> None:
        super().__init__(f"{state}: {reason}")
        self.state = state
        self.reason = reason


def run_probe(adapter: KunlunP800Adapter, pod: str, plugin: str, engine: str, timeout: int) -> dict:
    payload = base64.b64encode(PROBE.read_bytes()).decode()
    script = (
        "export VIRTUAL_ENV=/opt/vllm_kunlun PATH=/opt/vllm_kunlun/bin:$PATH; "
        f"echo {payload} | base64 -d > /tmp/mat027_probe.py && "
        f"python3 /tmp/mat027_probe.py {shlex.quote(plugin)} {shlex.quote(engine)} 2>/dev/null"
    )
    result = adapter.exec(pod, script, timeout=timeout)
    for line in reversed(result.stdout.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise DriftScanFailed(
        "NEEDS_HUMAN", f"probe produced no JSON: {(result.stdout + result.stderr)[-500:]}"
    )


def render_card(report: dict, pod: str) -> str:
    engine, plugin = report.get("engine", {}), report.get("plugin", {})
    failures = report.get("failures") or []
    lines = [
        f"# Runtime drift — {plugin.get('name')} against {engine.get('name')}",
        "",
        f"- scanned in: `{pod}` (installed packages, not a checkout)",
        f"- engine: `{engine.get('name')}` {engine.get('version')} at `{engine.get('path')}`",
        f"- plugin: `{plugin.get('name')}` {plugin.get('version')}",
        f"- modules imported: {plugin.get('modules_scanned')}, failed: {len(failures)}",
        f"- verdict: **{report.get('state')}**",
        "",
    ]
    if not failures:
        lines += ["Every plugin module imports against this engine.", ""]
        return "\n".join(lines)

    lines += [
        "| Module | Error | Missing | Lives now in |",
        "| --- | --- | --- | --- |",
    ]
    for failure in failures:
        resolution = failure.get("resolution") or {}
        missing = resolution.get("symbol") or resolution.get("module") or ""
        candidates = ", ".join(f"`{c}`" for c in (resolution.get("candidates") or [])) or "—"
        lines.append(
            f"| `{failure['module']}` | {failure['error_type']} | `{missing}` | {candidates} |"
        )
    lines += [
        "",
        "## Why this runs before anything is deployed",
        "",
        "- A registry entry proves a name is mapped, not that the module behind it imports.",
        "- Each row here is a failure that would otherwise appear one model load at a time,",
        "  because nothing imports an attention backend until a model needs it.",
        "- A row with no candidate is not automatically a wall: the symbol may be genuinely",
        "  gone, in which case the behaviour has to be reimplemented rather than repointed.",
        "",
        "## Limitations",
        "",
        "- Import success is not correctness. A module that imports can still hold a signature",
        "  that no longer matches, an abstract method it never implements, or a tensor layout the",
        "  engine changed underneath it. Those need MAT-028's toy bring-up, not this scan.",
        "- Candidates come from name matching in the engine source: same name, different module.",
        "  A renamed symbol looks gone, and a coincidental name match looks resolvable.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod", help="pod from the environment proof; overrides --env-status")
    parser.add_argument("--env-status", help="status.json of the environment proof")
    parser.add_argument("--plugin", default="vllm_kunlun", help="installed plugin package")
    parser.add_argument("--engine", default="vllm", help="installed engine package")
    parser.add_argument("--probe-timeout", type=int, default=900)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    pod = args.pod
    if not pod and args.env_status:
        status = json.loads(Path(args.env_status).read_text(encoding="utf-8"))
        if status.get("state") != "ENVIRONMENT_READY":
            raise DriftScanFailed(
                "NEEDS_HUMAN",
                f"the environment proof is {status.get('state')}: a drift report from an unproven "
                "runtime describes an install nobody validated",
            )
        pod = status.get("pod")
    if not pod:
        raise DriftScanFailed("CONTRACT_INVALID", "no pod: pass --pod or --env-status")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    adapter = KunlunP800Adapter()
    adapter.assert_owned(pod)

    report = run_probe(adapter, pod, args.plugin, args.engine, args.probe_timeout)
    report["scanned_in"] = pod
    report["state"] = "DRIFT_FOUND" if report.get("failures") else "DRIFT_CLEAR"

    (out / "runtime_drift.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "runtime_drift_card.md").write_text(render_card(report, pod), encoding="utf-8")

    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    gate = validate_drift_report(report, contract)
    (out / "drift_status.json").write_text(
        json.dumps(
            {"state": report["state"], "validator": {"passed": not gate, "errors": gate}}, indent=2
        ),
        encoding="utf-8",
    )
    if gate:
        raise DriftScanFailed("CONTRACT_INVALID", "; ".join(gate))

    failures = report.get("failures") or []
    print(f"{report['state']} {len(failures)} of {report['plugin']['modules_scanned']} modules")
    for failure in failures:
        resolution = failure.get("resolution") or {}
        target = (resolution.get("candidates") or ["<gone>"])[0]
        missing = resolution.get("symbol") or resolution.get("module") or failure["error_type"]
        print(f"  {failure['module']:52} {missing} -> {target}")
    print(f"artifacts: {out}/runtime_drift_card.md")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DriftScanFailed as failure:
        print(f"{failure.state}: {failure.reason}", file=sys.stderr)
        raise SystemExit(1) from failure
