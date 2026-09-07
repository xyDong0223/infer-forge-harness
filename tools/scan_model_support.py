"""MAT-002 Model Scan: what will actually run this architecture, in this pod.

Scans the installed runtime rather than a checkout, because the four override
mechanisms in vLLM-Kunlun (module redirection, post-import patching, OOT
registration, install-time overwrite) mean a repository and an installation can
disagree about what is registered.

Consumes the ModelRequest from MAT-001 and the pod from the environment proof, so
it needs no pod of its own and no launch parameter.
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
from validators.scan_validator import validate_support_card  # noqa: E402

PROBE = REPO_ROOT / "tools" / "probe" / "model_support_probe.py"
CONTRACT = REPO_ROOT / "tasks" / "mat-002-model-scan" / "task.yaml"
PROXY = "http://agent.baidu.com:8891"


class ScanFailed(RuntimeError):
    def __init__(self, state: str, reason: str) -> None:
        super().__init__(f"{state}: {reason}")
        self.state = state
        self.reason = reason


def run_probe(adapter: KunlunP800Adapter, pod: str, archs: list[str], proxy: str) -> dict:
    payload = base64.b64encode(PROBE.read_bytes()).decode()
    script = (
        "export VIRTUAL_ENV=/opt/vllm_kunlun PATH=/opt/vllm_kunlun/bin:$PATH "
        f"https_proxy={shlex.quote(proxy)} http_proxy={shlex.quote(proxy)} "
        "no_proxy=localhost,127.0.0.1,.baidu-int.com; "
        f"echo {payload} | base64 -d > /tmp/mat002_probe.py && "
        f"python3 /tmp/mat002_probe.py {' '.join(shlex.quote(a) for a in archs)} 2>/dev/null"
    )
    result = adapter.exec(pod, script, timeout=900)
    for line in reversed(result.stdout.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise ScanFailed("NEEDS_HUMAN", f"probe produced no JSON: {(result.stdout + result.stderr)[-500:]}")


def render_card(scan: dict, request: dict, pod: str) -> str:
    model = request.get("model", {})
    lines = [
        f"# Model support card — {model.get('id')}",
        "",
        f"- revision: `{model.get('revision')}`",
        f"- scanned in: `{pod}` (installed runtime, not a checkout)",
        f"- installed vLLM: {scan['results'][0].get('vllm_version') if scan.get('results') else 'unknown'}",
        f"- stack commit: `{(request.get('target') or {}).get('vllm_kunlun_commit')}`",
        "",
        "| Architecture | Verdict | What it means |",
        "| --- | --- | --- |",
    ]
    for entry in scan.get("results", []):
        lines.append(
            f"| {entry['architecture']} | {entry['verdict']} | {entry.get('meaning', '')} |"
        )
    pending = [
        pr
        for entry in scan.get("results", [])
        for pr in (entry.get("pull_requests") or [])
    ]
    if pending:
        lines += ["", "## Open upstream pull requests", ""]
        lines += [f"- #{pr['number']} {pr['title']} — {pr['url']}" for pr in pending]
    lines += [
        "",
        "## Limitations",
        "",
        "- A registered architecture is not a runtime guarantee: it says which code path loads,",
        "  not that every operator on that path works on this hardware.",
        "- `UPSTREAM_GENERIC` is not a gap. A failure on that path belongs to the operator or",
        "  attention backend, and classifying it as missing networking sends the next Task to",
        "  write a model file that already exists.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-request", required=True, help="model_request.yaml from MAT-001")
    parser.add_argument("--pod", help="pod from the environment proof; overrides --env-status")
    parser.add_argument("--env-status", help="status.json of the environment proof")
    parser.add_argument("--proxy", default=PROXY, help="in-pod proxy for the upstream lookups")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    request = yaml.safe_load(Path(args.model_request).read_text(encoding="utf-8"))
    archs = (request.get("identity") or {}).get("architectures") or []
    if not archs:
        raise ScanFailed("CONTRACT_INVALID", "the ModelRequest carries no identity.architectures")

    pod = args.pod
    if not pod and args.env_status:
        status = json.loads(Path(args.env_status).read_text(encoding="utf-8"))
        if status.get("state") != "ENVIRONMENT_READY":
            raise ScanFailed(
                "NEEDS_HUMAN",
                f"the environment proof is {status.get('state')}: scan results from an unproven "
                "runtime would describe an environment nobody validated",
            )
        pod = status.get("pod")
    if not pod:
        raise ScanFailed("CONTRACT_INVALID", "no pod: pass --pod or --env-status")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    adapter = KunlunP800Adapter()
    adapter.assert_owned(pod)
    scan = run_probe(adapter, pod, archs, args.proxy)
    scan["scanned_in"] = pod
    scan["model"] = {"id": (request.get("model") or {}).get("id"),
                     "revision": (request.get("model") or {}).get("revision")}
    (out / "model_support.json").write_text(json.dumps(scan, indent=2), encoding="utf-8")
    (out / "model_support_card.md").write_text(render_card(scan, request, pod), encoding="utf-8")

    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    gate = validate_support_card(scan, contract)
    (out / "scan_status.json").write_text(
        json.dumps({"state": scan.get("state"), "validator": {"passed": not gate, "errors": gate}}, indent=2),
        encoding="utf-8",
    )
    if gate:
        raise ScanFailed("CONTRACT_INVALID", "; ".join(gate))
    for entry in scan["results"]:
        print(f"{entry['architecture']:34} {entry['verdict']}")
    print(f"artifacts: {out}/model_support_card.md")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ScanFailed as failure:
        print(f"{failure.state}: {failure.reason}", file=sys.stderr)
        raise SystemExit(1) from failure
