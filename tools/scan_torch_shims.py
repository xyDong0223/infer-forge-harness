"""MAT-029 Shim Handoff: no torch shim serves without an operator request.

A torch shim that stands in for a vendor kernel is a legitimate way to keep
bring-up moving -- and a silent way to ship un-optimised hot-path arithmetic
forever. During GLM-5.2's adaptation, three shims
(kv_spans_from_batches, kunlun_convert_req_index_to_global_index,
kunlun_concat_and_cache_mla) were written mid-loop, the service went green, and
not one of them became an operator request. The dispatch path (MAT-024) existed
the whole time; nothing connected the place shims are born to it.

This tool closes that path. The in-pod probe nets candidate shims out of the
installed plugin; every signal must be explained by the shim registry --
declared, dispatched, or waived with a reason -- and every non-waived entry is
immediately turned into a durable operator request through the same
operator_lifecycle dispatch MAT-024 uses, so the candidate-integration loop
(MAT-026) can pick it up later. HANDOFF_FOUND is a work list, not a stop.
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
from tools.operator_lifecycle import dispatch as dispatch_requests  # noqa: E402
from validators.shim_validator import validate_shim_handoff  # noqa: E402

PROBE = REPO_ROOT / "tools" / "probe" / "torch_shim_probe.py"
CONTRACT = REPO_ROOT / "tasks" / "mat-029-shim-handoff" / "task.yaml"


class ShimScanFailed(RuntimeError):
    def __init__(self, state: str, reason: str) -> None:
        super().__init__(f"{state}: {reason}")
        self.state = state
        self.reason = reason


def run_probe(adapter: KunlunP800Adapter, pod: str, plugin: str, timeout: int) -> dict:
    payload = base64.b64encode(PROBE.read_bytes()).decode()
    script = (
        "export VIRTUAL_ENV=/opt/vllm_kunlun PATH=/opt/vllm_kunlun/bin:$PATH; "
        f"echo {payload} | base64 -d > /tmp/mat029_probe.py && "
        f"python3 /tmp/mat029_probe.py {shlex.quote(plugin)} 2>/dev/null"
    )
    result = adapter.exec(pod, script, timeout=timeout)
    for line in reversed(result.stdout.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise ShimScanFailed(
        "NEEDS_HUMAN", f"probe produced no JSON: {(result.stdout + result.stderr)[-500:]}"
    )


def match_signals_to_entries(
    signals: list[dict], entries: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Pair every signal with the entry that declares it.

    A signal is unmapped unless some entry names the same symbol and, when the
    entry records a file, that file is the one the signal was found in. Returns
    ``(unmapped_signals, matched_symbols)``; entries no signal matched stay in
    the registry regardless, because the declaration is the contract and the
    probe is only a net.
    """
    matched_symbols: list[dict] = []
    unmapped: list[dict] = []
    for signal in signals:
        hit = None
        for entry in entries:
            if entry.get("name") != signal.get("symbol"):
                continue
            location = str(entry.get("location") or "")
            file_tail = location.split(":")[0].rsplit("/", 1)[-1].replace(".py", "")
            signal_tail = str(signal.get("file") or "").rsplit(".", 1)[-1]
            if not file_tail or file_tail == signal_tail:
                hit = entry
                break
        if hit is None:
            unmapped.append(signal)
        else:
            matched_symbols.append(signal)
    return unmapped, matched_symbols


def render_card(report: dict) -> str:
    entries = report.get("entries") or []
    unmapped = report.get("unmapped_signals") or []
    lines = [
        f"# Torch shim handoff — {report.get('plugin')}",
        "",
        f"- scanned in: `{report.get('scanned_in')}` (installed packages, not a checkout)",
        f"- files scanned: {report.get('files_scanned')}, signals: {len(report.get('signals') or [])},"
        f" registry entries: {len(entries)}",
        f"- verdict: **{report.get('state')}**",
        "",
    ]
    if not entries and not unmapped:
        lines += ["No shim signal and no declared shim. Nothing owes an operator request.", ""]
        return "\n".join(lines)
    if entries:
        lines += ["| Shim | Replaces | Called | Status | Request |", "| --- | --- | --- | --- | --- |"]
        for entry in entries:
            lines.append(
                f"| `{entry.get('name')}` | {entry.get('replaced_kernel')} "
                f"| {entry.get('call_frequency')} | {entry.get('status')} "
                f"| {entry.get('request_id') or entry.get('reason') or '—'} |"
            )
        lines.append("")
    if unmapped:
        lines += [
            "## Unexplained signals — each is a shim the registry does not know about",
            "",
            "| Kind | Symbol | Where | Excerpt |",
            "| --- | --- | --- | --- |",
        ]
        for signal in unmapped:
            lines.append(
                f"| {signal.get('kind')} | `{signal.get('symbol')}` "
                f"| `{signal.get('file')}:{signal.get('line')}` | {signal.get('excerpt') or '—'} |"
            )
        lines += [
            "",
            "Declare each of these in the registry (location, replaced kernel, call frequency,",
            "semantics basis), then rerun. Until then the shim runs un-optimised and untracked.",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod", help="pod from the environment proof; overrides --env-status")
    parser.add_argument("--env-status", help="status.json of the environment proof")
    parser.add_argument("--plugin", default="vllm_kunlun", help="installed plugin package")
    parser.add_argument(
        "--registry",
        help="agent-maintained shim registry JSON (entries: name, location, replaced_kernel, "
        "call_frequency, semantics_basis, status, reason)",
    )
    parser.add_argument("--subject", help="subject for operator requests (default: plugin name)")
    parser.add_argument("--probe-timeout", type=int, default=600)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    pod = args.pod
    if not pod and args.env_status:
        status = json.loads(Path(args.env_status).read_text(encoding="utf-8"))
        if status.get("state") != "ENVIRONMENT_READY":
            raise ShimScanFailed(
                "NEEDS_HUMAN",
                f"the environment proof is {status.get('state')}: a shim scan of an unproven "
                "runtime describes an install nobody validated",
            )
        pod = status.get("pod")
    if not pod:
        raise ShimScanFailed("CONTRACT_INVALID", "no pod: pass --pod or --env-status")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    adapter = KunlunP800Adapter()
    adapter.assert_owned(pod)

    probe_report = run_probe(adapter, pod, args.plugin, args.probe_timeout)

    entries: list[dict] = []
    if args.registry:
        loaded = json.loads(Path(args.registry).read_text(encoding="utf-8"))
        entries = list(loaded.get("entries", loaded)) if isinstance(loaded, dict) else list(loaded)

    unmapped, _matched = match_signals_to_entries(probe_report.get("signals") or [], entries)

    # Dispatch immediately: a non-waived entry without a request becomes one now,
    # so the handoff never depends on a later manual step to become durable.
    dispatchable = [entry for entry in entries if entry.get("status") != "WAIVED"
                    and not entry.get("request_id")]
    dispatch_record = None
    if dispatchable:
        gaps = {
            "gaps": [
                {
                    "name": entry["name"],
                    "class": "TORCH_SHIM",
                    "location": entry["location"],
                    "replaced_kernel": entry["replaced_kernel"],
                    "call_frequency": entry["call_frequency"],
                    "semantics_basis": entry["semantics_basis"],
                    "source": "torch_shim_registry",
                }
                for entry in dispatchable
            ]
        }
        gaps_path = out / "shim_gaps.json"
        gaps_path.write_text(json.dumps(gaps, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        dispatch_record = dispatch_requests(gaps_path, out, args.subject or args.plugin)
        # dispatch() returns the request ids in creation order, one per gap.
        for entry, request_id in zip(dispatchable, dispatch_record.get("requests") or []):
            entry["request_id"] = request_id
            entry["status"] = "DISPATCHED"
            entry["dispatched_this_run"] = True

    report = {
        "plugin": args.plugin,
        "scanned_in": pod,
        "plugin_path": probe_report.get("plugin_path"),
        "files_scanned": probe_report.get("files_scanned"),
        "files_unparsable": probe_report.get("files_unparsable"),
        "signals": probe_report.get("signals") or [],
        "entries": entries,
        "unmapped_signals": unmapped,
        "dispatch": dispatch_record,
    }
    clean = not unmapped and all(
        entry.get("status") in ("DISPATCHED", "WAIVED") for entry in entries
    )
    report["state"] = "HANDOFF_CLEAR" if clean else "HANDOFF_FOUND"

    (out / "torch_shim_registry.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out / "torch_shim_card.md").write_text(render_card(report), encoding="utf-8")

    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    gate = validate_shim_handoff(report, contract)
    (out / "shim_status.json").write_text(
        json.dumps(
            {"state": report["state"], "validator": {"passed": not gate, "errors": gate}},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    if gate:
        raise ShimScanFailed("CONTRACT_INVALID", "; ".join(gate))

    print(
        f"{report['state']} signals={len(report['signals'])} entries={len(entries)} "
        f"unmapped={len(unmapped)}"
    )
    for entry in entries:
        print(f"  {entry['name']:52} {entry.get('status')} {entry.get('request_id') or ''}")
    for signal in unmapped:
        print(f"  UNDECLARED {signal['symbol']:42} {signal['file']}:{signal['line']}")
    print(f"artifacts: {out}/torch_shim_card.md")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ShimScanFailed as failure:
        print(f"{failure.state}: {failure.reason}", file=sys.stderr)
        raise SystemExit(1) from failure
