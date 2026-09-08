"""MAT-008 Capability Evaluation: run the capability, do not read about it.

MAT-003 grades static evidence — `REGISTRY` beats `MODULE` beats `ABSENT` — and
Qwen3-8B is the standing proof that the top of that scale is not support: every
axis matched and the server died in a decode kernel. This Task adds the tier above
`REGISTRY`: `EXERCISED`, meaning the capability ran on this hardware and its
numbers were compared against a reference that does not share the accelerator.

One contract covers four dimensions rather than four near-identical contracts,
because the method is the same in each: take the smallest unit that still carries
the real convention risk, run the operators the serving path uses, compare against
a float32 reference, and prove the comparison could have failed.

`--list-dimensions` is what the graph fans out over. It reads the CapabilityMatch
and names only the dimensions this model actually demands, so a dense bf16 model
does not get a quantization verdict it never needed.
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
from validators.evaluation_validator import validate_evaluation  # noqa: E402

CONTRACT = REPO_ROOT / "tasks" / "mat-008-capability-evaluation" / "task.yaml"

# Which axis of the static match makes a dimension worth exercising. Sliding window
# is deliberately keyed on the *required* value rather than an axis name: it is a
# parameter of the existing attention kernels (`swa_left`/`swa_right` for prefill,
# `max_window_size` for decode), not a separate backend.
DIMENSION_FROM_AXIS: dict[str, str] = {
    "quantization": "quantization",
    "moe": "moe",
    "multimodal": "multimodal",
}

PROBES: dict[str, str] = {
    "quantization": "tools/probe/quantized_linear_probe.py",
    "msa": "tools/probe/sliding_window_decode_probe.py",
    "moe": "tools/probe/moe_layer_probe.py",
}

# Files a probe needs next to it in the pod. The msa probe grades the patch we
# actually serve with, so it has to be the same file, not a copy that can drift.
SIDECARS: dict[str, dict[str, str]] = {
    "msa": {"--fallback": "patches/torch_paged_decode.py"},
}


class EvaluationFailed(RuntimeError):
    def __init__(self, state: str, reason: str) -> None:
        super().__init__(f"{state}: {reason}")
        self.state = state
        self.reason = reason


def dimensions_for(match: dict) -> list[str]:
    selected: list[str] = []
    for axis in match.get("axes", []):
        if axis.get("verdict") == "NOT_REQUIRED":
            continue
        name = DIMENSION_FROM_AXIS.get(axis.get("axis"))
        if name and name not in selected:
            selected.append(name)
        if axis.get("axis") == "attention" and axis.get("required") == "sliding_window":
            selected.append("msa")
    return selected


def thresholds_for(contract: dict, dimension: str) -> dict:
    table = (contract.get("checks") or {}).get("dimensions") or {}
    if dimension not in table:
        raise EvaluationFailed(
            "CONTRACT_INVALID",
            f"the contract declares no thresholds for dimension {dimension!r}",
        )
    return table[dimension]


def run_probe(adapter: KunlunP800Adapter, pod: str, probe: Path, argv: list[str],
              sidecars: dict[str, str] | None = None) -> dict:
    pushes = [f"echo {base64.b64encode(probe.read_bytes()).decode()} | base64 -d > "
              f"/tmp/mat008_{probe.stem}.py"]
    for flag, relative in (sidecars or {}).items():
        source = REPO_ROOT / relative
        remote = f"/tmp/mat008_{Path(relative).name}"
        pushes.append(f"echo {base64.b64encode(source.read_bytes()).decode()} | base64 -d > {remote}")
        argv = argv + [flag, remote]
    script = (
        "export VIRTUAL_ENV=/opt/vllm_kunlun PATH=/opt/vllm_kunlun/bin:$PATH; "
        + " && ".join(pushes)
        + f" && python3 /tmp/mat008_{probe.stem}.py "
        + " ".join(shlex.quote(part) for part in argv)
        + " 2>/dev/null"
    )
    result = adapter.exec(pod, script, timeout=1800)
    for line in reversed(result.stdout.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise EvaluationFailed(
        "NEEDS_HUMAN", f"probe produced no JSON: {(result.stdout + result.stderr)[-500:]}"
    )


def render_table(report: dict) -> str:
    lines = [
        f"# Capability evaluation — {report['dimension']}",
        "",
        f"- subject: `{report.get('subject')}`",
        f"- exercised in: `{report.get('exercised_in')}`",
        f"- weights: `{report.get('weights')}`",
        f"- operators: {', '.join(f'`{op}`' for op in report.get('operators', []))}",
        f"- state: **{report.get('state')}**",
        "",
        "| Case | Reference | cosine | relative L2 |",
        "| --- | --- | --- | --- |",
    ]
    for case in report.get("cases", []):
        lines.append(
            f"| {case['case']} | {case['reference']} | {case['cosine']:.6f} | "
            f"{case['relative_l2']:.6f} |"
        )
    control = report.get("control") or {}
    lines += [
        "",
        "## Negative control",
        "",
        f"- {control.get('description')}",
        f"- relative L2 {control.get('relative_l2')}, cosine {control.get('cosine')}",
        f"- discriminates: **{control.get('discriminates')}**",
    ]
    if control.get("cosine_is_blind_to_this"):
        lines.append(
            "- cosine stayed above its floor for the wrong answer, which is why this "
            "dimension gates on the relative L2 norm instead."
        )
    lines += [
        "",
        "## What this does and does not say",
        "",
        "- `EXERCISED` means these operators ran on this hardware and agreed with a",
        "  float32 reference computed off the accelerator.",
        "- It covers one layer, not the model. A served deployment still needs",
        "  KDP-001b and MAT-013.",
    ]
    return "\n".join(lines) + "\n"


def probe_argv(dimension: str, args, thresholds: dict) -> list[str]:
    """Every probe argument comes from the contract unless the caller overrides it."""
    common = ["--cosine-floor", str(thresholds["min_cosine"]),
              "--max-relative-l2", str(thresholds["max_relative_l2"])]
    if dimension == "quantization":
        if not args.model_path:
            raise EvaluationFailed("CONTRACT_INVALID",
                                   "--model-path is required to exercise a quantized layer")
        return [
            "--model-path", args.model_path,
            "--tensor", args.tensor or thresholds["tensor"],
            "--tokens", str(args.tokens or thresholds["tokens"]),
        ] + common
    if dimension == "msa":
        # No weights: the window is a property of the kernel and the mask, so the
        # geometry is what has to be pinned. Taking it from the contract keeps a
        # passing run from having been a differently shaped one.
        geometry = thresholds["geometry"]
        return [
            "--heads", str(geometry["heads"]),
            "--kv-heads", str(geometry["kv_heads"]),
            "--head-dim", str(geometry["head_dim"]),
            "--block-size", str(geometry["block_size"]),
            "--batch", str(geometry["batch"]),
            "--context-len", str(geometry["context_len"]),
            "--window", str(geometry["window"]),
        ] + common
    if dimension == "moe":
        # Also weightless, and also geometry-critical: the two token counts have to
        # straddle the 768 preprocessing switch or both cases take the same path.
        geometry = thresholds["geometry"]
        return [
            "--experts", str(geometry["experts"]),
            "--hidden", str(geometry["hidden"]),
            "--intermediate", str(geometry["intermediate"]),
            "--top-k", str(geometry["top_k"]),
            "--tokens-below", str(geometry["tokens_below"]),
            "--tokens-above", str(geometry["tokens_above"]),
        ] + common
    raise EvaluationFailed("CONTRACT_INVALID", f"no argument mapping for dimension {dimension!r}")


def evaluate(args, contract: dict) -> dict:
    thresholds = thresholds_for(contract, args.dimension)
    probe_path = PROBES.get(args.dimension)
    if probe_path is None:
        # Declared and not implemented is a fact, not a pass. Saying so keeps the
        # capability table from filling up with dimensions nobody exercised.
        return {
            "dimension": args.dimension,
            "state": "EVALUATION_UNIMPLEMENTED",
            "reason": "no probe is registered for this dimension yet",
            "subject": args.subject,
            "cases": [],
        }
    if not args.pod:
        raise EvaluationFailed("CONTRACT_INVALID", "--pod is required to exercise a capability")

    argv = probe_argv(args.dimension, args, thresholds)
    adapter = KunlunP800Adapter()
    adapter.assert_owned(args.pod)
    report = run_probe(adapter, args.pod, REPO_ROOT / probe_path, argv,
                       SIDECARS.get(args.dimension))
    report["subject"] = args.subject
    report["exercised_in"] = args.pod
    report["weights"] = args.model_path
    report["threshold_source"] = str(CONTRACT.relative_to(REPO_ROOT))
    report["probe"] = probe_path
    return report


def aggregate(children: list[Path]) -> dict:
    """Fan-in. The node is only as good as its weakest dimension."""
    order = ["EVALUATION_ERROR", "EXERCISED_FAIL", "EVALUATION_INCONCLUSIVE",
             "EVALUATION_UNIMPLEMENTED", "EXERCISED_PASS"]
    rows = []
    for child in children:
        path = child / "capability_evaluation.json"
        if not path.exists():
            rows.append({"dimension": child.name, "state": "EVALUATION_ERROR",
                         "reason": f"{path} is missing"})
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"dimension": report.get("dimension", child.name),
                     "state": report.get("state", "UNKNOWN"),
                     "artifacts": str(child)})
    worst = min((row["state"] for row in rows), key=lambda s: order.index(s) if s in order else 0,
                default="EVALUATION_ERROR")
    state = {"EXERCISED_PASS": "EVALUATION_PASS"}.get(worst)
    if state is None:
        state = "EVALUATION_PARTIAL" if worst == "EVALUATION_UNIMPLEMENTED" else "EVALUATION_FAIL"
    return {"state": state, "worst_dimension_state": worst, "dimensions": rows}


def main() -> int:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capability-match", help="capability_match.json from MAT-003")
    parser.add_argument("--list-dimensions", action="store_true",
                        help="print the dimensions this model demands, as JSON")
    parser.add_argument("--aggregate", action="store_true", help="fan-in over --child dirs")
    parser.add_argument("--child", action="append", type=Path, default=[])
    parser.add_argument("--dimension", choices=["quantization", "msa", "moe", "multimodal"])
    parser.add_argument("--subject")
    parser.add_argument("--pod")
    parser.add_argument("--model-path", help="weights to exercise; may differ from the subject")
    parser.add_argument("--tensor", help="override the contract's tensor")
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))

    if args.list_dimensions:
        if not args.capability_match:
            raise EvaluationFailed("CONTRACT_INVALID", "--list-dimensions needs --capability-match")
        match = json.loads(Path(args.capability_match).read_text(encoding="utf-8"))
        print(json.dumps(dimensions_for(match)))
        return 0

    if not args.out:
        raise EvaluationFailed("CONTRACT_INVALID", "--out is required")
    args.out.mkdir(parents=True, exist_ok=True)

    if args.aggregate:
        summary = aggregate(args.child)
        (args.out / "evaluation_status.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(f"{summary['state']} over {len(summary['dimensions'])} dimension(s)")
        for row in summary["dimensions"]:
            print(f"  {row['dimension']:14} {row['state']}")
        return 0 if summary["state"] != "EVALUATION_FAIL" else 1

    if not args.dimension:
        raise EvaluationFailed("CONTRACT_INVALID", "--dimension is required")
    report = evaluate(args, contract)
    (args.out / "capability_evaluation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if report.get("cases"):
        (args.out / "capability_evaluation_table.md").write_text(
            render_table(report), encoding="utf-8"
        )
    gate = validate_evaluation(report, contract)
    (args.out / "evaluation_status.json").write_text(
        json.dumps({"state": report.get("state"),
                    "validator": {"passed": not gate, "errors": gate}}, indent=2),
        encoding="utf-8",
    )
    if gate:
        raise EvaluationFailed("CONTRACT_INVALID", "; ".join(gate))
    print(f"{report['dimension']}: {report['state']}")
    for case in report.get("cases", []):
        print(f"  {case['case']:34} cos={case['cosine']:.6f} relL2={case['relative_l2']:.6f}")
    return 0 if report["state"] != "EXERCISED_FAIL" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvaluationFailed as failure:
        print(f"{failure.state}: {failure.reason}", file=sys.stderr)
        raise SystemExit(1) from failure
