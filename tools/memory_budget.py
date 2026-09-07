"""Reconcile a vLLM-Kunlun startup log against real P800 device counters.

The upstream capacity planner
(https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS, skill
`llm-serving-capacity-planner`) already decomposes a vLLM/SGLang startup log
into weights / KV pool / graph capture. It cannot do two things on Kunlun:

1. it has no P800 entry in `references/gpu-specs.json`, so total HBM is guessed;
2. it expects `nvidia-smi`, which does not exist in a P800 container.

This Tool supplies both from the cluster — `xpu_smi` rendered in nvidia-smi CSV
shape and the device facts from `catalog/xpu_specs.yaml` — then reconciles the
log-derived categories against what the cards actually report. It only collects
and computes; the pass/fail decision belongs to `validators/memory_validator.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.kunlun_p800.adapter import KunlunP800Adapter  # noqa: E402
from validators.memory_validator import validate_memory_budget  # noqa: E402

SPECS_PATH = REPO_ROOT / "catalog" / "xpu_specs.yaml"
CONTRACT = REPO_ROOT / "tasks" / "mem-001-memory-budget" / "task.yaml"
ANALYZER_RELATIVE = Path("skills/llm-serving-capacity-planner/scripts/capacity_analyzer.py")
MIB_PER_GIB = 1024


class BudgetError(RuntimeError):
    """Raised when evidence cannot be collected or is self-inconsistent."""


def load_device_spec(device: str, path: Path = SPECS_PATH) -> dict:
    import yaml

    entries = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("entries", {})
    if device not in entries:
        raise BudgetError(f"unknown device {device!r}: add it to {path.name} with cluster evidence")
    return entries[device]


def resolve_analyzer(explicit: str | None) -> Path:
    """Locate the vendored capacity analyzer without hardcoding a clone path."""
    candidates = [explicit, os.environ.get("AI_INFRA_SKILLS_DIR")]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_file():
            return path
        if (path / ANALYZER_RELATIVE).is_file():
            return path / ANALYZER_RELATIVE
    raise BudgetError(
        "capacity analyzer not found: pass --analyzer or export AI_INFRA_SKILLS_DIR to a clone of "
        "https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS"
    )


def run_analyzer(analyzer: Path, log_file: Path, smi_file: Path) -> dict:
    result = subprocess.run(
        [
            sys.executable,
            str(analyzer),
            "--log-file",
            str(log_file),
            "--nvidia-smi-file",
            str(smi_file),
            "--format",
            "json",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise BudgetError(f"capacity analyzer failed: {result.stderr.strip()}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:  # the analyzer prints warnings on stderr, not stdout
        raise BudgetError(f"capacity analyzer produced no JSON: {exc}") from exc


def reconcile(report: dict, cards: list[dict[str, int]], spec: dict, rank: int = 0) -> dict:
    """Compare log-attributed memory with what the cards report.

    The analyzer folds `smi_used - sum_of_known` into `other_gib`, so that value
    is a claim about unattributed memory rather than a measurement. It is
    recomputed here from the device counters and the two are compared: a
    disagreement means the log and the snapshot describe different processes.
    """
    card = next((entry for entry in cards if entry["index"] == rank), None)
    if card is None:
        raise BudgetError(f"xpu_smi has no card at rank {rank}")

    breakdown = report.get("memory_breakdown", {})
    attributed_mib = round(
        sum(
            float(breakdown.get(key, 0.0) or 0.0)
            for key in ("model_weights_gib", "kv_pool_gib", "cuda_graph_gib", "framework_overhead_gib")
        )
        * MIB_PER_GIB
    )
    unattributed_mib = card["used_mib"] - attributed_mib
    analyzer_other_mib = round(float(breakdown.get("other_gib", 0.0) or 0.0) * MIB_PER_GIB)

    used = [entry["used_mib"] for entry in cards]
    spec_total = int(spec["hbm_mib"])
    # The analyzer rounds every category to 0.01 GiB (~10 MiB), and four of them
    # feed the comparison, so anything under ~2 rounding steps is not a
    # disagreement about the process, only about precision.
    rounding_slack_mib = 32
    return {
        "device": spec.get("display_name"),
        "rank": rank,
        "hbm_total_mib": spec_total,
        "hbm_total_matches_device": all(entry["total_mib"] == spec_total for entry in cards),
        "measured_used_mib": card["used_mib"],
        "measured_free_mib": card["free_mib"],
        "utilization_pct": round(card["used_mib"] / spec_total * 100, 1),
        "log_attributed_mib": attributed_mib,
        "unattributed_mib": unattributed_mib,
        "analyzer_other_mib": analyzer_other_mib,
        "reconciled": abs(unattributed_mib - analyzer_other_mib) <= rounding_slack_mib,
        "cards": len(cards),
        "card_spread_mib": max(used) - min(used),
        "categories_mib": {
            "model_weights": round(float(breakdown.get("model_weights_gib", 0.0) or 0.0) * MIB_PER_GIB),
            "kv_pool": round(float(breakdown.get("kv_pool_gib", 0.0) or 0.0) * MIB_PER_GIB),
            "graph_capture": round(float(breakdown.get("cuda_graph_gib", 0.0) or 0.0) * MIB_PER_GIB),
            "framework": round(float(breakdown.get("framework_overhead_gib", 0.0) or 0.0) * MIB_PER_GIB),
        },
        "kv_cache_tokens": (report.get("vllm") or {}).get("gpu_kv_cache_tokens"),
        "max_model_len": (report.get("vllm") or {}).get("max_model_len"),
        "reported_concurrency": (report.get("vllm") or {}).get("maximum_concurrency"),
    }


def render(budget: dict, cards: list[dict[str, int]]) -> str:
    total = budget["hbm_total_mib"]
    lines = [
        f"# Memory budget — {budget['device']} rank {budget['rank']}",
        "",
        f"HBM {total} MiB per card, {budget['cards']} cards, "
        f"{budget['utilization_pct']}% used, {budget['measured_free_mib']} MiB free",
        "",
        "| Category | MiB | % of HBM | Source |",
        "| --- | --- | --- | --- |",
    ]
    sources = {
        "model_weights": "server log: Model loading took",
        "kv_pool": "server log: Available KV cache memory",
        "graph_capture": "server log: Graph capturing finished",
        "framework": "server log: initial free memory",
    }
    for name, value in budget["categories_mib"].items():
        lines.append(f"| {name} | {value} | {value / total * 100:.1f}% | {sources[name]} |")
    gap = budget["unattributed_mib"]
    lines += [
        f"| unattributed | {gap} | {gap / total * 100:.1f}% | xpu_smi used - log attributed |",
        f"| free | {budget['measured_free_mib']} | "
        f"{budget['measured_free_mib'] / total * 100:.1f}% | xpu_smi |",
        "",
        f"KV cache: {budget['kv_cache_tokens']} tokens, max_model_len {budget['max_model_len']}, "
        f"reported concurrency {budget['reported_concurrency']}x",
        f"Per-card spread: {budget['card_spread_mib']} MiB",
        "",
        "| Card | Used MiB | Free MiB |",
        "| --- | --- | --- |",
    ]
    lines += [f"| {c['index']} | {c['used_mib']} | {c['free_mib']} |" for c in cards]
    if not budget["hbm_total_matches_device"]:
        lines.append("")
        lines.append("WARNING: xpu_smi total HBM disagrees with catalog/xpu_specs.yaml")
    if not budget["reconciled"]:
        lines.append("")
        lines.append(
            "WARNING: the analyzer's `other` bucket "
            f"({budget['analyzer_other_mib']} MiB) does not match the recomputed gap "
            f"({gap} MiB) — log and snapshot may describe different processes"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod", help="pod to snapshot; omit only with --xpu-smi-file")
    parser.add_argument("--server-log", help="in-pod server log path to pull")
    parser.add_argument("--log-file", help="local server log, instead of pulling from the pod")
    parser.add_argument("--xpu-smi-file", help="pre-captured `xpu_smi -m` output")
    parser.add_argument("--analyzer", help="capacity analyzer script or its repository root")
    parser.add_argument("--device", default="p800")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--out", required=True, help="evidence directory to write")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = load_device_spec(args.device)
    analyzer = resolve_analyzer(args.analyzer)

    if args.xpu_smi_file:
        # An offline snapshot keeps the report reproducible without cluster access.
        cards = KunlunP800Adapter.parse_xpu_smi(Path(args.xpu_smi_file).read_text(encoding="utf-8"))
        if not cards:
            raise BudgetError(f"no parsable card in {args.xpu_smi_file}")
        adapter = None
    else:
        if not args.pod:
            raise BudgetError("either --pod or --xpu-smi-file is required")
        adapter = KunlunP800Adapter()
        cards = adapter.xpu_smi(args.pod)

    if args.log_file:
        log_file = Path(args.log_file)
    else:
        if not (adapter and args.server_log):
            raise BudgetError("either --log-file or --pod with --server-log is required")
        result = adapter.exec(args.pod, f"cat {args.server_log}", timeout=300)
        if result.returncode != 0:
            raise BudgetError(f"cannot read {args.server_log} in {args.pod}: {result.stderr.strip()}")
        log_file = out / "server_log.txt"
        log_file.write_text(result.stdout, encoding="utf-8")

    smi_file = out / "xpu_smi.csv"
    smi_file.write_text(KunlunP800Adapter.as_nvidia_smi_csv(cards), encoding="utf-8")
    report = run_analyzer(analyzer, log_file, smi_file)
    (out / "capacity_planner.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    budget = reconcile(report, cards, spec, args.rank)
    budget["evidence"] = {
        "log_file": str(log_file),
        "xpu_smi": str(smi_file),
        "capacity_planner": str(out / "capacity_planner.json"),
        "analyzer": str(analyzer),
    }
    (out / "memory_budget.json").write_text(json.dumps(budget, indent=2), encoding="utf-8")
    text = render(budget, cards)
    (out / "memory_budget.md").write_text(text, encoding="utf-8")
    print(text, end="")

    # The Tool does not judge its own output: thresholds come from the Task
    # contract, so tightening the contract tightens the verdict.
    import yaml

    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    thresholds = (contract.get("checks") or {}).get("thresholds") or {}
    gate = validate_memory_budget(budget, thresholds)
    state = "BUDGET_ACCEPTABLE" if not gate else "BUDGET_REJECTED"
    (out / "budget_status.json").write_text(
        json.dumps(
            {"state": state, "thresholds": thresholds, "validator": {"passed": not gate, "errors": gate}},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"state: {state}")
    for error in gate:
        print(f"  - {error}")
    return 0 if not gate else 6


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BudgetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
