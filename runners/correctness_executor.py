"""Sequence the three correctness tasks that used to stop the walk as MANUAL.

mat-021, mat-022 and mat-023 share one shape: exercise the real path, produce an
independent reference, compare, and prove with a discriminating control that the
comparison could have failed. The probes and tools already did the hard parts —
`sliding_window_decode_probe` and `block_sparse_attention_probe` compute CPU
float32 references and negative controls in the pod; `accuracy_differential`
compares the served model against transformers on CPU. What was missing was the
sequence and the packaging into each contract's evidence shape, which is what
kept all three in the MANUAL set and every correctness question in a person's
queue.

The verdicts stay mechanical and the validators stay independent: a control that
does not discriminate is AMBIGUOUS, never a pass; a probe that cannot produce
the tensors grades nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from adapters.kunlun_p800.adapter import KunlunP800Adapter, push_snippet  # noqa: E402
from validators.correctness_validator import (  # noqa: E402
    validate_end_to_end,
    validate_kernel_grade,
    validate_long_context,
)

KERNEL_CONTRACT = REPO_ROOT / "tasks" / "mat-021-platform-kernel-correctness" / "task.yaml"
E2E_CONTRACT = REPO_ROOT / "tasks" / "mat-022-end-to-end-accuracy" / "task.yaml"
LONG_CONTEXT_CONTRACT = REPO_ROOT / "tasks" / "mat-023-long-context-sparse-correctness" / "task.yaml"
WINDOW_PROBE = REPO_ROOT / "tools" / "probe" / "sliding_window_decode_probe.py"
SPARSE_PROBE = REPO_ROOT / "tools" / "probe" / "block_sparse_attention_probe.py"
FALLBACK = REPO_ROOT / "tools" / "torch" / "paged_decode.py"
TENSOR_DIFF = REPO_ROOT / "tools" / "tensor_diff.py"

# The reference provenance every mode asserts. "independently written" is a
# claim about the probe sources: the CPU references are computed from the paged
# cache directly, not by calling the kernel under test.
PROVENANCE = {"implementation": "torch", "device": "cpu", "dtype": "float32",
              "independently_written": True}


class CorrectnessOps:
    """Pod and subprocess access behind one seam, so tests inject fakes."""

    def __init__(self, adapter: KunlunP800Adapter):
        self.adapter = adapter

    def run_probe(self, pod: str, probe: Path, files: dict[str, Path], args: str) -> dict[str, Any]:
        """Inject a probe (plus files it needs) into the pod and run it.

        Probes print one JSON object on stdout; the last such line wins, the
        same convention the other in-pod tools use.
        """
        setup = [push_snippet(probe, "/tmp/kdp_probe.py")]
        for name, path in files.items():
            setup.append(push_snippet(path, f"/tmp/{name}"))
        command = " && ".join(setup) + f" && python3 /tmp/kdp_probe.py {args}"
        result = self.adapter.exec(pod, command, timeout=1800)
        text = result.stdout + result.stderr
        for line in reversed(text.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except ValueError:
                    continue
        return {"state": "EVALUATION_ERROR", "error": text.strip()[-500:] or "probe produced no output"}

    def fetch(self, pod: str, remote: str) -> str:
        result = self.adapter.exec(pod, f"cat {remote} 2>/dev/null", timeout=120)
        return result.stdout

    def run_tool(self, command: list[str]) -> Any:
        import subprocess

        result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
        return result


def _last_json(text: str) -> Any:
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return None


def _grade_with_tensor_diff(ops: CorrectnessOps, tensors: Path, max_relative_l2: float,
                            artifacts: Path) -> dict[str, Any] | None:
    """Run the contract's own grader on captured tensors."""
    result = ops.run_tool([
        "python3", str(TENSOR_DIFF),
        "--candidate", str(tensors / "candidate.json"),
        "--reference", str(tensors / "reference.json"),
        "--control", str(tensors / "control.json"),
        "--max-relative-l2", str(max_relative_l2),
        "--out", str(artifacts / "grade_raw.json"),
    ])
    if result.returncode not in (0, 6):  # 6 is a graded FAIL, still a grade
        return None
    payload = _last_json(result.stdout) or json.loads(
        (artifacts / "grade_raw.json").read_text(encoding="utf-8"))
    return payload


def _fetch_tensors(ops: CorrectnessOps, pod: str, remote_dir: str, names: list[str],
                   local_dir: Path) -> bool:
    local_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        content = ops.fetch(pod, f"{remote_dir}/{name}")
        if not content.strip():
            return False
        (local_dir / name).write_text(content, encoding="utf-8")
    return True


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _finish(status: dict[str, Any], out: Path, report: dict[str, Any],
            errors: list[str], pass_state: str, fail_state: str, ambiguous_state: str) -> dict[str, Any]:
    """Apply the validator: its disagreement overrides any computed verdict."""
    if errors:
        report["validation_errors"] = errors
        if report.get("state") == "PASS":
            report["state"] = "AMBIGUOUS"
    mapping = {"PASS": pass_state, "FAIL": fail_state, "AMBIGUOUS": ambiguous_state}
    status.update(report)
    status["state"] = mapping.get(report.get("state"), ambiguous_state)
    _write(out / "status.json", status)
    return status


def run_kernel_grade(pod: str, ops: CorrectnessOps, out: Path,
                     max_relative_l2: float) -> dict[str, Any]:
    """mat-021: grade the platform decode kernel against an independent reference."""
    status: dict[str, Any] = {"task_id": "mat-021-platform-kernel-correctness", "pod": pod}
    remote = "/tmp/kdp-mat021"
    probe = ops.run_probe(
        pod, WINDOW_PROBE, {"paged_decode.py": FALLBACK},
        f"--fallback /tmp/paged_decode.py --dump {remote}",
    )
    if probe.get("state") == "EVALUATION_ERROR":
        # The kernel call itself raised; that is measured failure, not missing evidence.
        return _finish(status, out, {"state": "FAIL",
                                     "reason": "the platform kernel raised instead of computing",
                                     "probe": probe},
                       [], "KERNEL_PASS", "KERNEL_FAIL", "KERNEL_AMBIGUOUS")
    if probe.get("state") != "EXERCISED_PASS" and probe.get("state") != "EXERCISED_FAIL":
        return _finish(status, out, {"state": "AMBIGUOUS", "probe": probe,
                                     "reason": probe.get("error", probe.get("state"))},
                       [], "KERNEL_PASS", "KERNEL_FAIL", "KERNEL_AMBIGUOUS")
    tensors = out / "tensors"
    if not _fetch_tensors(ops, pod, remote,
                          ["candidate.json", "reference.json", "control.json"], tensors):
        return _finish(status, out, {"state": "AMBIGUOUS",
                                     "reason": "the probe did not leave gradeable tensors"},
                       [], "KERNEL_PASS", "KERNEL_FAIL", "KERNEL_AMBIGUOUS")
    grade = _grade_with_tensor_diff(ops, tensors, max_relative_l2, out)
    if grade is None:
        return _finish(status, out, {"state": "AMBIGUOUS",
                                     "reason": "tensor_diff could not grade the captured tensors"},
                       [], "KERNEL_PASS", "KERNEL_FAIL", "KERNEL_AMBIGUOUS")
    state = "AMBIGUOUS" if not grade.get("control_discriminates") else (
        "PASS" if grade.get("pass") else "FAIL")
    report = {
        "state": state,
        "relative_l2": grade.get("relative_l2"),
        "max_abs_error": grade.get("max_abs_error"),
        "shape_candidate": grade.get("shape_candidate"),
        "shape_reference": grade.get("shape_reference"),
        "max_relative_l2": max_relative_l2,
        "reference_provenance": dict(PROVENANCE),
        "control_discriminates": grade.get("control_discriminates"),
        "control": grade.get("control"),
        "geometry": probe.get("geometry"),
        "operators": probe.get("operators"),
        "artifacts": ["grade_raw.json", "kernel_grade.json", "kernel_status.json",
                      "tensors/candidate.json", "tensors/reference.json", "tensors/control.json"],
    }
    _write(out / "kernel_grade.json", report)
    return _finish(status, out, report,
                   validate_kernel_grade({**report, "state": state}),
                   "KERNEL_PASS", "KERNEL_FAIL", "KERNEL_AMBIGUOUS")


def run_end_to_end(pod: str, ops: CorrectnessOps, out: Path, model_request: Path,
                   served_model_name: str, port: int) -> dict[str, Any]:
    """mat-022: the integrated serving path against an off-accelerator reference."""
    status: dict[str, Any] = {"task_id": "mat-022-end-to-end-accuracy", "pod": pod}
    if not model_request:
        status["state"] = "CONTRACT_INVALID"
        status["reason"] = "end-to-end accuracy needs the ModelRequest (pass --model-request)"
        _write(out / "status.json", status)
        return status
    differential = out / "differential"
    result = ops.run_tool([
        "python3", "tools/accuracy_differential.py",
        "--pod", pod, "--model-request", str(model_request),
        "--served-model-name", served_model_name, "--port", str(port),
        "--out", str(differential),
    ])
    payload_path = differential / "accuracy_differential.json"
    if not payload_path.exists():
        status["state"] = "NEEDS_HUMAN"
        status["reason"] = (result.stderr or result.stdout or "").strip()[-500:] \
            or "accuracy_differential produced no report"
        _write(out / "status.json", status)
        return status
    differential_report = json.loads(payload_path.read_text(encoding="utf-8"))
    reference = differential_report.get("reference") or {}
    report = {
        "state": differential_report.get("status"),
        "composition": "integrated_serving_path",
        "subject": differential_report.get("subject"),
        "revision": differential_report.get("revision"),
        "reference_provenance": {
            "implementation": reference.get("implementation", "transformers on CPU"),
            "device": reference.get("device", "cpu"),
            "dtype": reference.get("dtype"),
            "transformers_version": reference.get("transformers_version"),
            "must_not_see_accelerator": True,
        },
        "candidate": {"device": "kunlun-p800", "pod": pod,
                      "served_model_name": served_model_name, "port": port},
        "metric": differential_report.get("metric"),
        "top1_agreement": differential_report.get("top1_agreement"),
        "cases": [
            {"prompt": case.get("prompt"), "evidence": case}
            for case in differential_report.get("cases") or []
        ],
        "artifacts": ["end_to_end_accuracy.json", "accuracy_status.json",
                      "differential/accuracy_differential.json"],
    }
    _write(out / "end_to_end_accuracy.json", report)
    errors = validate_end_to_end(report)
    if errors:
        status["state"] = "CONTRACT_INVALID"
        status["validation_errors"] = errors
        _write(out / "status.json", status)
        return status
    status.update(report)
    status["state"] = report["state"]
    _write(out / "status.json", status)
    return status


def run_long_context(pod: str, ops: CorrectnessOps, out: Path, context_len: int,
                     block_size: int, topk: int, max_relative_l2: float) -> dict[str, Any]:
    """mat-023: sparse selection correctness beyond one block-selection boundary."""
    status: dict[str, Any] = {"task_id": "mat-023-long-context-sparse-correctness", "pod": pod}
    boundary = block_size * topk
    if context_len <= boundary:
        status["state"] = "CONTRACT_INVALID"
        status["reason"] = (f"context_len {context_len} does not cross block_size*topk "
                            f"= {boundary}, so the sparse path is not exercised")
        _write(out / "status.json", status)
        return status
    remote = "/tmp/kdp-mat023"
    probe = ops.run_probe(
        pod, SPARSE_PROBE, {},
        f"--context-len {context_len} --block-size {block_size} --topk {topk} "
        f"--max-relative-l2 {max_relative_l2} --dump {remote}",
    )
    if probe.get("state") == "EVALUATION_ERROR":
        return _finish(status, out, {"state": "FAIL",
                                     "reason": "the sparse kernel raised instead of computing",
                                     "probe": probe},
                       [], "LONG_CONTEXT_PASS", "LONG_CONTEXT_FAIL", "LONG_CONTEXT_AMBIGUOUS")
    if probe.get("state") != "EXERCISED_PASS" and probe.get("state") != "EXERCISED_FAIL":
        return _finish(status, out, {"state": "AMBIGUOUS", "probe": probe,
                                     "reason": probe.get("error", probe.get("state"))},
                       [], "LONG_CONTEXT_PASS", "LONG_CONTEXT_FAIL", "LONG_CONTEXT_AMBIGUOUS")
    tensors = out / "tensors"
    if not _fetch_tensors(ops, pod, remote,
                          ["candidate.json", "reference.json", "control.json",
                           "selected_blocks.json"], tensors):
        return _finish(status, out, {"state": "AMBIGUOUS",
                                     "reason": "the probe did not leave gradeable tensors"},
                       [], "LONG_CONTEXT_PASS", "LONG_CONTEXT_FAIL", "LONG_CONTEXT_AMBIGUOUS")
    selected_blocks = json.loads((tensors / "selected_blocks.json").read_text(encoding="utf-8"))
    grade = _grade_with_tensor_diff(ops, tensors, max_relative_l2, out)
    if grade is None:
        return _finish(status, out, {"state": "AMBIGUOUS",
                                     "reason": "tensor_diff could not grade the captured tensors"},
                       [], "LONG_CONTEXT_PASS", "LONG_CONTEXT_FAIL", "LONG_CONTEXT_AMBIGUOUS")
    state = "AMBIGUOUS" if not grade.get("control_discriminates") else (
        "PASS" if grade.get("pass") else "FAIL")
    report = {
        "state": state,
        "geometry": {"context_len": context_len, "block_size": block_size, "topk": topk,
                     "boundary": boundary},
        "selected_blocks": selected_blocks,
        "path": "sparse",
        "relative_l2": grade.get("relative_l2"),
        "max_relative_l2": max_relative_l2,
        "reference_provenance": dict(PROVENANCE),
        "control_discriminates": grade.get("control_discriminates"),
        "operators": probe.get("operators"),
        "artifacts": ["grade_raw.json", "long_context_grade.json", "long_context_status.json",
                      "tensors/candidate.json", "tensors/reference.json",
                      "tensors/control.json", "tensors/selected_blocks.json"],
    }
    _write(out / "long_context_grade.json", report)
    return _finish(status, out, report,
                   validate_long_context({**report, "state": state}),
                   "LONG_CONTEXT_PASS", "LONG_CONTEXT_FAIL", "LONG_CONTEXT_AMBIGUOUS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("kernel", "end-to-end", "long-context"))
    parser.add_argument("--pod", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-request", type=Path, default=None)
    parser.add_argument("--served-model-name", default="")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--context-len", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--max-relative-l2", type=float, default=0.02)
    args = parser.parse_args()

    ops = CorrectnessOps(KunlunP800Adapter())
    if args.mode == "kernel":
        status = run_kernel_grade(args.pod, ops, args.out, args.max_relative_l2)
    elif args.mode == "end-to-end":
        status = run_end_to_end(args.pod, ops, args.out, args.model_request,
                                args.served_model_name, args.port)
    else:
        status = run_long_context(args.pod, ops, args.out, args.context_len,
                                  args.block_size, args.topk, args.max_relative_l2)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    passing = {"KERNEL_PASS", "ACCURACY_PASS", "LONG_CONTEXT_PASS"}
    return 0 if status.get("state") in passing else 1


if __name__ == "__main__":
    raise SystemExit(main())
