"""MAT-013 Accuracy Differential: is the served model still computing the model.

A deployment proof shows the server answers. It does not show the answer is right,
and a wrong attention kernel produces fluent text — which is exactly the risk after
replacing a decode kernel with a torch fallback.

The comparison is at the distribution, not the wording. One forward pass over a
fixed prompt, greedy, and the top-k next-token distribution from the server is
compared against a reference implementation of the same checkpoint. Comparing
generated strings would be both weaker (a rank-3 swap deep in a sentence is
invisible) and unstable (any sampling difference reads as a failure).

The reference is `transformers` on CPU, on the same weights: slow, independent of
the accelerator, and therefore the only thing on this cluster that can disagree
with the accelerator in a meaningful way.
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
from validators.accuracy_validator import validate_accuracy_report  # noqa: E402

CONTRACT = REPO_ROOT / "tasks" / "mat-013-accuracy-differential" / "task.yaml"
REFERENCE_PROBE = REPO_ROOT / "tools" / "probe" / "cpu_reference_logits_probe.py"
PROMPTS = [
    "The capital of France is",
    "2 + 2 =",
    "def add(a, b):\n    return",
]


class AccuracyFailed(RuntimeError):
    def __init__(self, state: str, reason: str) -> None:
        super().__init__(f"{state}: {reason}")
        self.state = state
        self.reason = reason


def server_topk(adapter: KunlunP800Adapter, pod: str, port: int, model: str,
                prompt: str, top_k: int) -> list[dict]:
    """Ask the running server for its next-token distribution."""
    payload = json.dumps(
        {"model": model, "prompt": prompt, "max_tokens": 1, "temperature": 0, "logprobs": top_k}
    )
    script = (
        f"curl -s http://127.0.0.1:{port}/v1/completions "
        f"-H 'Content-Type: application/json' -d {shlex.quote(payload)}"
    )
    result = adapter.exec(pod, script, timeout=300)
    text = result.stdout[result.stdout.find("{"):]
    try:
        body = json.loads(text)
    except json.JSONDecodeError as error:
        raise AccuracyFailed("NEEDS_HUMAN", f"server returned no JSON: {result.stdout[-300:]}") from error
    logprobs = (body.get("choices") or [{}])[0].get("logprobs") or {}
    ranked = (logprobs.get("top_logprobs") or [{}])[0]
    return [
        {"token": token, "logprob": value}
        for token, value in sorted(ranked.items(), key=lambda item: -item[1])
    ]


def reference_topk(adapter: KunlunP800Adapter, pod: str, model_path: str,
                   prompts: list[str], top_k: int, timeout: int) -> dict:
    payload = base64.b64encode(REFERENCE_PROBE.read_bytes()).decode()
    script = (
        "export VIRTUAL_ENV=/opt/vllm_kunlun PATH=/opt/vllm_kunlun/bin:$PATH "
        # The reference must not touch the accelerator, or it stops being independent.
        "CUDA_VISIBLE_DEVICES= XPU_VISIBLE_DEVICES=; "
        f"echo {payload} | base64 -d > /tmp/mat013_reference.py && "
        f"python3 /tmp/mat013_reference.py {shlex.quote(model_path)} {top_k} "
        + " ".join(shlex.quote(prompt) for prompt in prompts)
    )
    result = adapter.exec(pod, script, timeout=timeout)
    for line in reversed(result.stdout.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise AccuracyFailed("NEEDS_HUMAN", f"reference produced no JSON: {(result.stdout + result.stderr)[-400:]}")


def compare(candidate: list[dict], reference: list[dict], top_k: int) -> dict:
    """Top-1 identity plus set overlap: rank noise deep in the tail is not a defect."""
    candidate_tokens = [entry["token"] for entry in candidate][:top_k]
    reference_tokens = [entry["token"] for entry in reference][:top_k]
    overlap = len(set(candidate_tokens[:5]) & set(reference_tokens[:5]))
    return {
        "top1_candidate": candidate_tokens[0] if candidate_tokens else None,
        "top1_reference": reference_tokens[0] if reference_tokens else None,
        "top1_match": bool(candidate_tokens and reference_tokens
                           and candidate_tokens[0] == reference_tokens[0]),
        "top5_overlap": overlap,
        "candidate_top5": candidate_tokens[:5],
        "reference_top5": reference_tokens[:5],
    }


def main() -> int:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod", required=True)
    parser.add_argument("--model-request", required=True, help="mat-001 model_request.yaml")
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--reference-timeout", type=int, default=3600)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    request = yaml.safe_load(Path(args.model_request).read_text(encoding="utf-8"))
    model_path = (request.get("model") or {}).get("source")
    if not model_path:
        raise AccuracyFailed("CONTRACT_INVALID", "the ModelRequest carries no model.source")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    adapter = KunlunP800Adapter()
    adapter.assert_owned(args.pod)

    reference = reference_topk(adapter, args.pod, model_path, PROMPTS, args.top_k,
                              args.reference_timeout)
    if reference.get("state") != "REFERENCE_READY":
        raise AccuracyFailed("NEEDS_HUMAN", f"reference: {reference.get('reason')}")

    cases = []
    for prompt in PROMPTS:
        candidate = server_topk(adapter, args.pod, args.port, args.served_model_name,
                                prompt, args.top_k)
        entry = compare(candidate, reference["results"][prompt], args.top_k)
        entry["prompt"] = prompt
        cases.append(entry)

    matched = sum(1 for case in cases if case["top1_match"])
    report = {
        "status": "ACCURACY_PASS" if matched == len(cases) else "ACCURACY_FAIL",
        "subject": (request.get("model") or {}).get("id"),
        "revision": (request.get("model") or {}).get("revision"),
        "reference": {
            "implementation": "transformers on CPU",
            "device": "cpu",
            "dtype": reference.get("dtype"),
            "transformers_version": reference.get("transformers_version"),
        },
        "candidate": {"pod": args.pod, "served_model_name": args.served_model_name},
        "metric": "top-1 next-token identity and top-5 set overlap",
        "threshold_source": "tasks/mat-013-accuracy-differential/task.yaml",
        "top1_agreement": f"{matched}/{len(cases)}",
        "cases": cases,
    }
    (out / "accuracy_differential.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    thresholds = (contract.get("checks") or {}).get("thresholds") or {}
    gate = validate_accuracy_report(report, thresholds)
    (out / "accuracy_status.json").write_text(
        json.dumps({"state": report["status"], "validator": {"passed": not gate, "errors": gate}}, indent=2),
        encoding="utf-8",
    )
    for case in cases:
        mark = "OK  " if case["top1_match"] else "DIFF"
        print(f"{mark} {case['prompt']!r} -> {case['top1_candidate']!r} vs {case['top1_reference']!r} "
              f"(top5 overlap {case['top5_overlap']}/5)")
    print(f"status: {report['status']}  validator: {'passed' if not gate else gate}")
    return 0 if not gate else 6


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AccuracyFailed as failure:
        print(f"{failure.state}: {failure.reason}", file=sys.stderr)
        raise SystemExit(1) from failure
