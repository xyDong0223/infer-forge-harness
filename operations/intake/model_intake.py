"""MAT-001 Model Intake: pin a model identity to a reproducible revision.

Intake answers exactly one question — *which* model, *which* revision, *where*,
against *which* stack commit — and nothing about how it will be run. Launch
parameters belong to the deployment plan, so a wrong dtype guess here cannot
contaminate the identity fact.

Two facts cannot be established from this host: the weights live on a PVC that
is only visible inside the cluster, and the stack ref must be resolved to a
commit. So intake creates a throwaway XPU-free Pod, fingerprints the checkpoint
through `tools/probe/model_fingerprint_probe.py`, removes the Pod, and resolves
the ref with `git ls-remote`.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from core.paths import REPO_ROOT

from adapters import SafetyViolation, get_hardware, push_snippet  # noqa: E402
KunlunP800Adapter = get_hardware()
from core.errors import ToolFailed
from runners.deployment_proof import render_manifest  # noqa: E402
from validators.intake_validator import validate_model_request  # noqa: E402

PROBE = REPO_ROOT / "tools" / "probe" / "model_fingerprint_probe.py"
CONTRACT = REPO_ROOT / "tasks" / "mat-001-model-intake" / "task.yaml"
POD_TEMPLATE = REPO_ROOT / "tasks" / "mat-001-model-intake" / "manifests" / "probe-pod.template.yaml"
TASK_ID = "mat-001-model-intake"
UPSTREAM = "https://github.com/baidu/vLLM-Kunlun"


class IntakeFailed(ToolFailed):
    """Task-specific name for the shared (state, reason) failure."""


def resolve_stack_commit(ref: str, repo: str = UPSTREAM, attempts: int = 3) -> str:
    """Resolve a branch or tag to a commit sha.

    A branch moves, so recording `v0.25.1-dev` would make every downstream
    conclusion irreproducible — this is the same failure that once left a local
    clone 23 commits behind origin.

    Retried because the outbound proxy intermittently drops the TLS handshake
    (`gnutls_handshake() failed`), and a flaky network must not be reported as a
    missing ref.
    """
    last = ""
    for attempt in range(1, attempts + 1):
        result = subprocess.run(
            ["git", "ls-remote", repo, f"refs/heads/{ref}", f"refs/tags/{ref}"],
            check=False,
            text=True,
            capture_output=True,
            timeout=180,
        )
        if result.returncode == 0:
            lines = [line.split("\t")[0] for line in result.stdout.strip().splitlines() if line.strip()]
            if not lines:
                raise IntakeFailed("CONTRACT_INVALID", f"ref {ref!r} does not exist in {repo}")
            return lines[0]
        last = result.stderr.strip()
        if attempt < attempts:
            time.sleep(5)
    raise IntakeFailed("NEEDS_HUMAN", f"cannot reach {repo} after {attempts} attempts: {last}")


def run_probe(adapter: KunlunP800Adapter, pod: str, model_path: str, timeout: int) -> dict:
    push = push_snippet(PROBE, '/tmp/mat001_probe.py')
    script = (
        f"{push} && "
        f"python3 /tmp/mat001_probe.py {shlex.quote(model_path)}"
    )
    result = adapter.exec(pod, script, timeout=timeout)
    if result.returncode != 0:
        raise IntakeFailed("NEEDS_HUMAN", f"probe failed in {pod}: {result.stderr.strip()[:400]}")
    for line in reversed(result.stdout.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise IntakeFailed("NEEDS_HUMAN", f"probe produced no JSON: {result.stdout[-400:]}")


class ProbePod:
    """A throwaway Pod, removed even when the probe fails.

    Retaining it on failure would be the wrong default here: unlike a deployment
    proof, an intake failure leaves nothing useful inside the Pod — the evidence
    is the probe's JSON, which is already captured.
    """

    def __init__(self, adapter: KunlunP800Adapter, values: dict[str, str], out: Path) -> None:
        self.adapter = adapter
        self.values = values
        self.out = out
        self.name = values["RESOURCE_NAME"]

    def __enter__(self) -> str:
        manifest = render_manifest(POD_TEMPLATE, self.values)
        path = self.out / "probe_pod.yaml"
        path.write_text(manifest, encoding="utf-8")
        result = self.adapter.apply(path)
        if result.returncode != 0:
            raise IntakeFailed("NEEDS_HUMAN", f"cannot create probe pod: {result.stderr.strip()}")
        deadline = time.monotonic() + int(self.values["READY_TIMEOUT"])
        while time.monotonic() < deadline:
            if self.adapter.pod_ready(self.name):
                return self.name
            time.sleep(5)
        raise IntakeFailed("NEEDS_HUMAN", f"probe pod {self.name} did not become ready")

    def __exit__(self, *exc: object) -> None:
        try:
            self.adapter.delete_ephemeral(
                "pod", self.name, TASK_ID, self.values["ATTEMPT_ID"]
            )
        except SafetyViolation as violation:
            # Never swallow this: a refusal here means the Pod is not what we
            # think it is, and a human has to look.
            print(f"warning: probe pod not removed: {violation}", file=sys.stderr)


def build_request(args: argparse.Namespace, probe: dict, stack_commit: str) -> dict:
    identity = probe.get("identity") or {}
    return {
        "api_version": "infer.kunlun/v1alpha1",
        "kind": "ModelRequest",
        "metadata": {"task": TASK_ID, "attempt_id": args.attempt_id},
        "model": {
            "id": args.model_id,
            "source": args.model_path,
            "pvc": args.pvc,
            "revision": probe["revision"],
            "revision_method": probe["revision_method"],
            "shard_count": probe["shard_count"],
            "total_weight_bytes": probe["total_weight_bytes"],
            "trust_remote_code_required": probe["trust_remote_code_required"],
        },
        "identity": identity,
        "target": {
            "hardware": args.hardware,
            "vllm_kunlun_ref": args.stack_ref,
            "vllm_kunlun_commit": stack_commit,
        },
        # Deliberately absent: dtype, tensor_parallel_size, max_model_len. Those
        # are decisions, not facts about the checkpoint.
        "limitations": [
            "revision covers structure files in full and the first/last 1 MiB of every shard",
            "no runtime capability is claimed: a fingerprint is not a support statement",
        ],
    }


def execute(args) -> int:
    import yaml

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    adapter = KunlunP800Adapter()
    stack_commit = resolve_stack_commit(args.stack_ref)

    if args.attach_pod:
        adapter.assert_owned(args.attach_pod)
        if not adapter.pod_ready(args.attach_pod):
            raise IntakeFailed("NEEDS_HUMAN", f"{args.attach_pod} is not ready")
        probe = run_probe(adapter, args.attach_pod, args.model_path, args.probe_timeout)
        probe["probed_in"] = f"imported context: {args.attach_pod}"
    else:
        values = {
            "RESOURCE_NAME": f"{adapter.config.resource_prefix}mat001-probe-{args.attempt_id}",
            "NAMESPACE": adapter.config.namespace,
            "TASK_ID": TASK_ID,
            "ATTEMPT_ID": args.attempt_id,
            "IMAGE": args.image,
            "MODEL_PVC": args.pvc,
            "DEDICATED_POOL": args.dedicated_pool,
            "TTL_SECONDS": str(args.ttl_seconds),
            "READY_TIMEOUT": str(args.ready_timeout),
        }
        with ProbePod(adapter, values, out) as pod:
            probe = run_probe(adapter, pod, args.model_path, args.probe_timeout)
            probe["probed_in"] = f"ephemeral pod: {pod}"

    (out / "resolved_revision.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")
    if probe["state"] != "INTAKE_READY":
        raise IntakeFailed(probe["state"], probe.get("reason", "probe rejected the checkpoint"))

    request = build_request(args, probe, stack_commit)
    (out / "model_request.yaml").write_text(
        yaml.safe_dump(request, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    # The Tool does not get to declare its own output acceptable. The contract is
    # the source of the rules, and the artifact is written before the verdict so
    # a rejection stays inspectable.
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    gate = validate_model_request(request, contract, probe)
    (out / "intake_status.json").write_text(
        json.dumps({"state": "INTAKE_READY", "validator": {"passed": not gate, "errors": gate}}, indent=2),
        encoding="utf-8",
    )
    if gate:
        raise IntakeFailed("CONTRACT_INVALID", "; ".join(gate))

    print(f"INTAKE_READY {args.model_id} revision={probe['revision'][:16]} stack={stack_commit[:12]}")
    print(f"artifacts: {out}/model_request.yaml, {out}/resolved_revision.json")
    return 0
