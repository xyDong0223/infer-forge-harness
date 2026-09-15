"""Execute MAT-007 patch placement as one orchestrated sequence.

Placing the reversible torch fallback is also not a single command: apply the
patch, rerun the service proof so the server itself validates it, compare the
fallback numerically against the vendor kernel where that kernel works, and
record the placement. The apply tool already guarantees reversibility (a
.kdp_backup and the KDP_DECODE_KERNEL runtime switch); this runner adds the
validation halves the contract demands and stops at PATCH_REJECTED when either
fails — a patch that only makes the server start is a rewrite, not a patch.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from adapters.kunlun_p800.adapter import KunlunP800Adapter, push_snippet  # noqa: E402
from validators.patch_validator import validate_patch_placement  # noqa: E402

CONTRACT = REPO_ROOT / "tasks" / "mat-007-patch-placement" / "task.yaml"

# Runs inside the pod, where both the vendor kernel and the installed fallback
# exist. Setting KDP_DECODE_KERNEL=speculative before import disables the
# dispatch hook, so the probe sees the raw vendor symbol — the patch stays
# installed for the server; this step only validates implementations.
#
# Two questions, one script, because they share the geometry machinery:
#   1. reproduction: does the vendor kernel still fail at the geometry triage
#      captured? If not, the evidence for needing this patch is gone and the
#      placement is rejected no matter how well the fallback agrees.
#   2. comparison: does the fallback agree with the vendor kernel where the
#      vendor kernel works? Agreement there is what licenses using the
#      fallback where it does not.
# Geometry and seeding mirror tools/probe/kernel_ut_replay.py so the numbers
# are comparable with triage evidence. The captured geometry arrives on stdin.
POD_VALIDATION_SCRIPT = r'''
import json, os, sys
os.environ["KDP_DECODE_KERNEL"] = "speculative"  # validate raw symbols, not the hook
import torch
import vllm_kunlun  # noqa: F401 - installs the torch_xmlir mapping
import kunlun_ops
from kdp_torch_paged_decode import torch_paged_decode

captured = json.load(sys.stdin)

def build_case(heads, kv_heads, dim, batch, context_len, block_size, max_context_len, blocks):
    dtype = torch.bfloat16
    max_blocks = max(1, (max_context_len + block_size - 1) // block_size)
    used = max(1, (context_len + block_size - 1) // block_size)
    kv_cache = torch.zeros((2, blocks, kv_heads, block_size, dim), dtype=dtype, device="cuda")
    tables = torch.zeros((batch, max_blocks), dtype=torch.int32, device="cuda")
    for request in range(batch):
        for block in range(used):
            tables[request, block] = (request * used + block) % blocks
    lens = torch.full((batch,), context_len, dtype=torch.int32)
    return {
        "out": torch.zeros((batch, heads, dim), dtype=dtype, device="cuda"),
        "q": torch.randn((batch, 1, heads, dim), dtype=dtype, device="cuda"),
        "k_cache": kv_cache[0], "v_cache": kv_cache[1],
        "context_lens_cpu": lens, "context_lens_xpu": lens.to("cuda"),
        "batch_num": batch, "qlen": 1, "max_context_len": max_context_len,
        "head_num": heads, "head_dim": dim, "scale": dim ** -0.5,
        "kv_head_num": kv_heads, "block_size": block_size,
        "max_num_blocks_per_seq": max_blocks, "max_window_size": -1,
        "block_tables": tables, "sink": None,
    }

def cosine(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float((a @ b / (a.norm() * b.norm() + 1e-12)).item())

def captured_case():
    g = captured
    used = max(1, (int(g["context_len"]) + int(g["block_size"]) - 1) // int(g["block_size"]))
    return build_case(int(g["head_num"]), int(g["kv_head_num"]), int(g["head_dim"]),
                      int(g["batch_num"]), int(g["context_len"]), int(g["block_size"]),
                      int(g.get("max_context_len", 32768)),
                      int(g.get("batch_num", 1)) * used + 1)

reproduction = {"attempted": True, "reproduced": False, "error": None}
try:
    torch.manual_seed(20260914)
    kunlun_ops.speculative_attention(**captured_case())
    reproduction["note"] = "vendor kernel accepted the captured geometry"
except Exception as error:
    reproduction["reproduced"] = True
    reproduction["error"] = "%s: %s" % (type(error).__name__, error)

results = []
for batch in (1, 4, 16, 64):
    for context in (1, 128, 4096):
        for block_size in (16, 64, 128):
            label = "b%d/c%d/k%d" % (batch, context, block_size)
            try:
                torch.manual_seed(20260914)
                case = build_case(32, 8, 128, batch, context, block_size, 32768,
                                  batch * max(1, (context + block_size - 1) // block_size) + 1)
                vendor_out, torch_out = case["out"].clone(), case["out"].clone()
                kunlun_ops.speculative_attention(**{**case, "out": vendor_out})
                torch_paged_decode(**{**case, "out": torch_out})
                results.append({"case": label, "cosine": cosine(vendor_out, torch_out)})
            except Exception as error:
                results.append({"case": label, "error": "%s: %s" % (type(error).__name__, error)})
print(json.dumps({"reproduction": reproduction, "results": results}))
'''


class PatchOps:
    """Pod and subprocess access behind one seam, so tests inject fakes."""

    def __init__(self, adapter: KunlunP800Adapter):
        self.adapter = adapter

    def apply(self, pod: str) -> str:
        import subprocess

        result = subprocess.run(
            ["python3", "tools/patches/apply_torch_decode_patch.py", "--pod", pod],
            cwd=REPO_ROOT, text=True, capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"patch apply failed: {(result.stdout + result.stderr).strip()[-300:]}")
        return (result.stdout + result.stderr).strip()

    def remove(self, pod: str) -> str:
        import subprocess

        result = subprocess.run(
            ["python3", "tools/patches/apply_torch_decode_patch.py", "--pod", pod, "--remove"],
            cwd=REPO_ROOT, text=True, capture_output=True,
        )
        return (result.stdout + result.stderr).strip()

    def rerun_service(self, pod: str, contract_instance: Path, out: Path) -> dict[str, Any]:
        command = [
            "python3", "runners/task_runner.py", str(contract_instance),
            "--execute", "--phase", "service", "--attach-pod", pod,
            "--artifact-dir", str(out),
        ]
        import subprocess

        result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
        status_path = out / "status.json"
        payload = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
        payload.setdefault("reason", result.stderr.strip()[-500:])
        return payload

    def numerical_comparison(self, pod: str, geometry: dict[str, Any]) -> dict[str, Any]:
        """Validate the fallback inside the pod: reproduce, then compare.

        When the pod cannot answer, that is a REJECTED outcome, not a skipped
        one — a comparison that never ran licenses nothing.
        """
        feed = base64.b64encode(json.dumps(geometry).encode()).decode()
        command = (
            push_snippet(POD_VALIDATION_SCRIPT, "/tmp/kdp_patch_validate.py") + " && "
            f"echo {feed} | base64 -d | python3 /tmp/kdp_patch_validate.py"
        )
        result = self.adapter.exec(pod, command, timeout=600)
        text = result.stdout + result.stderr
        for line in reversed(text.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except ValueError:
                    continue
        return {"error": text.strip()[-500:] or "validation produced no output"}


class PatchExecutor:
    def __init__(self, pod: str, contract_instance: Path | None,
                 ops: PatchOps, out: Path, triage_report: Path | None = None):
        self.pod = pod
        self.contract_instance = contract_instance
        self.ops = ops
        self.out = out
        self.triage_report = triage_report

    def run(self) -> dict[str, Any]:
        status: dict[str, Any] = {"task_id": "mat-007-patch-placement", "pod": self.pod,
                                  "state": "PATCH_REJECTED"}
        try:
            report = self._place()
        except Exception as error:  # noqa: BLE001 - rejection is a verdict, not a crash
            status["reason"] = f"{type(error).__name__}: {error}"
            self._write(self.out / "status.json", status)
            return status
        status.update(report)
        status["state"] = report.get("state", "PATCH_REJECTED")
        self._write(self.out / "status.json", status)
        return status

    def _place(self) -> dict[str, Any]:
        self.out.mkdir(parents=True, exist_ok=True)
        # 1. Apply. Reversible by construction: .kdp_backup plus the
        #    KDP_DECODE_KERNEL runtime switch.
        self.ops.apply(self.pod)
        try:
            # 2. The server validates it; isolation already passed, so only the
            #    server can say the fallback works in context.
            if not self.contract_instance:
                raise RuntimeError(
                    "patch placement reruns the service proof for validation: "
                    "pass --contract-instance"
                )
            server = self.ops.rerun_service(self.pod, self.contract_instance,
                                            self.out / "service_rerun")
            # 3. Reproduce the original failure with the patch's routing off,
            #    and compare the fallback numerically where the vendor kernel
            #    works. Both run inside the pod in one pass.
            validation = self.ops.numerical_comparison(self.pod, self._captured_geometry())
            self._write(self.out / "numerical_comparison.txt", validation)
        except Exception:
            self.ops.remove(self.pod)
            raise

        reproduction = validation.get("reproduction") or {}
        cases = validation.get("results") or []
        usable = [case for case in cases if "cosine" in case]
        min_cosine = min((case["cosine"] for case in usable), default=None)
        if server.get("state") != "DEPLOYMENT_READY":
            self.ops.remove(self.pod)
            return {"state": "PATCH_REJECTED",
                    "reason": "service proof failed with the patch in place",
                    "server_validation": server, "numerical": validation}
        if reproduction.get("reproduced") is not True:
            self.ops.remove(self.pod)
            return {"state": "PATCH_REJECTED",
                    "reason": "the original failure no longer reproduces with the patch "
                              "disabled; the evidence for needing this patch is gone",
                    "reproduction": reproduction, "numerical": validation}
        if not usable or min_cosine is None or min_cosine < 0.9999:
            self.ops.remove(self.pod)
            return {"state": "PATCH_REJECTED",
                    "reason": "numerical evidence missing or below threshold",
                    "numerical": validation}

        report = {
            "state": "PATCH_PLACED",
            "mechanism": "post_import_patch",
            "numerical": {
                "reference": "kunlun_ops.speculative_attention",
                "cases": cases,
                "min_cosine_similarity": min_cosine,
            },
            "server_validation": server,
            "reversibility": {
                "backup_path": "/opt/vllm_kunlun/lib/python3.10/site-packages/"
                               "vllm_kunlun/__init__.py.kdp_backup",
                "runtime_switch": "KDP_DECODE_KERNEL",
                "backup_required": True,
                "reproduces_original_failure": True,
            },
            "scope": {"routes_only": "qlen == 1"},
            "limitations": [
                "shape-dynamic span requires --enforce-eager until the span is static",
            ],
            "artifacts": ["numerical_comparison.txt", "placement_report.json", "status.json"],
        }
        contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
        errors = validate_patch_placement(report, contract)
        if errors:
            self.ops.remove(self.pod)
            report["state"] = "PATCH_REJECTED"
            report["validation_errors"] = errors
        self._write(self.out / "placement_report.json", report)
        return report

    def _captured_geometry(self) -> dict[str, Any]:
        """The geometry triage captured for the original failure.

        Without a triage report the reproduction probe cannot run, and a patch
        whose original failure was never reproduced is a rewrite: refuse it.
        """
        if not self.triage_report:
            raise RuntimeError(
                "patch placement needs the triage report: the original failure's "
                "captured geometry is what the reproduction probe replays"
            )
        report = json.loads(self.triage_report.read_text(encoding="utf-8"))
        captured = report.get("captured_arguments") or {}
        geometry = {key: captured.get(key) for key in
                    ("head_num", "kv_head_num", "head_dim", "block_size",
                     "batch_num", "max_context_len")}
        lens = captured.get("context_lens_cpu") or {}
        geometry["context_len"] = lens.get("max", 128)
        if not all(isinstance(geometry[key], (int, float)) for key in
                   ("head_num", "kv_head_num", "head_dim", "block_size", "batch_num")):
            raise RuntimeError(
                f"the triage report's captured arguments are incomplete: {sorted(captured)}"
            )
        return geometry

    @staticmethod
    def _write(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod", required=True)
    parser.add_argument("--contract-instance", type=Path, default=None)
    parser.add_argument("--triage", type=Path, default=None,
                        help="triage report with the captured failure geometry")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    executor = PatchExecutor(args.pod, args.contract_instance,
                             PatchOps(KunlunP800Adapter()), args.out,
                             triage_report=args.triage)
    status = executor.run()
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if status.get("state") == "PATCH_PLACED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
