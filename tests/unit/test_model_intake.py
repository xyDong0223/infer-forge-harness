"""MAT-001 intake: fingerprint semantics and the ephemeral-delete exemption.

The probe is stdlib-only, so it runs directly on this host against temporary
directories — no cluster needed to test what it accepts and rejects.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.kunlun_p800.adapter import KunlunP800Adapter, SafetyViolation  # noqa: E402
from tools.model_intake import IntakeFailed, build_request, resolve_stack_commit  # noqa: E402

PROBE = ROOT / "tools" / "probe" / "model_fingerprint_probe.py"


def run_probe(path: Path) -> dict:
    result = subprocess.run(
        [sys.executable, str(PROBE), str(path)], text=True, capture_output=True, check=True
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def make_checkpoint(root: Path, shard_bytes: int = 4096, config: dict | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps(config or {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
                              "torch_dtype": "bfloat16", "max_position_embeddings": 40960}),
        encoding="utf-8",
    )
    (root / "model.safetensors").write_bytes(b"w" * shard_bytes)
    return root


class ProbeAcceptanceTest(unittest.TestCase):
    def test_servable_checkpoint_is_fingerprinted(self):
        with tempfile.TemporaryDirectory() as tmp:
            probe = run_probe(make_checkpoint(Path(tmp) / "model"))
            self.assertEqual(probe["state"], "INTAKE_READY")
            self.assertEqual(probe["shard_count"], 1)
            self.assertEqual(probe["identity"]["architectures"], ["Qwen3ForCausalLM"])
            self.assertFalse(probe["trust_remote_code_required"])

    def test_mcore_checkpoint_is_rejected_not_fingerprinted(self):
        """Model-named mcore dirs sit next to servable weights on this PVC."""
        with tempfile.TemporaryDirectory() as tmp:
            root = make_checkpoint(Path(tmp) / "Qwen3_8B_mcore_tp2pp1")
            (root / "latest_checkpointed_iteration.txt").write_text("7", encoding="utf-8")
            probe = run_probe(root)
            self.assertEqual(probe["state"], "UNSUPPORTED_FORMAT")
            self.assertIn("mcore", probe["reason"])

    def test_directory_without_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "weights"
            root.mkdir()
            (root / "model.safetensors").write_bytes(b"w" * 16)
            self.assertEqual(run_probe(root)["state"], "UNSUPPORTED_FORMAT")

    def test_config_without_weights_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "empty"
            root.mkdir()
            (root / "config.json").write_text("{}", encoding="utf-8")
            probe = run_probe(root)
            self.assertEqual(probe["state"], "UNSUPPORTED_FORMAT")
            self.assertIn("no weight shard", probe["reason"])

    def test_missing_path_is_unavailable(self):
        self.assertEqual(run_probe(Path("/nonexistent/model"))["state"], "MODEL_UNAVAILABLE")

    def test_remote_code_is_part_of_the_revision(self):
        """MiniMax ships modelling code; changing it must change the revision."""
        with tempfile.TemporaryDirectory() as tmp:
            root = make_checkpoint(
                Path(tmp) / "m", config={"model_type": "minimax_m2", "architectures": ["X"],
                                         "auto_map": {"AutoConfig": "configuration_minimax_m2.MiniMaxConfig"}}
            )
            (root / "configuration_minimax_m2.py").write_text("VERSION = 1\n", encoding="utf-8")
            first = run_probe(root)
            self.assertTrue(first["trust_remote_code_required"])
            self.assertIn("configuration_minimax_m2.py", first["structure"])
            (root / "configuration_minimax_m2.py").write_text("VERSION = 2\n", encoding="utf-8")
            self.assertNotEqual(run_probe(root)["revision"], first["revision"])


class RevisionStabilityTest(unittest.TestCase):
    def test_same_content_gives_the_same_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = run_probe(make_checkpoint(Path(tmp) / "a"))
            b = run_probe(make_checkpoint(Path(tmp) / "b"))
            self.assertEqual(a["revision"], b["revision"])

    def test_a_truncated_shard_changes_the_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            full = run_probe(make_checkpoint(Path(tmp) / "full", shard_bytes=8192))
            short = run_probe(make_checkpoint(Path(tmp) / "short", shard_bytes=4096))
            self.assertNotEqual(full["revision"], short["revision"])


class RequestShapeTest(unittest.TestCase):
    class Args:
        model_id = "Qwen3-8B"
        model_path = "/mnt/cluster/aiak-inference-test/Qwen3-8B"
        pvc = "rapidfs-baige-v3-pvc"
        hardware = "Kunlunxin-3-P800"
        stack_ref = "v0.25.1-dev"
        attempt_id = "20260907-3"

    def setUp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.probe = run_probe(make_checkpoint(Path(tmp) / "model"))

    def test_intake_records_identity_and_the_pinned_commit(self):
        request = build_request(self.Args(), self.probe, "3ced109af2510479e1b2eb846a8aca1fbdcbdf62")
        self.assertEqual(request["target"]["vllm_kunlun_commit"][:12], "3ced109af251")
        self.assertEqual(request["model"]["revision"], self.probe["revision"])
        self.assertIn("architectures", request["identity"])

    def test_intake_records_no_runtime_decision(self):
        """dtype/TP/max_model_len are decisions made later, not intake facts."""
        request = build_request(self.Args(), self.probe, "0" * 40)
        flat = json.dumps(request)
        for forbidden in ('"dtype"', "tensor_parallel_size", "max_model_len"):
            self.assertNotIn(forbidden, flat)


class EphemeralDeleteTest(unittest.TestCase):
    class FakeAdapter(KunlunP800Adapter):
        def __init__(self, labels: dict[str, str]) -> None:  # no cluster, no config
            self.labels = labels
            self.deleted: list[str] = []

        @property
        def config(self):  # type: ignore[override]
            class Config:
                resource_prefix = "dongxinyu03-"
                namespace = "pd-test"

            return Config()

        def get(self, kind, name=None, output=None):
            return subprocess.CompletedProcess([], 0, json.dumps(self.labels), "")

        def run(self, args, timeout=None):
            self.deleted.append(args[2])
            return subprocess.CompletedProcess([], 0, "deleted", "")

    def labels(self, **overrides: str) -> dict[str, str]:
        base = {
            "infer.kunlun/ephemeral": "true",
            "infer.kunlun/task-id": "mat-001-model-intake",
            "infer.kunlun/attempt-id": "20260907-3",
        }
        base.update(overrides)
        return base

    def test_this_attempts_probe_pod_is_deleted(self):
        adapter = self.FakeAdapter(self.labels())
        adapter.delete_ephemeral("pod", "dongxinyu03-mat001-probe-20260907-3",
                                 "mat-001-model-intake", "20260907-3")
        self.assertEqual(adapter.deleted, ["dongxinyu03-mat001-probe-20260907-3"])

    def test_a_resource_without_the_permit_label_is_refused(self):
        for labels in (self.labels(**{"infer.kunlun/ephemeral": "false"}),
                       {"app": "vllm"},
                       self.labels(**{"infer.kunlun/attempt-id": "someone-else"})):
            adapter = self.FakeAdapter(labels)
            with self.subTest(labels=labels), self.assertRaises(SafetyViolation):
                adapter.delete_ephemeral("pod", "dongxinyu03-mat001-probe-20260907-3",
                                         "mat-001-model-intake", "20260907-3")
            self.assertEqual(adapter.deleted, [])

    def test_another_operators_resource_is_refused_before_any_label_read(self):
        adapter = self.FakeAdapter(self.labels())
        with self.assertRaises(SafetyViolation):
            adapter.delete_ephemeral("pod", "gjj-vllm-deepseek-v4-decode-rb",
                                     "mat-001-model-intake", "20260907-3")


class StackPinTest(unittest.TestCase):
    def test_an_unknown_ref_is_a_contract_error_not_a_network_error(self):
        with self.assertRaises(IntakeFailed) as ctx:
            resolve_stack_commit("no-such-ref-xyz", repo=str(ROOT), attempts=1)
        self.assertEqual(ctx.exception.state, "CONTRACT_INVALID")


if __name__ == "__main__":
    unittest.main()
