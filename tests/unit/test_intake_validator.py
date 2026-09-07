"""MAT-001 Validator: what may be called an identity fact.

The reference request is the one produced against the cluster on 2026-09-07, so
a regression shows up as a diff against a real artifact.
"""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validators.intake_validator import validate_model_request  # noqa: E402

CONTRACT_PATH = ROOT / "tasks" / "mat-001-model-intake" / "task.yaml"

REQUEST = {
    "api_version": "infer.kunlun/v1alpha1",
    "kind": "ModelRequest",
    "model": {
        "id": "Qwen3-8B",
        "source": "/mnt/cluster/aiak-inference-test/Qwen3-8B",
        "pvc": "rapidfs-baige-v3-pvc",
        "revision": "e962c91b1987a03d133ba4e4712107fa3af4408bb39e75e91a4260a443b93872",
        "revision_method": "structure-sha256 + head/tail 1048576B per shard",
        "shard_count": 5,
        "total_weight_bytes": 16381516776,
        "trust_remote_code_required": False,
    },
    "identity": {
        "architectures": ["Qwen3ForCausalLM"],
        "torch_dtype": "bfloat16",
        "max_position_embeddings": 40960,
        "remote_code": [],
    },
    "target": {
        "hardware": "Kunlunxin-3-P800",
        "vllm_kunlun_ref": "v0.25.1-dev",
        "vllm_kunlun_commit": "3ced109af2510479e1b2eb846a8aca1fbdcbdf62",
    },
}

PROBE = {
    "state": "INTAKE_READY",
    "probed_in": "ephemeral pod: dongxinyu03-mat001-probe-20260907-3",
    "revision": REQUEST["model"]["revision"],
}


class IntakeValidatorTest(unittest.TestCase):
    def setUp(self) -> None:
        import yaml

        self.contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
        self.request = copy.deepcopy(REQUEST)
        self.probe = copy.deepcopy(PROBE)

    def check(self) -> list[str]:
        return validate_model_request(self.request, self.contract, self.probe)

    def test_the_real_qwen3_intake_passes(self):
        self.assertEqual(self.check(), [])

    def test_rules_come_from_the_contract(self):
        """Tightening the contract must tighten the verdict."""
        self.contract["acceptance"]["required_fields"].append("model.license")
        self.assertIn("model.license is required by the contract but is empty", self.check())

    def test_a_runtime_decision_is_rejected(self):
        self.request["model"]["max_model_len"] = 40960
        errors = self.check()
        self.assertTrue(any("max_model_len" in error for error in errors), errors)

    def test_an_unresolved_ref_is_rejected(self):
        self.request["target"]["vllm_kunlun_commit"] = "v0.25.1-dev"
        errors = self.check()
        self.assertTrue(any("resolved 40-char commit" in error for error in errors), errors)

    def test_a_full_hash_claim_is_rejected(self):
        self.request["model"]["revision_method"] = "sha256 over every byte of every shard"
        errors = self.check()
        self.assertTrue(any("claims full coverage" in error for error in errors), errors)

    def test_remote_code_without_trust_flag_is_rejected(self):
        self.request["identity"]["remote_code"] = ["configuration_minimax_m2.MiniMaxConfig"]
        errors = self.check()
        self.assertTrue(any("own modelling code" in error for error in errors), errors)

    def test_a_request_whose_digest_disagrees_with_the_probe_is_rejected(self):
        self.probe["revision"] = "0" * 64
        errors = self.check()
        self.assertTrue(any("does not match the probe" in error for error in errors), errors)

    def test_evidence_is_mandatory(self):
        errors = validate_model_request(self.request, self.contract, None)
        self.assertTrue(any("must accompany the request" in error for error in errors), errors)

    def test_a_rejected_checkpoint_cannot_pass_as_an_identity_fact(self):
        self.probe["state"] = "UNSUPPORTED_FORMAT"
        errors = self.check()
        self.assertTrue(any("not INTAKE_READY" in error for error in errors), errors)


if __name__ == "__main__":
    unittest.main()
