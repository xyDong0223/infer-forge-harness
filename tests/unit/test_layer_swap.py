"""Layer-swap: the single-layer isolation method, as a tool not a one-off.

Run glm52-int-w8a8-p800-001's correctness bisection was a hand-built probe
per question. These tests pin the standardized tool's contract on a
fabricated miniature checkpoint: the swap keeps real weights only where
asked, fabricates inert int8 experts elsewhere, the comparison reports
relL2 with a verdict, and the protocol guards (same tokens, shared stages)
refuse silently-corrupting comparisons.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
save_file = pytest.importorskip("safetensors.torch").save_file

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.probe import layer_swap  # noqa: E402


def make_checkpoint(root: Path, depth: int = 3) -> None:
    """A miniature HF-layout MoE checkpoint: 3 layers, layer 0 dense."""
    hidden, inter, experts = 32, 16, 4
    config = {
        "num_hidden_layers": depth, "hidden_size": hidden,
        "moe_intermediate_size": inter, "n_shared_experts": 1,
        "n_routed_experts": experts, "first_k_dense_replace": 2,
        "num_nextn_predict_layers": 1,
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")

    tensors = {"model.embed_tokens.weight": torch.randn(hidden, hidden),
               "model.norm.weight": torch.ones(hidden),
               "lm_head.weight": torch.randn(hidden, hidden)}
    generator = torch.Generator().manual_seed(3)
    for layer in range(depth):
        prefix = f"model.layers.{layer}."
        tensors[prefix + "input_layernorm.weight"] = torch.ones(hidden)
        tensors[prefix + "self_attn.q_proj.weight"] = torch.randn(hidden, hidden, generator=generator)
        tensors[prefix + "self_attn.o_proj.weight"] = torch.randn(hidden, hidden, generator=generator)
        tensors[prefix + "post_attention_layernorm.weight"] = torch.ones(hidden)
        if layer < 2:  # dense head
            tensors[prefix + "mlp.gate_proj.weight"] = torch.randn(inter, hidden, generator=generator)
            tensors[prefix + "mlp.down_proj.weight"] = torch.randn(hidden, inter, generator=generator)
        else:  # MoE layer
            tensors[prefix + "mlp.gate.weight"] = torch.randn(experts, hidden, generator=generator)
            for expert in range(experts):
                for name, out_f in (("gate_proj", inter), ("up_proj", inter), ("down_proj", hidden)):
                    tensors[prefix + f"mlp.experts.{expert}.{name}.weight"] = torch.randint(
                        -8, 8, (out_f, hidden), generator=generator, dtype=torch.int8)
    save_file(tensors, str(root / "model.safetensors"))
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0},
                    "weight_map": {k: "model.safetensors" for k in tensors}}),
        encoding="utf-8")


class BuildSwapTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.source = self.tmp / "source"
        self.source.mkdir()
        make_checkpoint(self.source)

    def test_real_layers_stay_real_and_dummies_are_fabricated(self):
        manifest = layer_swap.build_swap(
            self.source, real_layers=[0], depth=4, out_dir=self.tmp / "swap")

        self.assertEqual(manifest["real_layers"], [0])
        self.assertEqual(manifest["dummy_layers"], [1, 2, 3])
        self.assertEqual(manifest["config_surgery"]["num_hidden_layers"], 4)
        self.assertEqual(manifest["config_dropped"], ["num_nextn_predict_layers"])

        config = json.loads((self.tmp / "swap" / "config.json").read_text())
        self.assertEqual(config["num_hidden_layers"], 4)
        self.assertNotIn("num_nextn_predict_layers", config)

        swap = layer_swap.Checkpoint(self.tmp / "swap")
        real_q = swap.get("model.layers.0.self_attn.q_proj.weight")
        source_q = layer_swap.Checkpoint(self.source).get(
            "model.layers.0.self_attn.q_proj.weight")
        self.assertTrue(torch.equal(real_q, source_q))
        # Fabricated expert: int8 with the inert scale, not a copy of anything.
        expert = swap.get("model.layers.1.mlp.experts.0.gate_proj.weight")
        self.assertEqual(expert.dtype, torch.int8)
        scale = swap.get("model.layers.1.mlp.experts.0.gate_proj.weight_scale")
        self.assertTrue(torch.allclose(scale, torch.full((16,), 1e-3)))
        self.assertEqual(int(config["n_routed_experts"]), 8)
        # Real global weights kept, donor attention cloned into dummies.
        self.assertIn("model.embed_tokens.weight", swap.index)
        self.assertIn("model.layers.2.self_attn.q_proj.weight", swap.index)
        self.assertNotIn("model.layers.3.mlp.experts.0.gate_proj.weight_scale_inv",
                         swap.index)

    def test_a_moe_donor_drops_its_experts_before_fabrication(self):
        # Real layer 2 (MoE) + dummies: the donor's expert keys are replaced,
        # never mixed with the fabricated recipe.
        manifest = layer_swap.build_swap(
            self.source, real_layers=[2], depth=4, out_dir=self.tmp / "swap2",
            dummy_experts=4)
        swap = layer_swap.Checkpoint(self.tmp / "swap2")
        # Layer 2's own expert stays real (it is a real layer)...
        self.assertIn("model.layers.2.mlp.experts.0.gate_proj.weight", swap.index)
        # ...and the dummy layer 0 does not inherit donor MoE keys by copy.
        fabricated = swap.get("model.layers.0.mlp.gate.weight")
        self.assertEqual(fabricated.shape[0], 4)
        self.assertIn("model.layers.1.mlp.shared_experts.down_proj.weight_scale",
                      swap.index)

    def test_real_layers_beyond_depth_are_refused(self):
        with self.assertRaises(SystemExit):
            layer_swap.build_swap(self.source, real_layers=[5], depth=4,
                                  out_dir=self.tmp / "bad")


class CompareTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def leg(self, name: str, tensors: dict, tokens: list[int]) -> Path:
        out = self.tmp / name
        out.mkdir()
        torch.save(tensors, out / "capture.pt")
        (out / "capture_manifest.json").write_text(json.dumps({
            "tokens": tokens, "next_token": 1,
            "stages": {k: list(v.shape) for k, v in tensors.items()},
        }), encoding="utf-8")
        return out

    def test_close_legs_pass_and_report_the_worst_stage(self):
        base = {"layers.0.in": torch.randn(6, 32),
                "layers.0.effective_out": torch.randn(6, 32)}
        noisy = {k: v + 0.001 * torch.randn_like(v) for k, v in base.items()}
        report = layer_swap.compare_captures(
            self.leg("a", base, [1, 2, 3]), self.leg("b", noisy, [1, 2, 3]),
            tolerance=0.05)
        self.assertEqual(report["state"], "PASS")
        self.assertLess(report["worst_rel_l2"], 0.05)
        self.assertEqual(len([s for s in report["stages"] if "rel_l2" in s]), 2)

    def test_a_broken_stage_fails_the_report(self):
        base = {"layers.0.in": torch.randn(6, 32)}
        broken = {"layers.0.in": -base["layers.0.in"]}
        report = layer_swap.compare_captures(
            self.leg("a", base, [1]), self.leg("b", broken, [1]),
            tolerance=0.05)
        self.assertEqual(report["state"], "FAIL")
        self.assertGreater(report["stages"][0]["rel_l2"], 1.0)

    def test_different_tokens_are_incomparable_not_wrong(self):
        base = {"layers.0.in": torch.randn(6, 32)}
        report = layer_swap.compare_captures(
            self.leg("a", base, [1, 2, 3]), self.leg("b", base, [4, 5, 6]))
        self.assertEqual(report["state"], "INCOMPARABLE")
        self.assertIn("different tokens", report["reason"])

    def test_shape_mismatch_is_named_not_crashed(self):
        report = layer_swap.compare_captures(
            self.leg("a", {"layers.0.in": torch.randn(6, 32)}, [1]),
            self.leg("b", {"layers.0.in": torch.randn(7, 32)}, [1]))
        self.assertEqual(report["state"], "INCOMPARABLE")
        self.assertEqual(report["stages"][0]["verdict"], "INCOMPARABLE")

    def test_no_shared_stages_is_incomparable(self):
        report = layer_swap.compare_captures(
            self.leg("a", {"layers.0.in": torch.randn(6, 32)}, [1]),
            self.leg("b", {"layers.9.in": torch.randn(6, 32)}, [1]))
        self.assertEqual(report["state"], "INCOMPARABLE")


if __name__ == "__main__":
    unittest.main()
