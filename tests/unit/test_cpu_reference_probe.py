import sys
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.probe.cpu_reference_logits_probe import (
    _checkpoint_quantization,
    _visible_accelerator,
)


class CpuReferenceProbeTest(unittest.TestCase):
    def test_xpu_visibility_is_rejected(self):
        torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            xpu=SimpleNamespace(is_available=lambda: True),
        )
        self.assertEqual(_visible_accelerator(torch), "xpu")

    def test_cpu_only_runtime_is_allowed(self):
        torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            xpu=SimpleNamespace(is_available=lambda: False),
        )
        self.assertIsNone(_visible_accelerator(torch))

    def test_unquantized_checkpoint_needs_no_manual_dequantization(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.json").write_text(
                json.dumps({"model_type": "qwen3"}), encoding="utf-8"
            )
            self.assertEqual(_checkpoint_quantization(tmp), "unquantized")

    def test_compressed_tensors_int8_checkpoint_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.json").write_text(json.dumps({
                "quantization_config": {
                    "quant_method": "compressed-tensors",
                    "format": "int-quantized",
                    "config_groups": {
                        "group_0": {
                            "weights": {
                                "type": "int",
                                "num_bits": 8,
                                "strategy": "channel",
                                "symmetric": True,
                            }
                        }
                    },
                }
            }), encoding="utf-8")
            self.assertEqual(
                _checkpoint_quantization(tmp), "compressed-tensors-int8"
            )

    def test_int_quantized_format_without_int8_channel_weights_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.json").write_text(json.dumps({
                "quantization_config": {
                    "quant_method": "compressed-tensors",
                    "format": "int-quantized",
                    "config_groups": {
                        "group_0": {
                            "weights": {
                                "type": "int",
                                "num_bits": 4,
                                "strategy": "channel",
                                "symmetric": True,
                            }
                        }
                    },
                }
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported CPU reference quantization"):
                _checkpoint_quantization(tmp)

    def test_missing_storage_format_is_not_assumed_to_be_integer(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.json").write_text(json.dumps({
                "quantization_config": {
                    "quant_method": "compressed-tensors",
                    "config_groups": {
                        "group_0": {
                            "weights": {
                                "type": "int",
                                "num_bits": 8,
                                "strategy": "channel",
                                "symmetric": True,
                            }
                        }
                    },
                }
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported CPU reference quantization"):
                _checkpoint_quantization(tmp)

    def test_unknown_quantization_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.json").write_text(json.dumps({
                "quantization_config": {"quant_method": "gptq"}
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported CPU reference quantization"):
                _checkpoint_quantization(tmp)


if __name__ == "__main__":
    unittest.main()
