import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.task_memory import finish_block, load, save, start_block  # noqa: E402


class TaskMemoryTest(unittest.TestCase):
    def test_memory_round_trip_and_atomic_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "task_memory.json"
            memory = load(path, "model_adaptation", "Qwen3-8B")
            start_block(
                memory,
                "lb-001",
                "model_scan",
                {"state_file": "scan_status.json"},
                {"model_tier": "mid", "topology": "single_agent"},
            )
            save(path, memory)
            restored = load(path, "model_adaptation", "Qwen3-8B")
            self.assertEqual(restored["current_loop_block"]["sub_target"], "model_scan")

            finish_block(restored, "SCAN_READY", ["/tmp/scan"])
            save(path, restored)
            final = load(path, "model_adaptation", "Qwen3-8B")
            self.assertIsNone(final["current_loop_block"])
            self.assertEqual(final["completed_loop_blocks"][0]["state"], "SCAN_READY")

    def test_memory_rejects_wrong_subject(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "task_memory.json"
            save(path, load(path, "model_adaptation", "Qwen3-8B"))
            with self.assertRaises(ValueError):
                load(path, "model_adaptation", "MiniMax-M3")


if __name__ == "__main__":
    unittest.main()
