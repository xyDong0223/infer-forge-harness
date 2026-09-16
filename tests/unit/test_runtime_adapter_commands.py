import unittest

from runtimes.vllm_kunlun import VllmKunlunRuntime


class RuntimeAdapterCommandTests(unittest.TestCase):
    def test_vllm_runtime_owns_runtime_specific_commands(self):
        runtime = VllmKunlunRuntime.load()
        self.assertIn("vllm_kunlun", runtime.import_check_command())
        self.assertIn("install_vllm_kunlun.sh", runtime.installer_name())
        self.assertIn("git rev-parse HEAD", runtime.worktree_revision_command("/workspace"))


if __name__ == "__main__":
    unittest.main()
