import unittest

from core.paths import REPO_ROOT
from runtimes.vllm_kunlun import VllmKunlunRuntime


class RuntimeAdapterCommandTests(unittest.TestCase):
    def test_vllm_runtime_owns_runtime_specific_commands(self):
        runtime = VllmKunlunRuntime.load()
        self.assertIn("vllm_kunlun", runtime.import_check_command())
        self.assertIn("install_vllm_kunlun.sh", runtime.installer_name())
        self.assertIn("git rev-parse HEAD", runtime.worktree_revision_command("/workspace"))

    def test_runtime_installer_is_owned_by_the_runtime_package(self):
        runtime = VllmKunlunRuntime.load()
        installer = runtime.installer_path(REPO_ROOT)
        self.assertEqual(installer.parent, REPO_ROOT / "runtimes" / "scripts")
        self.assertTrue(installer.is_file())


if __name__ == "__main__":
    unittest.main()
