import unittest

from runtimes.vllm_kunlun import VllmKunlunRuntime


class RuntimeFingerprintTests(unittest.TestCase):
    def test_runtime_owns_fingerprint_and_fallback_markers(self):
        runtime = VllmKunlunRuntime.load()
        self.assertIn("vllm-kunlun", runtime.environment_fingerprint_command("/workspace"))
        self.assertIn("fallback to cpu", runtime.fallback_markers())

    def test_cuda_names_are_not_fallback_evidence_on_kunlun(self):
        # The XPU is reached through the torch.cuda API, so a healthy P800 log
        # may legitimately mention cuda. Flagging it failed correct deployments.
        runtime = VllmKunlunRuntime.load()
        self.assertNotIn("using cuda backend", runtime.fallback_markers())
        for marker in runtime.fallback_markers():
            self.assertNotIn("cuda", marker)


if __name__ == "__main__":
    unittest.main()
