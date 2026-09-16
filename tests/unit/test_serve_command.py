import unittest

from runtimes.vllm_kunlun import VllmKunlunRuntime


class ServeCommandTests(unittest.TestCase):
    def test_vllm_command_is_built_by_runtime(self):
        runtime = VllmKunlunRuntime.load()
        command = runtime.build_serve_command({
            "port": 8356,
            "path": "/models/demo model",
            "max_model_len": 32768,
            "max_num_seqs": 4,
            "tensor_parallel_size": 8,
            "dtype": "float16",
            "served_model_name": "demo",
            "max_num_batched_tokens": 4096,
        })
        self.assertTrue(command.startswith("python -m vllm.entrypoints.openai.api_server"))
        self.assertIn("--model '/models/demo model'", command)
        self.assertIn("--max-num-batched-tokens 4096", command)


if __name__ == "__main__":
    unittest.main()
