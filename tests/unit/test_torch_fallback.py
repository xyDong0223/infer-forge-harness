import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.torch_fallback import collect_gaps, extract_operators, prepare  # noqa: E402


class TorchFallbackTest(unittest.TestCase):
    def write_json(self, path: Path, payload) -> Path:
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_extract_operators_handles_every_artifact_shape(self):
        self.assertEqual(
            extract_operators({"operator": "a", "symbol": "b", "name": "c"}),
            ["a", "b", "c"],
        )
        self.assertEqual(
            extract_operators({"class": "CAPABILITY_MISSING", "axis": "swiglu_oai"}),
            ["swiglu_oai"],
        )
        self.assertEqual(
            extract_operators({"operators": ["x"], "gaps": [{"name": "y"}]}),
            ["x", "y"],
        )
        # One request per operator, not per mention.
        self.assertEqual(
            extract_operators({"gaps": [{"name": "y"}, {"name": "y"}]}),
            ["y"],
        )

    def test_apply_dispatches_one_request_per_operator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gaps = self.write_json(
                root / "gaps.json",
                {"gaps": [{"operator": "foo"}, {"symbol": "bar"}]},
            )
            args = type("Args", (), {"gaps": gaps, "failure": [], "operator": []})()
            collected = collect_gaps(args)
            report = prepare(collected, root / "out", "Model")
            self.assertEqual(report["state"], "FALLBACK_APPLIED")
            self.assertEqual(report["operators"], ["foo", "bar"])
            self.assertEqual(len(report["request_ids"]), 2)
            self.assertTrue((root / "out/requests/Model-op-001.json").exists())

    def test_shim_child_writes_a_brief_that_names_the_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = prepare(
                [{"name": "kunlun_mla", "class": "TORCH_SHIM", "source": "torch_fallback"}],
                root / "child",
                "GLM-5.2",
            )
            self.assertEqual(report["request_ids"], ["GLM-5.2-op-001"])
            brief = (root / "child/shim_brief.md")
            self.assertTrue(brief.exists())
            text = brief.read_text(encoding="utf-8")
            self.assertIn("kunlun_mla", text)
            self.assertIn("GLM-5.2-op-001", text)
            self.assertIn("torch", text.lower())

    def test_the_fan_out_cli_contract_the_runner_relies_on(self):
        # list-operators prints one JSON array; each `shim` child owns a
        # directory; `aggregate` merges the children into one node status.
        import subprocess

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gaps = self.write_json(
                root / "gaps.json", {"gaps": [{"operator": "foo"}, {"operator": "bar"}]}
            )
            listed = subprocess.run(
                [sys.executable, str(ROOT / "tools/torch_fallback.py"), "list-operators",
                 "--gaps", str(gaps), "--subject", "Model"],
                cwd=ROOT, text=True, capture_output=True,
            )
            self.assertEqual(listed.returncode, 0, listed.stderr)
            operators = json.loads(listed.stdout.strip().splitlines()[-1])
            self.assertEqual(operators, ["foo", "bar"])

            children = []
            for operator in operators:
                child = root / operator
                shim = subprocess.run(
                    [sys.executable, str(ROOT / "tools/torch_fallback.py"), "shim",
                     "--operator", operator, "--subject", "Model", "--out", str(child)],
                    cwd=ROOT, text=True, capture_output=True,
                )
                self.assertEqual(shim.returncode, 0, shim.stderr)
                self.assertTrue((child / "shim_brief.md").exists())
                children.append(child)

            out = root / "node"
            aggregate = subprocess.run(
                [sys.executable, str(ROOT / "tools/torch_fallback.py"), "aggregate",
                 "--out", str(out)]
                + sum([["--child", str(child)] for child in children], []),
                cwd=ROOT, text=True, capture_output=True,
            )
            self.assertEqual(aggregate.returncode, 0, aggregate.stderr)
            status = json.loads((out / "fallback_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "FALLBACK_APPLIED")
            self.assertEqual(sorted(status["operators"]), ["bar", "foo"])
            self.assertEqual(len(status["request_ids"]), 2)


if __name__ == "__main__":
    unittest.main()
