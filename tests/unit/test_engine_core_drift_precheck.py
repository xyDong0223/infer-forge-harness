"""Engine-core-init drift precheck: seconds per drift, not a 15-minute load.

Run glm52-int-w8a8-p800-001 discovered twelve call-time drifts one full
707 GiB server restart at a time, because nothing exercised the plugin's
engine-facing surface before the weights loaded. These tests build miniature
engine/plugin package pairs — one drifted, one healthy — and pin that the
probe sees in seconds what the run paid a load per piece to see: moved
symbols, renamed functions, re-parameterised calls, removed envs, guarded
probes, opaque signatures. The wiring tests pin the deployment gate: drift
blocks the launch with the fix hint, pass records, a probe crash is a
failure — never a silent pass.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners.deployment_proof import ActionFailed, DeploymentProofRunner  # noqa: E402
from tools.probe import engine_core_drift_precheck as precheck  # noqa: E402


def write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body), encoding="utf-8")


ENGINE_NEW_API = """
    def get_rope(head_size, max_position, rope_parameters=None,
                 is_neox_style=True):
        return ("rope", head_size, max_position, rope_parameters)

    class MultiHeadLatentAttentionWrapper:
        def __init__(self, hidden_size, num_heads, indexer_rope=None):
            self.hidden_size = hidden_size
"""

ENGINE_INDEXER = """
    def split_indexer_prefill_chunks(seq_lens, query_lens, workspace,
                                     max_logits_bytes, request_offset=0):
        return (slice(0, 1), slice(0, 1))

    def flex(*args, **kwargs):
        return None
"""

PLUGIN_DRIFTED = """
    import fakevllm.envs as envs
    from fakevllm.model_executor.rope import get_rope
    from fakevllm.v1.indexer import split_indexer_prefill_chunks, flex
    from fakevllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper

    def build_attention(qk_rope_head_dim, max_position, rope_scaling):
        # Drift: the old get_rope kwargs — engine folded them into
        # rope_parameters. EngineCore would raise TypeError at layer build.
        return get_rope(
            qk_rope_head_dim,
            max_position=max_position,
            base=8000000,
            rope_scaling=rope_scaling,
        )

    def make_wrapper(hidden, heads):
        # Drift: constructor lost a positional and gained a keyword.
        return MultiHeadLatentAttentionWrapper(hidden, heads, None, "extra")

    def chunk(seq_lens, workspace, offset):
        # Drift: old 3-arg signature vs the new 5-parameter one.
        return split_indexer_prefill_chunks(seq_lens, workspace, offset)

    def chunk_flexible(seq_lens, workspace, offset, limit):
        # Unverifiable, not silently green: the engine signature is variadic.
        return flex(seq_lens, workspace, offset, limit)

    def backend_name():
        # Drift: removed from engine envs — crash at VllmConfig creation.
        return envs.VLLM_ATTENTION_BACKEND

    def guarded_backend_name():
        # WARN not DRIFT: the plugin probes for the optional symbol.
        try:
            return envs.VLLM_ATTENTION_BACKEND
        except AttributeError:
            return None
"""

PLUGIN_HEALTHY = """
    import fakevllm.envs as envs
    from fakevllm.model_executor.rope import get_rope
    from fakevllm.v1.indexer import split_indexer_prefill_chunks, flex

    def build_attention(qk_rope_head_dim, max_position, rope_parameters):
        return get_rope(
            qk_rope_head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
        )

    def chunk(seq_lens, query_lens, workspace, max_logits_bytes, offset):
        return split_indexer_prefill_chunks(
            seq_lens, query_lens, workspace, max_logits_bytes, request_offset=offset,
        )
"""


class PrecheckFixtureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self._saved_path = list(sys.path)
        sys.path.insert(0, str(self.tmp))
        self._purge("fakevllm")
        self._purge("fakeplugin")

    def tearDown(self) -> None:
        sys.path[:] = self._saved_path
        self._purge("fakevllm")
        self._purge("fakeplugin")

    @staticmethod
    def _purge(*packages: str) -> None:
        for name in list(sys.modules):
            if name.split(".")[0] in packages:
                del sys.modules[name]

    def build_engine(self) -> None:
        write(self.tmp / "fakevllm/__init__.py", "")
        write(self.tmp / "fakevllm/envs.py", "VLLM_OTHER = 1\n")
        write(self.tmp / "fakevllm/model_executor/__init__.py", "")
        write(self.tmp / "fakevllm/model_executor/rope.py", ENGINE_NEW_API)
        write(self.tmp / "fakevllm/model_executor/layers/__init__.py", "")
        write(self.tmp / "fakevllm/model_executor/layers/mla.py", ENGINE_NEW_API)
        write(self.tmp / "fakevllm/v1/__init__.py", "")
        write(self.tmp / "fakevllm/v1/indexer.py", ENGINE_INDEXER)

    def build_plugin(self, body: str) -> None:
        # Under models/: the registry-scoped surface the gate cares about.
        write(self.tmp / "fakeplugin/__init__.py", "")
        write(self.tmp / "fakeplugin/models/__init__.py", "")
        write(self.tmp / "fakeplugin/models/model.py", body)

    def drifts(self, report: dict) -> dict[str, dict]:
        return {c["symbol"]: c for c in report["checks"] if c["verdict"] == "DRIFT"}

    def test_the_twelve_loads_worth_of_drift_is_seen_in_one_pass(self):
        self.build_engine()
        self.build_plugin(PLUGIN_DRIFTED)

        report = precheck.build_report(
            "fakeplugin", "fakevllm", gate_prefixes=("models/",))

        self.assertEqual(report["state"], "DRIFT")
        self.assertEqual(report["summary"]["drift_gated"], 3)
        drifts = self.drifts(report)
        # get_rope: unexpected keywords (base=, rope_scaling=) — drift 7.
        self.assertIn("fakevllm.model_executor.rope.get_rope", drifts)
        self.assertIn("unexpected keyword", drifts["fakevllm.model_executor.rope.get_rope"]["detail"])
        # MultiHeadLatentAttentionWrapper: constructor arity — drift 1/9.
        self.assertIn("fakevllm.model_executor.layers.mla.MultiHeadLatentAttentionWrapper", drifts)
        # split_indexer_prefill_chunks: 3 positional vs 5 params — drift 3.
        self.assertIn("fakevllm.v1.indexer.split_indexer_prefill_chunks", drifts)
        # envs.VLLM_ATTENTION_BACKEND: removed env — drift 5.
        self.assertIn("fakevllm.envs.VLLM_ATTENTION_BACKEND", drifts)

    def test_a_guarded_probe_is_warn_not_drift(self):
        self.build_engine()
        self.build_plugin(PLUGIN_DRIFTED)

        report = precheck.build_report(
            "fakeplugin", "fakevllm", gate_prefixes=("models/",))

        # Same symbol, two sites: the bare reference is DRIFT (crash at
        # VllmConfig), the try/except AttributeError probe is WARN — the
        # plugin is allowed to test for an optional symbol.
        sites = [c for c in report["checks"]
                 if c["symbol"] == "fakevllm.envs.VLLM_ATTENTION_BACKEND"]
        self.assertEqual(
            sorted(c["verdict"] for c in sites), ["DRIFT", "WARN"],
            msg=f"expected one guarded and one bare site, got {sites}",
        )
        # Two WARNs total: the guarded probe plus the scope note (fixtures
        # pass no --model-config, so the unresolved scope is reported).
        self.assertEqual(report["summary"]["warn"], 2)

    def test_variadic_engine_signatures_are_honestly_unverifiable(self):
        self.build_engine()
        self.build_plugin(PLUGIN_DRIFTED)

        report = precheck.build_report(
            "fakeplugin", "fakevllm", gate_prefixes=("models/",))
        unverifiable = [c for c in report["checks"] if c["verdict"] == "UNVERIFIABLE"]
        self.assertTrue(
            any(c["symbol"] == "fakevllm.v1.indexer.flex" for c in unverifiable),
            msg=f"flex call should be UNVERIFIABLE, got {[c['symbol'] for c in unverifiable]}",
        )

    def test_a_healthy_plugin_passes_with_no_drift(self):
        self.build_engine()
        self.build_plugin(PLUGIN_HEALTHY)

        report = precheck.build_report(
            "fakeplugin", "fakevllm", gate_prefixes=("models/",))

        self.assertEqual(report["state"], "PASS")
        self.assertEqual(report["summary"]["drift"], 0)
        # Deduplicated PASS: two call sites, one entry per symbol.
        rope_pass = [c for c in report["checks"]
                     if c["symbol"] == "fakevllm.model_executor.rope.get_rope"]
        self.assertEqual(len(rope_pass), 1)

    def test_findings_outside_the_gate_are_reported_but_not_blocking(self):
        self.build_engine()
        self.build_plugin(PLUGIN_DRIFTED)

        # Same drifted plugin, no gate prefixes: the findings stay in the
        # report with their DRIFT verdicts, but nothing hard-gates.
        report = precheck.build_report("fakeplugin", "fakevllm", gate_prefixes=())

        self.assertEqual(report["state"], "PASS")
        self.assertGreater(report["summary"]["drift_report_only"], 0)
        self.assertTrue(any(c["gating"] == "report-only" and c["verdict"] == "DRIFT"
                            for c in report["checks"]))

    def test_a_real_missing_symbol_stays_drift_after_settlement(self):
        # A from-imported engine symbol that does not exist: the settlement
        # subprocess cannot resolve it either (the fixture packages are not
        # on its path), so the verdict stands and gates.
        self.build_engine()
        write(self.tmp / "fakeplugin/__init__.py", "")
        write(self.tmp / "fakeplugin/models/__init__.py", "")
        write(self.tmp / "fakeplugin/models/model.py",
              "from fakevllm.v1.indexer import gone_helper\n"
              "\n"
              "\n"
              "def build(x):\n"
              "    return gone_helper(x)\n")

        report = precheck.build_report(
            "fakeplugin", "fakevllm", gate_prefixes=("models/",))

        self.assertEqual(report["state"], "DRIFT")
        self.assertTrue(any(
            c["symbol"] == "fakevllm.v1.indexer.gone_helper"
            and c["verdict"] == "DRIFT"
            for c in report["checks"]))

    def test_an_import_order_artifact_is_downgraded_by_the_disputer(self):
        # Same missing-symbol shape, but the clean interpreter resolves it:
        # the probe's own import order broke the lookup, which is a WARN,
        # never a launch blocker.
        self.build_engine()
        write(self.tmp / "fakeplugin/__init__.py", "")
        write(self.tmp / "fakeplugin/models/__init__.py", "")
        write(self.tmp / "fakeplugin/models/model.py",
              "from fakevllm.v1.indexer import gone_helper\n"
              "\n"
              "\n"
              "def build(x):\n"
              "    return gone_helper(x)\n")

        report = precheck.build_report(
            "fakeplugin", "fakevllm", gate_prefixes=("models/",),
            disputer=lambda paths: {path: "resolved" for path in paths})

        self.assertEqual(report["state"], "PASS")
        entry = next(c for c in report["checks"]
                     if c["symbol"] == "fakevllm.v1.indexer.gone_helper")
        self.assertEqual(entry["verdict"], "WARN")
        self.assertIn("clean engine-first interpreter", entry["detail"])

    def test_an_unimportable_plugin_is_a_drift_not_a_crash(self):
        self.build_engine()
        write(self.tmp / "brokenplugin/__init__.py",
              "import fakevllm.definitely_missing\n")

        report = precheck.build_report("brokenplugin", "fakevllm")

        self.assertEqual(report["state"], "DRIFT")
        self.assertTrue(any("import brokenplugin" in c["id"] for c in report["checks"]))


class _FakeAdapter:
    def __init__(self, stdout: str) -> None:
        self.completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=stdout, stderr=""
        )

    def exec(self, pod, script, timeout=None):  # noqa: ANN001 - duck-typed seam
        return self.completed


class DeploymentPrecheckWiringTest(unittest.TestCase):
    """The deployment gate side: what start_server learns before launching."""

    def make_runner(self, stdout: str) -> DeploymentProofRunner:
        contract = {
            "metadata": {"name": "kdp-001b-service-proof"},
            "execution": {"server_log": "/workspace/server.log"},
            "context": {},
        }
        return DeploymentProofRunner(
            contract=contract,
            adapter=_FakeAdapter(stdout),
            repo_root=ROOT,
            artifact_dir=Path(tempfile.mkdtemp()),
            attach_pod="pod-x",
            phase="service",
        )

    def test_drift_blocks_the_launch_and_names_the_fix(self):
        report = {
            "state": "DRIFT",
            "summary": {"drift": 1, "drift_gated": 1},
            "checks": [{"id": "indexer.py:44 vllm...split_prefill_chunks",
                        "verdict": "DRIFT", "detail": "missing a required argument",
                        "scope": "path", "gating": "gate"}],
        }
        runner = self.make_runner("loading...\n" + json.dumps(report) + "\n")

        with self.assertRaises(ActionFailed) as ctx:
            runner.engine_core_drift_precheck()

        self.assertEqual(ctx.exception.state, "RUNTIME_DRIFT")
        self.assertIn("same pod", ctx.exception.reason)
        self.assertIn("split_prefill_chunks", ctx.exception.reason)
        self.assertEqual(runner.checks["engine_core_drift"], "DRIFT")
        # The full report is evidence, persisted before the gate decision.
        artifact = runner.artifact_dir / "engine_core_drift_precheck.json"
        self.assertTrue(artifact.exists())
        self.assertEqual(
            json.loads(artifact.read_text(encoding="utf-8"))["state"], "DRIFT"
        )

    def test_pass_is_recorded_and_does_not_block(self):
        report = {"state": "PASS", "summary": {"drift": 0, "pass": 41}, "checks": []}
        runner = self.make_runner(json.dumps(report) + "\n")

        runner.engine_core_drift_precheck()  # must not raise

        self.assertEqual(runner.checks["engine_core_drift"], "PASS")
        self.assertTrue(runner.records)

    def test_drift_out_of_path_is_reported_but_does_not_block(self):
        # Drift in other models' files is real, but this deployment does not
        # load them: reported in summary, never the launch blocker.
        report = {
            "state": "PASS",
            "summary": {"drift": 0, "drift_out_of_path": 9},
            "checks": [{"id": "models/gpt_oss.py:63 vllm...get_rope",
                        "verdict": "DRIFT", "detail": "unexpected kwarg",
                        "scope": "other-models", "gating": "report-only"}],
        }
        runner = self.make_runner(json.dumps(report) + "\n")

        runner.engine_core_drift_precheck()  # must not raise

        self.assertEqual(runner.checks["engine_core_drift"], "PASS")

    def test_conditional_branch_findings_are_reported_not_blocking(self):
        # A missing symbol inside a branch that never runs on this platform
        # (HPU arms, flashinfer options): a finding for triage, not a launch
        # blocker — the glm52 pod serves fine while carrying 23 of these.
        report = {
            "state": "PASS",
            "summary": {"drift": 3, "drift_report_only": 3, "drift_gated": 0},
            "checks": [{"id": "mla/common.py:516 vllm...get_mla_dims",
                        "verdict": "DRIFT", "detail": "no importable prefix",
                        "scope": "path", "gating": "report-only"}],
        }
        runner = self.make_runner(json.dumps(report) + "\n")

        runner.engine_core_drift_precheck()  # must not raise

        self.assertEqual(runner.checks["engine_core_drift"], "PASS")

    def test_a_crashed_probe_is_a_failure_not_a_silent_pass(self):
        runner = self.make_runner("Traceback (most recent call last):\n...")

        with self.assertRaises(ActionFailed) as ctx:
            runner.engine_core_drift_precheck()

        self.assertEqual(ctx.exception.state, "RUNTIME_DRIFT")
        self.assertIn("produced no report", ctx.exception.reason)


if __name__ == "__main__":
    unittest.main()
