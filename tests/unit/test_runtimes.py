"""Tests for the runtime axis (phase 1): profile, registry, factory, invariants.

Two guards here are the phase's real acceptance:

- `env_prefix` byte-equality with the pre-refactor hardcoded string — every
  generated pod command must be unchanged, and the golden tests in
  test_patch_placement / test_runtime_patches pin the same text.
- the import guard: no module outside `adapters/` imports the concrete
  `adapters.kunlun_p800` — that coupling is what made the harness a
  single-stack monolith.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters import get_hardware  # noqa: E402
from adapters.kunlun_p800.adapter import KunlunP800Adapter  # noqa: E402
from runtimes import RuntimeProfile, VllmKunlunRuntime, default_runtime, get_runtime  # noqa: E402
from runtimes.registry import CATALOG as RUNTIME_CATALOG  # noqa: E402


class TestRuntimeProfile(unittest.TestCase):
    def test_loads_the_deployed_profile(self) -> None:
        profile = RuntimeProfile.load()
        self.assertEqual(profile.framework, "vllm-kunlun")
        self.assertEqual(profile.venv, "/opt/vllm_kunlun")
        self.assertEqual(
            profile.site_packages, "/opt/vllm_kunlun/lib/python3.10/site-packages"
        )
        self.assertEqual(profile.engine_module, "vllm_kunlun")

    def test_missing_field_is_refused(self) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write("framework: vllm-kunlun\nvenv: /opt/x\n")
            path = handle.name
        try:
            with self.assertRaises(ValueError) as ctx:
                RuntimeProfile.load(path)
            self.assertIn("site_packages", str(ctx.exception))
        finally:
            Path(path).unlink()


class TestVllmKunlunRuntime(unittest.TestCase):
    def test_env_prefix_is_byte_identical_to_the_old_hardcode(self) -> None:
        # The string every site used to embed literally. If this changes,
        # every generated pod command changes with it.
        self.assertEqual(
            VllmKunlunRuntime.load().env_prefix(),
            "export VIRTUAL_ENV=/opt/vllm_kunlun PATH=/opt/vllm_kunlun/bin:$PATH",
        )

    def test_site_packages_and_engine_module_come_from_the_profile(self) -> None:
        runtime = VllmKunlunRuntime.load()
        self.assertEqual(runtime.site_packages, "/opt/vllm_kunlun/lib/python3.10/site-packages")
        self.assertEqual(runtime.engine_module, "vllm_kunlun")


class TestRegistry(unittest.TestCase):
    def test_default_runtime_is_the_only_wired_one(self) -> None:
        runtime = default_runtime()
        self.assertIsInstance(runtime, VllmKunlunRuntime)
        # default_runtime is the cached entry point; get_runtime builds fresh
        self.assertIs(default_runtime(), runtime)
        self.assertEqual(get_runtime("vllm-kunlun").profile, runtime.profile)

    def test_unknown_runtime_names_every_declared_entry(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            get_runtime("sglang-kunlun")
        self.assertIn("vllm-kunlun", str(ctx.exception))
        self.assertIn("catalog declares", str(ctx.exception))

    def test_catalog_entry_points_at_an_existing_profile(self) -> None:
        import yaml

        entries = yaml.safe_load(RUNTIME_CATALOG.read_text(encoding="utf-8"))["entries"]
        for entry in entries:
            with self.subTest(entry=entry.get("name")):
                self.assertTrue(
                    (REPO_ROOT / entry["profile"]).exists(),
                    f"profile {entry['profile']} does not exist",
                )
                self.assertIn(
                    "kunlun-p800", entry.get("supported_hardware") or [],
                    "supported_hardware must be non-empty and canonical",
                )


class TestHardwareFactory(unittest.TestCase):
    def test_default_hardware_is_the_p800_adapter(self) -> None:
        self.assertIs(get_hardware(), KunlunP800Adapter)

    def test_unknown_hardware_lists_what_is_registered(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            get_hardware("b200")
        self.assertIn("kunlun-p800", str(ctx.exception))


class TestNoConcreteAdapterImports(unittest.TestCase):
    """The invariant phase 1 exists to establish, kept green forever after."""

    def test_no_module_outside_adapters_imports_the_concrete_adapter(self) -> None:
        offenders = []
        for directory in ("runners", "tools", "engine", "validators", "operations", "cli", "core"):
            for path in (REPO_ROOT / directory).rglob("*.py"):
                if "__pycache__" in str(path):
                    continue
                source = path.read_text(encoding="utf-8")
                if "adapters.kunlun_p800" in source or "from adapters.kunlun_p800" in source:
                    offenders.append(str(path.relative_to(REPO_ROOT)))
        self.assertEqual(offenders, [], "concrete adapter imports must go through adapters.get_hardware()")

    def test_no_module_outside_runtimes_imports_the_concrete_runtime(self) -> None:
        offenders = []
        for directory in ("runners", "tools", "engine", "validators", "adapters", "operations", "cli", "core"):
            for path in (REPO_ROOT / directory).rglob("*.py"):
                if "__pycache__" in str(path):
                    continue
                source = path.read_text(encoding="utf-8")
                if "runtimes.vllm_kunlun" in source or "from runtimes.vllm_kunlun" in source:
                    offenders.append(str(path.relative_to(REPO_ROOT)))
        self.assertEqual(offenders, [], "concrete runtime imports must go through runtimes.get_runtime()")


if __name__ == "__main__":
    unittest.main()
