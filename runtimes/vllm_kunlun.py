"""Runtime profile for the vllm-kunlun stack on a prepared pod.

A runtime is the inference framework stack a service runs on (vllm-kunlun
today, sglang-kunlun planned). Everything here is *static environment
identity* — where the venv lives, where site-packages are, what the engine
module is called. Behavioural differences (launch command, readiness shape,
in-process model capture, drift precheck) grow on this class in phase 2.

The values come from `config/profiles/p800-vllm-kunlun.yaml`, never from
code: 19 files used to hardcode `/opt/vllm_kunlun` and drifted freely.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_PROFILE = (
    Path(__file__).resolve().parents[1] / "config" / "profiles" / "p800-vllm-kunlun.yaml"
)


@dataclass(frozen=True)
class RuntimeProfile:
    """Static environment identity of one runtime stack install."""

    framework: str
    venv: str
    site_packages: str
    engine_module: str

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PROFILE) -> "RuntimeProfile":
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        missing = [
            key
            for key in ("framework", "venv", "site_packages", "engine_module")
            if not raw.get(key)
        ]
        if missing:
            raise ValueError(f"runtime profile {path} is missing: {', '.join(missing)}")
        return cls(
            framework=str(raw["framework"]),
            venv=str(raw["venv"]),
            site_packages=str(raw["site_packages"]),
            engine_module=str(raw["engine_module"]),
        )


class VllmKunlunRuntime:
    """The vllm-kunlun runtime. Phase-1 surface: environment strings only."""

    def __init__(self, profile: RuntimeProfile) -> None:
        self.profile = profile

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PROFILE) -> "VllmKunlunRuntime":
        return cls(RuntimeProfile.load(path))

    def env_prefix(self) -> str:
        """Shell exports that put the pod shell inside this runtime's venv.

        Returned without a trailing separator so call sites keep their own
        `"; "`, `" && "`, or space exactly as before — the generated command
        text must stay byte-identical (the golden tests in
        test_patch_placement / test_runtime_patches pin it).
        """
        venv = self.profile.venv
        return f"export VIRTUAL_ENV={venv} PATH={venv}/bin:$PATH"

    @property
    def site_packages(self) -> str:
        return self.profile.site_packages

    @property
    def engine_module(self) -> str:
        return self.profile.engine_module
