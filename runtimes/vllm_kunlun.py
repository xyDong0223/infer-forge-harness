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
import shlex

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

    def installer_name(self) -> str:
        return "install_vllm_kunlun.sh"

    def installer_path(self, repo_root: Path) -> Path:
        return repo_root / "tools" / self.installer_name()

    def import_check_command(self) -> str:
        return 'python3 -c "import torch, vllm, vllm_kunlun"'

    def import_version_command(self) -> str:
        return (
            'python3 -c "import json, torch, vllm, vllm_kunlun; '
            "print(json.dumps({'torch': torch.__version__, "
            "'vllm': vllm.__version__, "
            "'vllm_kunlun': getattr(vllm_kunlun, '__version__', 'unknown')}))\""
        )

    def package_query_command(self) -> str:
        return "uv pip list | grep -iE '^(vllm|vllm-kunlun|torch|kunlun-ops|xspeedgate-ops) '"

    def worktree_check_command(self, workdir: str) -> str:
        return f"test -f {workdir}/vLLM-Kunlun/setup_env.sh"

    def worktree_revision_command(self, workdir: str) -> str:
        return f"cd {workdir}/vLLM-Kunlun && git rev-parse HEAD"

    def fallback_markers(self) -> tuple[str, ...]:
        """Log signals that mean the service left the vendor device path.

        CUDA names are deliberately absent. On Kunlun the XPU is exposed
        through the torch.cuda API (torch_xmlir maps it end to end, which is
        why `is_cuda_alike()` must answer True), so a cuda-sounding line in a
        healthy P800 server log is the native path, not a fallback. Treating
        it as one failed correct deployments with UNEXPECTED_FALLBACK. Only
        host/CPU escapes count here.
        """
        return ("falling back to", "fallback to cpu")

    def environment_fingerprint_command(self, workdir: str) -> str:
        return (
            f"{self.package_query_command()}; "
            f"echo '## {self.profile.framework} commit'; "
            f"{self.worktree_revision_command(workdir)}; "
            "echo '## device'; xpu_smi -L"
        )

    def build_serve_command(self, server: dict) -> str:
        """Build the legacy-compatible vLLM OpenAI server command."""
        required = (
            "port", "path", "max_model_len", "max_num_seqs",
            "tensor_parallel_size", "dtype", "served_model_name",
        )
        missing = [key for key in required if server.get(key) in (None, "")]
        if missing:
            raise ValueError(f"server configuration is missing: {', '.join(missing)}")
        values = [
            "python -m vllm.entrypoints.openai.api_server",
            "--host 0.0.0.0",
            f"--port {shlex.quote(str(server['port']))}",
            f"--model {shlex.quote(str(server['path']))}",
            "--trust-remote-code",
            f"--max-model-len {shlex.quote(str(server['max_model_len']))}",
            f"--max-num-seqs {shlex.quote(str(server['max_num_seqs']))}",
            f"--tensor-parallel-size {shlex.quote(str(server['tensor_parallel_size']))}",
            f"--dtype {shlex.quote(str(server['dtype']))}",
            f"--served-model-name {shlex.quote(str(server['served_model_name']))}",
        ]
        optional = (
            ("max_num_batched_tokens", "--max-num-batched-tokens"),
            ("block_size", "--block-size"),
            ("gpu_memory_utilization", "--gpu-memory-utilization"),
        )
        for key, flag in optional:
            if server.get(key) not in (None, ""):
                values.append(f"{flag} {shlex.quote(str(server[key]))}")
        return " ".join(values)
