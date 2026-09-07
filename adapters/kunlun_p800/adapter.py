"""Kunlun P800 / Kubernetes adapter.

Isolates every cluster difference behind one object so Tasks and the Runner
never build kubectl invocations themselves. Read operations are unrestricted;
write operations are refused unless the target resource is owned by the
configured operator prefix.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "harness" / "config.yaml"


class SafetyViolation(RuntimeError):
    """Raised when an action would touch a resource outside the owned prefix."""


@dataclass(frozen=True)
class ClusterConfig:
    kubeconfig: str
    namespace: str
    container: str
    resource_prefix: str
    deployment_kind: str
    context: str | None = None

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CONFIG) -> "ClusterConfig":
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        cluster = raw.get("cluster", {})
        missing = [
            key
            for key in ("kubeconfig", "namespace", "container", "resource_prefix", "deployment_kind")
            if not cluster.get(key)
        ]
        if missing:
            raise ValueError(f"harness config is missing cluster.{', cluster.'.join(missing)}")
        return cls(
            kubeconfig=cluster["kubeconfig"],
            namespace=cluster["namespace"],
            container=cluster["container"],
            resource_prefix=cluster["resource_prefix"],
            deployment_kind=cluster["deployment_kind"],
            context=cluster.get("context"),
        )


class KunlunP800Adapter:
    def __init__(self, config: ClusterConfig | None = None, timeout: int = 120) -> None:
        self.config = config or ClusterConfig.load()
        self.timeout = timeout

    # ---- safety -----------------------------------------------------------
    def assert_owned(self, name: str) -> None:
        prefix = self.config.resource_prefix
        if not name or not name.startswith(prefix):
            raise SafetyViolation(
                f"refusing to write to {name!r}: namespace {self.config.namespace} is shared, "
                f"writes are limited to resources starting with {prefix!r}"
            )

    def assert_manifest_owned(self, manifest: dict[str, Any]) -> None:
        self.assert_owned(manifest.get("metadata", {}).get("name", ""))

    # ---- primitives -------------------------------------------------------
    def _base(self) -> list[str]:
        args = ["kubectl", "--kubeconfig", self.config.kubeconfig, "-n", self.config.namespace]
        if self.config.context:
            args += ["--context", self.config.context]
        return args

    def run(self, args: list[str], timeout: int | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [*self._base(), *args],
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout or self.timeout,
        )

    # ---- reads ------------------------------------------------------------
    def can_create_pods(self) -> bool:
        result = self.run(["auth", "can-i", "create", "pods"])
        return result.returncode == 0 and result.stdout.strip().lower() == "yes"

    def get(self, kind: str, name: str | None = None, output: str | None = None) -> subprocess.CompletedProcess[str]:
        args = ["get", kind]
        if name:
            args.append(name)
        if output:
            args += ["-o", output]
        return self.run(args)

    def pod_ready(self, pod: str) -> bool:
        result = self.get(
            "pod", pod, output='jsonpath={.status.conditions[?(@.type=="Ready")].status}'
        )
        return result.returncode == 0 and result.stdout.strip() == "True"

    def logs(self, pod: str, tail: int = 200) -> str:
        result = self.run(["logs", pod, "-c", self.config.container, f"--tail={tail}"])
        return result.stdout + result.stderr

    def exec(self, pod: str, script: str, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
        return self.run(
            ["exec", pod, "-c", self.config.container, "--", "bash", "-lc", script],
            timeout=timeout,
        )

    def http_probe(self, pod: str, path: str, port: int) -> tuple[int, str]:
        """Probe an in-pod endpoint. Returns (status_code, body)."""
        url = f"http://127.0.0.1:{port}{path}"
        result = self.exec(pod, f"curl -sS -o /tmp/probe.out -w '%{{http_code}}' {shlex.quote(url)}; cat /tmp/probe.out")
        text = result.stdout
        code, _, body = text.partition("\n")
        try:
            return int(code.strip()[:3]), body
        except ValueError:
            return 0, text + result.stderr

    # ---- guarded writes ---------------------------------------------------
    def apply(self, manifest_path: Path | str) -> subprocess.CompletedProcess[str]:
        import yaml

        path = Path(manifest_path)
        for doc in yaml.safe_load_all(path.read_text(encoding="utf-8")):
            if doc:
                self.assert_manifest_owned(doc)
        return self.run(["apply", "-f", str(path)], timeout=300)

    def copy_into(self, pod: str, local: Path | str, remote: str) -> subprocess.CompletedProcess[str]:
        self.assert_owned(pod)
        return subprocess.run(
            [
                "kubectl",
                "--kubeconfig",
                self.config.kubeconfig,
                "cp",
                str(local),
                f"{self.config.namespace}/{pod}:{remote}",
                "-c",
                self.config.container,
            ],
            check=False,
            text=True,
            capture_output=True,
            timeout=300,
        )

    def delete(self, kind: str, name: str, confirmed: bool = False) -> subprocess.CompletedProcess[str]:
        self.assert_owned(name)
        if not confirmed:
            raise SafetyViolation("delete requires an explicit human gate: pass confirmed=True")
        return self.run(["delete", kind, name], timeout=300)
