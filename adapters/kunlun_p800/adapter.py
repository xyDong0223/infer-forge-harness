"""Kunlun P800 / Kubernetes adapter.

Isolates every cluster difference behind one object so Tasks and the Runner
never build kubectl invocations themselves. Read operations are unrestricted;
write operations are refused unless the target resource is owned by the
configured operator prefix.
"""

from __future__ import annotations

import os
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
        # Credential paths come from the environment, never from the repository.
        kubeconfig = os.path.expandvars(str(cluster.get("kubeconfig", "")))
        if "${" in kubeconfig or not kubeconfig:
            raise ValueError(
                "cluster.kubeconfig is unresolved: export KUBECONFIG before running a task"
            )
        if not Path(kubeconfig).exists():
            raise ValueError(f"kubeconfig does not exist: {kubeconfig}")
        missing = [
            key
            for key in ("namespace", "container", "resource_prefix", "deployment_kind")
            if not cluster.get(key)
        ]
        if missing:
            raise ValueError(f"harness config is missing cluster.{', cluster.'.join(missing)}")
        return cls(
            kubeconfig=kubeconfig,
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
            env=self._env(),
        )

    @staticmethod
    def _env() -> dict[str, str]:
        """kubectl must never go through the external HTTP proxy.

        Reaching GitHub in the same process requires http(s)_proxy to be set, and
        with it exported the API server call fails with `Unable to connect to the
        server: EOF` — the proxy accepts the CONNECT and then drops it.
        """
        return {
            key: value
            for key, value in os.environ.items()
            if key.lower() not in {"http_proxy", "https_proxy", "all_proxy"}
        }

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

    def xpu_smi(self, pod: str) -> list[dict[str, int]]:
        """Per-card memory from `xpu_smi -m`, the P800 counterpart of nvidia-smi."""
        result = self.exec(pod, "xpu_smi -m")
        if result.returncode != 0:
            raise RuntimeError(f"xpu_smi failed on {pod}: {result.stderr.strip()}")
        cards = self.parse_xpu_smi(result.stdout)
        if not cards:
            raise RuntimeError(f"xpu_smi returned no parsable card on {pod}")
        return cards

    @staticmethod
    def parse_xpu_smi(text: str) -> list[dict[str, int]]:
        """Parse `xpu_smi -m` output.

        Machine-readable columns are positional, as documented by `xpu_smi -h`:
        index 2 is dev_id, 17 is Memory_used in MB and 18 is Memory_size in MB.
        Only these three are read — column 21 (`model`) is an unquoted string
        with a space in it ("P800 OAM"), so nothing after it can be addressed by
        position.
        """
        cards: list[dict[str, int]] = []
        for line in text.splitlines():
            fields = line.split()
            if len(fields) < 19:
                continue
            try:
                index, used, total = int(fields[2]), int(fields[17]), int(fields[18])
            except ValueError:
                continue
            cards.append(
                {"index": index, "used_mib": used, "total_mib": total, "free_mib": total - used}
            )
        return sorted(cards, key=lambda card: card["index"])

    @staticmethod
    def as_nvidia_smi_csv(cards: list[dict[str, int]]) -> str:
        """Render cards the way `nvidia-smi --format=csv,noheader` would.

        Lets the vendored capacity analyzer cross-validate a P800 deployment
        without teaching it about XPUs.
        """
        return "".join(
            f"{card['index']}, {card['used_mib']} MiB, {card['free_mib']} MiB\n" for card in cards
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
            env=self._env(),
        )

    def delete(self, kind: str, name: str, confirmed: bool = False) -> subprocess.CompletedProcess[str]:
        self.assert_owned(name)
        if not confirmed:
            raise SafetyViolation("delete requires an explicit human gate: pass confirmed=True")
        return self.run(["delete", kind, name], timeout=300)

    def delete_ephemeral(
        self, kind: str, name: str, task_id: str, attempt_id: str
    ) -> subprocess.CompletedProcess[str]:
        """Delete a throwaway resource this run created, without a human gate.

        The human gate on `delete` exists because the namespace holds other
        engineers' live services. A probe Pod is different: it is created and
        removed inside one Task, and leaving it behind wastes shared quota. The
        exemption is therefore narrowed by what the cluster itself reports, not
        by what the caller claims — the resource must be owner-prefixed, carry
        `infer.kunlun/ephemeral=true`, and match this attempt's labels. Anything
        else, including a missing label, falls back to the gate.
        """
        self.assert_owned(name)
        result = self.get(kind, name, output="jsonpath={.metadata.labels}")
        if result.returncode != 0:
            raise SafetyViolation(f"cannot read labels of {kind}/{name}: {result.stderr.strip()}")
        import json

        labels = json.loads(result.stdout or "{}")
        expected = {
            "infer.kunlun/ephemeral": "true",
            "infer.kunlun/task-id": task_id,
            "infer.kunlun/attempt-id": attempt_id,
        }
        mismatched = {key: labels.get(key) for key, value in expected.items() if labels.get(key) != value}
        if mismatched:
            raise SafetyViolation(
                f"refusing to auto-delete {kind}/{name}: it is not this attempt's ephemeral "
                f"resource (labels {mismatched} do not match {expected})"
            )
        return self.run(["delete", kind, name, "--wait=false"], timeout=300)
