"""KDP-001 deployment-proof executor.

Runs the Task's actions against a real Kunlun P800 namespace through the
adapter, records evidence for every step, and leaves the accept/reject decision
to validators.deployment_validator.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from adapters.kunlun_p800 import KunlunP800Adapter

_PLACEHOLDER = re.compile(r"\$\{([A-Z0-9_]+)\}")

FALLBACK_MARKERS = ("falling back to", "fallback to cpu", "using cuda backend")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ActionFailed(RuntimeError):
    def __init__(self, state: str, reason: str) -> None:
        super().__init__(f"{state}: {reason}")
        self.state = state
        self.reason = reason


def render_manifest(template: Path, values: dict[str, str]) -> str:
    text = template.read_text(encoding="utf-8")
    missing = sorted({m for m in _PLACEHOLDER.findall(text) if m not in values})
    if missing:
        raise ActionFailed("CONTRACT_INVALID", f"manifest placeholders unresolved: {missing}")
    return _PLACEHOLDER.sub(lambda m: values[m.group(1)], text)


def manifest_values(contract: dict[str, Any], attempt_id: str, workdir: str, image: str) -> dict[str, str]:
    context = contract["context"]
    execution = contract["execution"]
    target = context["target"]
    server = context["server"]
    return {
        "RESOURCE_NAME": execution["resource_name"],
        "NAMESPACE": execution["namespace"],
        "TASK_ID": contract["metadata"]["name"],
        "ATTEMPT_ID": attempt_id,
        "MODEL_NAME": context["model"]["name"],
        "MODEL_PVC": context["model"]["pvc"],
        "IMAGE": context["software"].get("image", image),
        "SERVER_PORT": str(server["port"]),
        "XPU_COUNT": str(target["device_count"]),
        "VOLCANO_QUEUE": target["volcano_queue"],
        "DEDICATED_POOL": target["dedicated_pool"],
        "WORKDIR": workdir,
    }


class DeploymentProofRunner:
    def __init__(
        self,
        contract: dict[str, Any],
        adapter: KunlunP800Adapter,
        repo_root: Path,
        artifact_dir: Path,
        workdir: str = "/workspace",
        image: str = "",
        attach_pod: str | None = None,
    ) -> None:
        self.contract = contract
        self.adapter = adapter
        self.repo_root = repo_root
        self.artifact_dir = artifact_dir
        self.workdir = workdir
        self.image = image
        self.attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.attach_pod = attach_pod
        self.pod: str | None = None
        self.records: list[dict[str, Any]] = []
        self.checks: dict[str, Any] = {}
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    # ---- evidence ---------------------------------------------------------
    def record(self, action: str, ok: bool, detail: str = "") -> None:
        self.records.append({"action": action, "ok": ok, "at": now(), "detail": detail[-4000:]})

    def write(self, name: str, content: str) -> Path:
        path = self.artifact_dir / name
        path.write_text(content, encoding="utf-8")
        return path

    # ---- actions ---------------------------------------------------------
    def preflight(self) -> None:
        if not self.adapter.can_create_pods():
            raise ActionFailed("NEEDS_HUMAN", "no permission to create pods in the namespace")
        self.record("preflight", True, "kubectl create-pods permission confirmed")

    def prepare_environment(self) -> None:
        execution = self.contract["execution"]
        template = self.repo_root / execution["manifest"]
        rendered = render_manifest(
            template, manifest_values(self.contract, self.attempt_id, self.workdir, self.image)
        )
        manifest_path = self.write("deployment_manifest.yaml", rendered)
        applied = self.adapter.apply(manifest_path)
        if applied.returncode != 0:
            raise ActionFailed("SERVER_START_FAILED", f"apply failed: {applied.stderr.strip()}")
        self.record("apply_manifest", True, applied.stdout.strip())
        self.pod = self.wait_for_pod(execution)
        self.install_runtime(execution)

    def wait_for_pod(self, execution: dict[str, Any]) -> str:
        deadline = time.time() + int(execution.get("startup_timeout_seconds", 900))
        selector = f"infer.kunlun/attempt-id={self.attempt_id}"
        while time.time() < deadline:
            listed = self.adapter.run(["get", "pods", "-l", selector, "-o", "name"])
            names = [line.split("/")[-1] for line in listed.stdout.split() if line.strip()]
            if names and self.adapter.pod_ready(names[0]):
                self.record("pod_ready", True, names[0])
                self.checks["pod_ready"] = True
                spec = self.adapter.get("pod", names[0], output="yaml")
                self.write("pod_spec.yaml", spec.stdout)
                return names[0]
            time.sleep(10)
        self.checks["pod_ready"] = False
        raise ActionFailed("READINESS_TIMEOUT", "no ready pod for this attempt before the timeout")

    def install_runtime(self, execution: dict[str, Any]) -> None:
        pod = self.pod or ""
        script = self.repo_root / "tools" / "install_vllm_kunlun.sh"
        copied = self.adapter.copy_into(pod, script, f"{self.workdir}/install_vllm_kunlun.sh")
        if copied.returncode != 0:
            raise ActionFailed("INSTALL_FAILED", f"cannot copy installer: {copied.stderr.strip()}")
        for command in execution.get("commands", {}).get("install", []):
            result = self.adapter.exec(pod, command, timeout=5400)
            self.write("install_log.txt", result.stdout + result.stderr)
            if result.returncode != 0:
                self.record("install", False, result.stderr[-2000:])
                raise ActionFailed("INSTALL_FAILED", f"{command!r} exited {result.returncode}")
            self.record("install", True, command)
        versions = self.adapter.exec(
            pod,
            "export VIRTUAL_ENV=/opt/vllm_kunlun PATH=/opt/vllm_kunlun/bin:$PATH; "
            "uv pip list | grep -iE '^(vllm|vllm-kunlun|torch|kunlun-ops|xspeedgate-ops) '; "
            f"cd {self.workdir}/vLLM-Kunlun && git rev-parse HEAD",
        )
        self.write("environment_versions.txt", versions.stdout + versions.stderr)
        self.record("environment_versions", versions.returncode == 0)

    def start_server(self) -> None:
        pod = self.pod or ""
        execution = self.contract["execution"]
        commands = execution.get("commands", {})
        health = self.contract["checks"]["health"]
        port = self.contract["context"]["server"]["port"]
        if self.attach_pod:
            # Launching a second server on an occupied port would leave the
            # bundle inconsistent: the checks would answer from the running
            # process while server_log came from the failing duplicate.
            code, _ = self.adapter.http_probe(pod, health["path"], port)
            if code == int(health["expected_status"]):
                self.record(
                    "start_server", True, "imported context: already serving, not relaunched"
                )
                return
        setup = " && ".join(commands.get("setup", []))
        serve = " ".join(commands.get("serve", []))
        if not serve:
            raise ActionFailed("CONTRACT_INVALID", "execution.commands.serve is empty")
        launch = (
            f"cd {self.workdir} && "
            "export VIRTUAL_ENV=/opt/vllm_kunlun PATH=/opt/vllm_kunlun/bin:$PATH && "
            f"{setup + ' && ' if setup else ''}"
            f"nohup {serve} > {self.server_log_path()} 2>&1 & echo started $!"
        )
        result = self.adapter.exec(pod, launch, timeout=120)
        if result.returncode != 0:
            raise ActionFailed("SERVER_START_FAILED", result.stderr.strip())
        self.record("start_server", True, result.stdout.strip())

    def poll_health(self) -> None:
        pod = self.pod or ""
        execution = self.contract["execution"]
        health = self.contract["checks"]["health"]
        port = self.contract["context"]["server"]["port"]
        need = int(execution.get("health_successes_required", 3))
        interval = int(execution.get("health_interval_seconds", 10))
        deadline = time.time() + int(execution.get("startup_timeout_seconds", 900))
        streak = 0
        last = ""
        while time.time() < deadline:
            code, body = self.adapter.http_probe(pod, health["path"], port)
            last = f"status={code} body={body[:200]}"
            streak = streak + 1 if code == int(health["expected_status"]) else 0
            if streak >= need:
                self.write("health_result.txt", last)
                self.checks["health_check"] = int(health["expected_status"])
                self.record("poll_health", True, f"{need} consecutive successes")
                return
            time.sleep(interval)
        self.write("health_result.txt", last)
        self.record("poll_health", False, last)
        raise ActionFailed("READINESS_TIMEOUT", f"health never stabilised: {last}")

    def run_chat_smoke(self) -> None:
        pod = self.pod or ""
        chat = self.contract["checks"]["chat"]
        port = self.contract["context"]["server"]["port"]
        payload = json.dumps(chat.get("payload", {}))
        remote_payload = f"{self.workdir}/chat_payload.json"
        self.adapter.exec(pod, f"cat > {remote_payload} <<'JSON'\n{payload}\nJSON")
        result = self.adapter.exec(
            pod,
            f"curl -sS -X {chat.get('method', 'POST')} "
            f"-H 'Content-Type: application/json' -d @{remote_payload} "
            f"http://127.0.0.1:{port}{chat['path']}",
            timeout=300,
        )
        body = result.stdout
        self.write("chat_result.json", body + result.stderr)
        text, finish = "", ""
        try:
            choices = json.loads(body).get("choices", [])
            if choices:
                message = choices[0].get("message", {}) or {}
                finish = choices[0].get("finish_reason", "")
                # Reasoning models (e.g. a minimax_m2 reasoning parser) put the
                # generated tokens in `reasoning` and leave `content` null until
                # they finish thinking, so both count as generated text.
                text = message.get("content") or message.get("reasoning") or ""
                text = text or message.get("reasoning_content") or ""
        except (json.JSONDecodeError, AttributeError, IndexError):
            text = ""
        if chat.get("expected_non_empty_text", True) and not text.strip():
            self.checks["chat_completion"] = "empty"
            self.record("run_chat_smoke", False, body[:1000])
            raise ActionFailed("API_SMOKE_FAILED", "chat completion returned no text")
        if finish == "length":
            self.record("run_chat_smoke", True, "truncated by max_tokens; raise it for a full answer")
        self.checks["chat_completion"] = "non_empty"
        self.record("run_chat_smoke", True, f"finish_reason={finish} text={text[:160]!r}")

    def server_log_path(self) -> str:
        return self.contract["execution"].get("server_log") or f"{self.workdir}/server.log"

    def verify_backend(self) -> None:
        pod = self.pod or ""
        backend = self.contract["checks"].get("backend", {})
        expected = backend.get("expected", "kunlun")
        log = self.adapter.exec(pod, f"cat {self.server_log_path()}", timeout=120).stdout
        self.write("server_log.txt", log)
        lowered = log.lower()
        self.checks["expected_backend"] = expected if expected in lowered else "unknown"
        fallbacks = [marker for marker in FALLBACK_MARKERS if marker in lowered]
        self.checks["unexpected_fallback"] = bool(fallbacks)
        if backend.get("reject_unexpected_fallback", True) and fallbacks:
            self.record("verify_backend", False, f"fallback markers: {fallbacks}")
            raise ActionFailed("UNEXPECTED_FALLBACK", f"server log shows {fallbacks}")
        if self.checks["expected_backend"] != expected:
            self.record("verify_backend", False, f"{expected!r} not found in server log")
            raise ActionFailed("UNEXPECTED_FALLBACK", f"{expected!r} backend not confirmed in log")
        self.record("verify_backend", True, expected)

    def collect_artifacts(self, state: str, reason: str = "") -> dict[str, Any]:
        import yaml

        self.write("task_contract.yaml", yaml.safe_dump(self.contract, sort_keys=False, allow_unicode=True))
        self.write(
            "reproduce_command.txt",
            "KUBECONFIG={kubeconfig} python3 runners/task_runner.py {contract} --execute\n".format(
                kubeconfig=self.adapter.config.kubeconfig,
                contract=self.contract["metadata"]["name"],
            ),
        )
        status = {
            "task_id": self.contract["metadata"]["name"],
            "state": state,
            "updated_at": now(),
            "checks": self.checks,
            "artifacts": sorted(str(p.name) for p in self.artifact_dir.iterdir()),
        }
        if reason:
            status["reason"] = reason
        self.write("status.json", json.dumps(status, indent=2, ensure_ascii=False) + "\n")
        self.write("execution_records.json", json.dumps(self.records, indent=2, ensure_ascii=False) + "\n")
        # status.json is listed too, so refresh the artifact list once written.
        status["artifacts"] = sorted(str(p.name) for p in self.artifact_dir.iterdir())
        return status

    # ---- main loop --------------------------------------------------------
    def run(self) -> dict[str, Any]:
        try:
            if self.attach_pod:
                # Imported Context: the Pod and its runtime were prepared by an
                # earlier attempt, so those two actions are not re-proven here.
                self.adapter.assert_owned(self.attach_pod)
                if not self.adapter.pod_ready(self.attach_pod):
                    raise ActionFailed("NEEDS_HUMAN", f"{self.attach_pod} is not ready")
                self.pod = self.attach_pod
                self.checks["pod_ready"] = True
                self.write("pod_spec.yaml", self.adapter.get("pod", self.pod, output="yaml").stdout)
                self.record("attach_pod", True, f"imported context: {self.attach_pod}")
            else:
                self.preflight()
                self.prepare_environment()
            self.start_server()
            self.poll_health()
            self.run_chat_smoke()
            self.verify_backend()
        except ActionFailed as failure:
            retain = self.contract["execution"].get("retain_on_failure", True)
            self.record("outcome", False, f"{failure.state}; retain_on_failure={retain}")
            return self.collect_artifacts(failure.state, failure.reason)
        except subprocess.TimeoutExpired as timeout:
            self.record("outcome", False, f"command timed out: {timeout.cmd}")
            return self.collect_artifacts("NEEDS_HUMAN", f"command timed out after {timeout.timeout}s")
        return self.collect_artifacts("DEPLOYMENT_READY")




