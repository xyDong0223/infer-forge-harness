"""KDP-001 deployment-proof executor.

Runs the Task's actions against a real Kunlun P800 namespace through the
adapter, records evidence for every step, and leaves the accept/reject decision
to validators.deployment_validator.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from adapters.kunlun_p800 import KunlunP800Adapter, SafetyViolation

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
        "USER_ID": execution["resource_name"].split("-", 1)[0],
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
        phase: str = "all",
    ) -> None:
        self.contract = contract
        self.adapter = adapter
        self.repo_root = repo_root
        self.artifact_dir = artifact_dir
        self.workdir = workdir
        self.image = image
        self.attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.attach_pod = attach_pod
        # "environment" proves the stack without launch parameters, "service"
        # proves the server given a prepared pod, "all" keeps the original
        # single-shot behaviour.
        if phase not in ("all", "environment", "service"):
            raise ActionFailed("CONTRACT_INVALID", f"unknown phase {phase!r}")
        self.phase = phase
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

    def discover_base_model(self) -> None:
        model = self.contract.get("context", {}).get("model", {})
        path = model.get("path")
        if not path:
            raise ActionFailed("CONTRACT_INVALID", "base model path is missing")
        result = self.adapter.exec(self.pod or "", f"test -f {shlex.quote(path + '/config.json')} && find {shlex.quote(path)} -maxdepth 1 -type f -name '*.safetensors' | sort", timeout=120)
        identity = {"name": model.get("name"), "path": path, "files": result.stdout.splitlines()}
        self.write("base_model_identity.json", json.dumps(identity, indent=2) + "\n")
        self.checks["base_model_loaded"] = result.returncode == 0 and bool(identity["files"])
        if not self.checks["base_model_loaded"]:
            raise ActionFailed("MODEL_NOT_FOUND", f"base model is not readable: {path}")
        self.record("discover_base_model", True, f"{path}: {len(identity['files'])} weight files")

    def cleanup_service_processes(self, skip_if_healthy: bool = False) -> None:
        port = self.contract.get("context", {}).get("server", {}).get("port", 8356)
        if skip_if_healthy and self.attach_pod:
            health = self.contract.get("checks", {}).get("health", {})
            code, _ = self.adapter.http_probe(self.attach_pod, health.get("path", "/health"), port)
            if code == int(health.get("expected_status", 200)):
                self.record(
                    "cleanup_service_processes", True,
                    f"skipped: {self.attach_pod} is already serving a healthy base model",
                )
                return
        # vLLM renames its processes after launch (VLLM::APIServer,
        # VLLM::EngineCore, VLLM::Worker_TP*), so matching only the original
        # command line leaves the TP workers alive holding HBM: the
        # 2026-09-14 attach rerun orphaned eight workers at ~89 GiB/card and
        # the relaunch died on "Free memory on device cuda:0 (6.97/96.0
        # GiB)". The patterns are bracket-quoted because a plain pattern
        # matches the bash -lc process running this very command line: the
        # first kill -9 then terminates the loop's own shell (exit 137) and
        # nothing is killed at all — the cleanup used to report success
        # while only fuser reached the port-owning APIServer. The
        # multiprocessing resource_tracker is included because it outlives
        # the engine and re-parents to init. The survivor probe then proves
        # the tree is actually gone instead of assuming it.
        command = (
            "pkill -9 -f '[v]llm.entrypoints.openai.api_server'; "
            "pkill -9 -f '[v]llm.engine'; "
            "pkill -9 -f '[E]ngineCore'; "
            "pkill -9 -f '[V]LLM::'; "
            "pkill -9 -f '[m]ultiprocessing.resource_tracker'; "
            f"(command -v fuser >/dev/null 2>&1 && fuser -k {int(port)}/tcp) || true"
        )
        self.adapter.exec(self.pod or "", command, timeout=60)
        survivor = ""
        for _ in range(10):
            probe = self.adapter.exec(
                self.pod or "",
                "ps -eo pid=,args= | grep -E '[V]LLM::|[v]llm.entrypoints|[m]ultiprocessing.resource_tracker' "
                "| grep -v grep | head -5",
                timeout=30,
            )
            survivor = probe.stdout.strip()
            if not survivor:
                break
            time.sleep(3)
        if survivor:
            raise ActionFailed(
                "NEEDS_HUMAN",
                f"service processes survived cleanup and still hold HBM:\n{survivor}",
            )
        self.record("cleanup_service_processes", True, f"service process tree and port {port} cleared")

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

    def _runtime_already_installed(self) -> bool:
        """Cheap idempotence gate for the attach path.

        Reinstalling the venv while a server from an earlier attempt is still
        serving replaces package files that a running process may lazily
        import: the 2026-09-14 attach rerun killed a healthy base server that
        way. If the runtime imports and the pinned worktree is present, the
        install already happened on this pod and must not run again.
        """
        pod = self.pod or ""
        probe = self.adapter.exec(
            pod,
            "export VIRTUAL_ENV=/opt/vllm_kunlun PATH=/opt/vllm_kunlun/bin:$PATH; "
            'python3 -c "import torch, vllm, vllm_kunlun" && '
            f"test -f {shlex.quote(self.workdir)}/vLLM-Kunlun/setup_env.sh",
            timeout=300,
        )
        return probe.returncode == 0

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

    def _serves_target_model(self, body: str) -> bool:
        """Is the running server serving *this* contract's model?

        Health proves a server exists, not that it is the right server: the
        environment phase's base-smoke server must not silently answer the
        target model's service proof.
        """
        try:
            models = json.loads(body).get("data") or []
        except json.JSONDecodeError:
            return False
        served = self.contract.get("context", {}).get("server", {}).get(
            "served_model_name", "")
        return any(model.get("id") == served for model in models)

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
            code, body = self.adapter.http_probe(pod, "/v1/models", port)
            if code == 200 and self._serves_target_model(body):
                self.record(
                    "start_server", True, "imported context: already serving, not relaunched"
                )
                return
            if code == 200:
                self.cleanup_service_processes()
                self.record(
                    "start_server", True,
                    "imported context served a different model; cleared before launch",
                )
        setup = " && ".join(commands.get("setup", []))
        serve = " ".join(commands.get("serve", []))
        if not serve:
            raise ActionFailed("CONTRACT_INVALID", "execution.commands.serve is empty")
        # The whole chain must be grouped and detached, not only the serve
        # command: `a && b && serve > log &` backgrounds one subshell whose own
        # stdout/stderr are still the kubectl exec session's pipes, so the
        # session never sees EOF and a healthy launch is misread as a 120s
        # timeout. stdin detach alone did not fix this (run
        # glm52-int-w8a8-p800-001, six NEEDS_HUMAN false negatives); grouping
        # with a group-wide redirect does (reproduced on the prepared pod
        # 2026-09-14: ungrouped hangs kubectl exec until the client timeout,
        # grouped returns in <1s with the server still running).
        # Archive any previous attempt's log before the launch truncates it.
        # Without this, a later attempt (the mat-006 triage rerun, a manual
        # relaunch) destroys the crash evidence of the attempt that just
        # failed — run glm52-int-w8a8-p800-001 lost the entire engine-side
        # stack trace of a first-request EngineCore crash this way, 40
        # seconds after the crash.
        launch = (
            f"cp -f {self.server_log_path()} {self.server_log_path()}.prev 2>/dev/null; "
            f"( cd {self.workdir} && "
            "export VIRTUAL_ENV=/opt/vllm_kunlun PATH=/opt/vllm_kunlun/bin:$PATH && "
            f"{setup + ' && ' if setup else ''}"
            f"{serve} ) < /dev/null > {self.server_log_path()} 2>&1 & echo started $!"
        )
        result = self.adapter.exec(pod, launch, timeout=120)
        if result.returncode != 0:
            raise ActionFailed("SERVER_START_FAILED", result.stderr.strip())
        self.record("start_server", True, result.stdout.strip())

    def poll_health(self, prefix: str = "") -> None:
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
                self.write(f"{prefix}health_result.txt" if prefix else "health_result.txt", last)
                self.checks[f"{prefix}health_check" if prefix else "health_check"] = int(health["expected_status"])
                self.record("poll_health", True, f"{need} consecutive successes")
                return
            # A server that died mid-startup must fail now, not after the full
            # timeout: the log tail is the diagnosis, and an hour of polling a
            # corpse (707 GiB models time out at ~1 h) hides it. The bracket
            # trick keeps pgrep from matching its own command line - an
            # unbracketed pattern always matched the probe itself and the
            # check reported every dead server as alive.
            alive = self.adapter.exec(
                pod,
                "pgrep -f '[v]llm.entrypoints.openai.api_server' >/dev/null && echo up || echo dead",
                timeout=30,
            )
            if alive.stdout.strip() == "dead" and streak == 0:
                log = self.adapter.exec(pod, f"tail -40 {self.server_log_path()}", timeout=60).stdout
                self.write(f"{prefix}health_result.txt" if prefix else "health_result.txt", last)
                self.record("poll_health", False, "server process died during startup")
                raise ActionFailed(
                    "SERVER_START_FAILED",
                    f"the server process died during startup; log tail:\n{log[-1500:]}",
                )
            time.sleep(interval)
        self.write(f"{prefix}health_result.txt" if prefix else "health_result.txt", last)
        self.record("poll_health", False, last)
        raise ActionFailed("READINESS_TIMEOUT", f"health never stabilised: {last}")

    def run_chat_smoke(self, prefix: str = "") -> None:
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
        self.write(f"{prefix}chat_result.json" if prefix else "chat_result.json", body + result.stderr)
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
        completion_key = f"{prefix}chat_completion" if prefix else "chat_completion"
        if chat.get("expected_non_empty_text", True) and not text.strip():
            self.checks[completion_key] = "empty"
            self.record("run_chat_smoke", False, body[:1000])
            # A 5xx body ("EngineCore encountered an issue") means the engine
            # process crashed on this request: the health endpoint answered
            # 200 a moment earlier, so the API server is up while the core is
            # dead. The engine-side stack trace is the only root cause, and it
            # lives in the server log — persist it before anything relaunches
            # and truncates the file (run glm52-int-w8a8-p800-001 lost it).
            log = ""
            try:
                log = self.adapter.exec(
                    pod, f"tail -n 200 {self.server_log_path()}", timeout=120
                ).stdout
            except Exception:  # noqa: BLE001 - log capture must never mask the failure
                pass
            if log:
                key = f"{prefix}chat_failure_server_log.txt" if prefix else "chat_failure_server_log.txt"
                self.write(key, log)
            raise ActionFailed(
                "API_SMOKE_FAILED",
                f"chat completion returned no text; response: {body[:500]}; "
                f"server log tail:\n{log[-1500:]}",
            )
        if finish == "length":
            self.record("run_chat_smoke", True, "truncated by max_tokens; raise it for a full answer")
        self.checks[completion_key] = "non_empty"
        if prefix:
            # Service reachability is separate from correctness. These fields
            # only prove that prompt ingestion and token generation occurred.
            self.checks[f"{prefix}prefill"] = True
            self.checks[f"{prefix}decode"] = bool(text.strip())
        self.record("run_chat_smoke", True, f"finish_reason={finish} text={text[:160]!r}")

    def server_log_path(self) -> str:
        return self.contract["execution"].get("server_log") or f"{self.workdir}/server.log"

    def verify_backend(self, prefix: str = "") -> None:
        pod = self.pod or ""
        backend = self.contract["checks"].get("backend", {})
        expected = backend.get("expected", "kunlun")
        log = self.adapter.exec(pod, f"cat {self.server_log_path()}", timeout=120).stdout
        self.write(f"{prefix}server_log.txt" if prefix else "server_log.txt", log)
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
            # The pod is part of the deliverable: the service phase imports this
            # exact pod, and a scan Task runs inside it.
            "pod": self.pod,
            "phase": self.phase,
            "checks": self.checks,
            "artifacts": sorted(str(p.name) for p in self.artifact_dir.iterdir()),
        }
        if reason:
            status["reason"] = reason
            self.write("diagnosis.json", json.dumps({"state": state, "error": reason, "pod": self.pod, "next_action": "DIAGNOSE_RUNTIME"}, indent=2) + "\n")
        self.write("status.json", json.dumps(status, indent=2, ensure_ascii=False) + "\n")
        self.write("execution_records.json", json.dumps(self.records, indent=2, ensure_ascii=False) + "\n")
        # status.json is listed too, so refresh the artifact list once written.
        status["artifacts"] = sorted(str(p.name) for p in self.artifact_dir.iterdir())
        return status

    # ---- main loop --------------------------------------------------------
    def run(self) -> dict[str, Any]:
        try:
            if self.phase == "service" and not self.attach_pod:
                # Checked before anything is created: preparing a fresh pod here
                # would reinstall the runtime and invalidate the environment proof
                # this phase is supposed to import.
                raise ActionFailed(
                    "CONTRACT_INVALID",
                    "the service phase needs a prepared pod: pass --attach-pod with the pod "
                    "recorded by the environment phase",
                )
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
                if self.phase == "environment":
                    # An attached Pod may only have the base image, so
                    # environment proof owns runtime installation. But
                    # reinstalling while a server from an earlier attempt is
                    # still serving replaces package files a lazy import may
                    # need: the 2026-09-14 attach rerun killed a healthy base
                    # server that way. The install runs only when this pod
                    # does not already have an importable runtime.
                    if self._runtime_already_installed():
                        self.record(
                            "install", True,
                            "skipped: attached pod already has an importable runtime and pinned worktree",
                        )
                    else:
                        self.install_runtime(self.contract["execution"])
            else:
                self.preflight()
                self.prepare_environment()
            if self.phase == "environment":
                self.verify_runtime_importable()
                self.discover_base_model()
                # skip_if_healthy: an attached pod that is already serving the
                # base model is the imported context this phase is supposed to
                # prove, not a stale server to clear. A re-run must not pay a
                # second 215 GiB load for the same evidence.
                self.cleanup_service_processes(skip_if_healthy=True)
                self.start_server()
                self.poll_health(prefix="base_")
                self.run_chat_smoke(prefix="base_")
                self.verify_backend(prefix="base_")
                return self.collect_artifacts("ENVIRONMENT_READY")
            self.start_server()
            self.poll_health()
            self.run_chat_smoke()
            self.verify_backend()
        except SafetyViolation as violation:
            # The shared-namespace ownership guard is an input error (wrong pod
            # name, unset USER_ID), not a cluster failure: report it as a
            # machine-readable status instead of a traceback so the graph can
            # route it instead of crashing the executor.
            self.record("outcome", False, f"safety violation: {violation}")
            return self.collect_artifacts("CONTRACT_INVALID", str(violation))
        except ActionFailed as failure:
            retain = self.contract["execution"].get("retain_on_failure", True)
            self.record("outcome", False, f"{failure.state}; retain_on_failure={retain}")
            return self.collect_artifacts(failure.state, failure.reason)
        except subprocess.TimeoutExpired as timeout:
            self.record("outcome", False, f"command timed out: {timeout.cmd}")
            return self.collect_artifacts("NEEDS_HUMAN", f"command timed out after {timeout.timeout}s")
        return self.collect_artifacts("DEPLOYMENT_READY")

    def verify_runtime_importable(self) -> None:
        """Prove the stack is usable, not merely installed.

        `uv pip list` showing vllm-kunlun proves nothing: the 2026-09-03 failure
        had the package present while `import vllm_kunlun` died on a C++ extension
        built by the wrong compiler. Importing is the cheapest check that
        separates a broken environment from a model problem, and proving it here
        means a later kernel failure cannot be blamed on the install.
        """
        pod = self.pod or ""
        script = (
            "export VIRTUAL_ENV=/opt/vllm_kunlun PATH=/opt/vllm_kunlun/bin:$PATH; "
            'python3 -c "import json, torch, vllm, vllm_kunlun; '
            "print(json.dumps({'torch': torch.__version__, 'vllm': vllm.__version__, "
            "'vllm_kunlun': getattr(vllm_kunlun, '__version__', 'unknown')}))\""
        )
        result = self.adapter.exec(pod, script, timeout=600)
        self.write("runtime_import.txt", result.stdout + result.stderr)
        if result.returncode != 0:
            self.checks["runtime_importable"] = False
            raise ActionFailed("INSTALL_FAILED", f"import failed: {result.stderr.strip()[-1500:]}")
        self.checks["runtime_importable"] = True
        self.record("verify_runtime_importable", True, result.stdout.strip()[-300:])
        self.verify_code_and_device_ready()
        self.collect_environment_fingerprint()

    def verify_code_and_device_ready(self) -> None:
        """Prove later investigation runs in the prepared code/XPU context."""
        pod = self.pod or ""
        worktree = f"{self.workdir}/vLLM-Kunlun"
        code_result = self.adapter.exec(
            pod,
            " && ".join(
                [
                    f"test -d {shlex.quote(worktree)}/.git",
                    f"test -f {shlex.quote(worktree)}/setup_env.sh",
                    f"cd {shlex.quote(worktree)} && git rev-parse HEAD",
                ]
            ),
            timeout=120,
        )
        code_payload = {
            "worktree": worktree,
            "git_head": code_result.stdout.strip().splitlines()[-1]
            if code_result.stdout.strip()
            else None,
            "setup_env": code_result.returncode == 0,
        }
        self.write("code_readiness.json", json.dumps(code_payload, indent=2) + "\n")
        self.checks["code_ready"] = code_result.returncode == 0 and bool(code_payload["git_head"])
        if not self.checks["code_ready"]:
            raise ActionFailed("CODE_NOT_READY", "vLLM-Kunlun worktree or setup_env.sh is unavailable")

        try:
            cards = self.adapter.xpu_smi(pod)
            expected = int(self.contract.get("context", {}).get("target", {}).get("device_count", 1))
            device_payload = {"cards": cards, "expected_count": expected, "count": len(cards)}
            device_ok = len(cards) >= expected
        except (RuntimeError, ValueError) as error:
            device_payload = {"cards": [], "error": str(error)}
            device_ok = False
        self.write("device_readiness.json", json.dumps(device_payload, indent=2) + "\n")
        self.checks["device_ready"] = device_ok
        if not device_ok:
            raise ActionFailed("DEVICE_NOT_READY", "the prepared pod has no expected XPU devices")

    def collect_environment_fingerprint(self) -> None:
        """Record what this conclusion is only true of.

        Every adaptation claim holds for one combination of stack, vendor kernels
        and driver. Without the fingerprint, "it worked yesterday" cannot be told
        apart from "something under us moved".
        """
        pod = self.pod or ""
        script = (
            "export VIRTUAL_ENV=/opt/vllm_kunlun PATH=/opt/vllm_kunlun/bin:$PATH; "
            "echo '## packages'; uv pip list | grep -iE "
            "'^(vllm|vllm-kunlun|torch|torch-xmlir|kunlun-ops|xspeedgate-ops|triton) '; "
            f"echo '## vllm-kunlun commit'; (cd {self.workdir}/vLLM-Kunlun && git rev-parse HEAD); "
            "echo '## driver'; xpu_smi | sed -n '3p'; "
            "echo '## device'; xpu_smi -m | head -1; "
            "echo '## os'; . /etc/os-release && echo \"$PRETTY_NAME\"; gcc --version | head -1"
        )
        result = self.adapter.exec(pod, script, timeout=300)
        self.write("environment_fingerprint.txt", result.stdout + result.stderr)
        self.record("environment_fingerprint", result.returncode == 0)
