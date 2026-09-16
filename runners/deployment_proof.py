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

from adapters import SafetyViolation, get_hardware, push_snippet
from runtimes import default_runtime
from runners import evidence
from core.storage import ArtifactStore, WritePolicyError, ensure_external, locate_attempt

KunlunP800Adapter = get_hardware()

_PLACEHOLDER = re.compile(r"\$\{([A-Z0-9_]+)\}")

# Legacy default for callers that inject no runtime. CUDA names stay out:
# Kunlun exposes the XPU through the torch.cuda API, so a cuda-sounding log
# line is the native path, not a fallback (see VllmKunlunRuntime.fallback_markers).
FALLBACK_MARKERS = ("falling back to", "fallback to cpu")
PATCH_NOT_APPLICABLE = 2


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
        runtime: object | None = None,
    ) -> None:
        self.contract = contract
        self.adapter = adapter
        self.repo_root = repo_root
        self.artifact_dir = ensure_external(artifact_dir, repo_root)
        self.store = ArtifactStore(self.artifact_dir)
        attempt = locate_attempt(self.artifact_dir)
        if attempt is not None:
            if self.artifact_dir != attempt.output and attempt.output not in self.artifact_dir.parents:
                raise WritePolicyError("deployment output must reside inside its attempt output/")
            if (attempt.root / "manifest.json").exists():
                raise WritePolicyError("deployment attempt has a formal result; use a fresh attempt")
        elif self.artifact_dir.exists() and any(self.artifact_dir.iterdir()):
            raise WritePolicyError("unmanaged deployment output must be fresh and empty")
        if self.store.path("status.json").exists() or self.store.path("manifest.json").exists():
            raise WritePolicyError("deployment output already has a formal result; use a fresh directory")
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
        self.runtime = runtime or default_runtime()
        self.pod: str | None = None
        self.records: list[dict[str, Any]] = []
        self.checks: dict[str, Any] = {}
        # Pod-side startup watch state: last observed server-log size and
        # how many consecutive polls it has been frozen (see
        # _watch_server_log).
        self._watch_last_bytes = -2
        self._watch_stall_polls = 0
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    # ---- evidence ---------------------------------------------------------
    def record(self, action: str, ok: bool, detail: str = "") -> None:
        self.records.append({"action": action, "ok": ok, "at": now(), "detail": detail[-4000:]})

    def write(self, name: str, content: str) -> Path:
        return self.store.write_text(name, content, overwrite=True)

    def write_unique(self, name: str, content: str) -> Path:
        """Write evidence that no later attempt can overwrite.

        ``write`` replaces — correct for state, fatal for crash evidence: a
        rerun of this task must not destroy the previous attempt's proof while
        "recording" its own.
        """
        path = self.store.path(name)
        suffix = 0
        while True:
            candidate = path if suffix == 0 else path.with_name(f"{path.name}.{suffix}")
            try:
                return self.store.write_text(candidate.relative_to(self.artifact_dir), content)
            except FileExistsError:
                suffix += 1

    def append(self, name: str, line: str) -> Path:
        """Append one line to an append-only journal artifact.

        Watches journal beats across a whole startup; replacing would leave
        only the last beat, which is precisely the evidence gap the watch
        exists to close.
        """
        path = self.store.path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        return path

    def _watch_server_log(self, poll_index: int, code: int) -> None:
        """Journal startup progress from the pod-side server log.

        Health polling answers 503 for ~15 minutes of a 707 GiB load with
        nothing durable in between — run glm52-int-w8a8-p800-001 had four
        "any progress?" interruptions and no on-disk answer. Each poll now
        journals the server log's size and last line, and flags a stall when
        the log stops growing well before the deadline: a vLLM loader that
        is loading prints; a frozen log is the early signal that the wait
        will not end well.
        """
        pod = self.pod or ""
        log = self.server_log_path()
        try:
            probe = self.adapter.exec(
                pod,
                f"stat -c %s {shlex.quote(log)} 2>/dev/null; "
                f"tail -n 1 {shlex.quote(log)} 2>/dev/null",
                timeout=30,
            )
        except Exception:  # noqa: BLE001 - watching must never mask polling
            return
        lines = (probe.stdout or "").strip().splitlines()
        size = int(lines[0]) if lines and lines[0].isdigit() else -1
        last = lines[1][:200].strip() if len(lines) > 1 else ""
        if size == self._watch_last_bytes:
            self._watch_stall_polls += 1
        else:
            self._watch_stall_polls = 0
            self._watch_last_bytes = size
        entry = {
            "at": now(), "poll": poll_index, "health": code,
            "log_bytes": size, "last_line": last,
        }
        threshold = int(self.contract["execution"].get("watch_stall_polls", 30))
        if self._watch_stall_polls >= threshold:
            entry["stall"] = True
            entry["stall_polls"] = self._watch_stall_polls
            if not self.checks.get("startup_log_stall"):
                self.checks["startup_log_stall"] = True
                self.record(
                    "watch_server_log", False,
                    f"server log frozen at {size} bytes for "
                    f"{self._watch_stall_polls} polls — the load is not "
                    "progressing",
                )
        self.append("startup_watch.jsonl", json.dumps(entry, ensure_ascii=False))

    def archive_server_crash(self, tag: str) -> str:
        """Crash-first: freeze the server log the moment death is detected.

        Two copies, both non-overwritable: the pod-side ``server.log.crash-N``
        survives even a manual relaunch inside the pod, and the artifact copy
        carries the full log — not a tail, because the stack trace is not
        always in the last 40 lines of a 707 GiB load. Archiving failures are
        recorded but never mask the failure being archived.
        """
        pod = self.pod or ""
        log = ""
        try:
            self.adapter.exec(pod, evidence.archive_crash_remote(self.server_log_path()), timeout=60)
            log = self.adapter.exec(pod, f"cat {self.server_log_path()}", timeout=120).stdout
        except Exception:  # noqa: BLE001 - log capture must never mask the failure
            self.record("archive_server_crash", False, f"{tag}: log unreachable")
            return ""
        if log:
            path = self.write_unique(f"server_crash_log_{tag}.txt", log)
            self.record("archive_server_crash", True, f"{tag}: {path.name}")
        else:
            self.record("archive_server_crash", False, f"{tag}: server log was empty")
        return log

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
            f"{self.runtime.env_prefix()}; "
            f"{self.runtime.import_check_command()} && "
            f"{self.runtime.worktree_check_command(shlex.quote(self.workdir))}",
            timeout=300,
        )
        return probe.returncode == 0

    def install_runtime(self, execution: dict[str, Any]) -> None:
        pod = self.pod or ""
        script = self.runtime.installer_path(self.repo_root)
        installer_name = self.runtime.installer_name()
        copied = self.adapter.copy_into(pod, script, f"{self.workdir}/{installer_name}")
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
            f"{self.runtime.env_prefix()}; "
            f"{self.runtime.package_query_command()}; "
            f"{self.runtime.worktree_revision_command(self.workdir)}",
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
        # Archive any previous attempt's log before the launch truncates it,
        # to the next free .prev-N — never a fixed name. A single `.prev`
        # protected exactly one generation: the third attempt (a manual
        # relaunch, the MAT-006 triage reproof) destroyed the second's crash
        # evidence while "archiving" it, 40 s after the crash it was sent to
        # explain (run glm52-int-w8a8-p800-001, 2026-09-14).
        launch = (
            evidence.archive_before_truncate(self.server_log_path())
            + f"( cd {self.workdir} && "
            f"{self.runtime.env_prefix()} && "
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
        poll_index = 0
        while time.time() < deadline:
            code, body = self.adapter.http_probe(pod, health["path"], port)
            last = f"status={code} body={body[:200]}"
            # Startup watch: journal the server log's progress every poll,
            # stall-flag when it stops growing (see _watch_server_log).
            self._watch_server_log(poll_index, code)
            poll_index += 1
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
                # Crash-first: the full log is snapshotted pod-side and as an
                # artifact before anything else runs; the tail below is only a
                # fallback when archiving itself failed.
                log = self.archive_server_crash("startup_death")
                if not log:
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
        stall = (
            f"; the server log has been frozen at {self._watch_last_bytes} bytes "
            f"for {self._watch_stall_polls} polls (startup_watch.jsonl) — the "
            "load is not progressing"
            if self.checks.get("startup_log_stall")
            else ""
        )
        raise ActionFailed("READINESS_TIMEOUT", f"health never stabilised: {last}{stall}")

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
            # lives in the server log — crash-first: the full log is frozen
            # pod-side (.crash-N) and as a non-overwritable artifact before
            # anything relaunches and truncates the file (run
            # glm52-int-w8a8-p800-001 lost it this way).
            log = self.archive_server_crash("chat_failure")
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

    def apply_runtime_patches(self) -> None:
        """Replay the repo's idempotent runtime patches (protocol hard rule).

        AGENTS.md: repairs written into runtime state must also exist as
        replayable patches under tools/patches/. This runs them after every
        install and every attach, before the drift precheck verifies the
        result — so a reinstalled pod self-heals instead of silently
        regressing to the unpatched state (run glm52-int-w8a8-p800-001:
        thirteen drift repairs lived only in one pod's site-packages and
        evaporated on the next pod).

        A patch script is an exact-anchor repair for one engine/plugin source
        shape. On a different pair its anchors miss and it exits 2: that is
        recorded as SKIPPED and the run continues. Other non-zero exits are
        real execution failures and must not be mislabeled as incompatibility.
        The drift precheck that follows is the verdict on what an unmatched
        environment actually needs.
        """
        patches_dir = self.repo_root / "tools" / "patches"
        scripts = sorted(patches_dir.glob("patch_*.py")) if patches_dir.exists() else []
        if not scripts:
            self.record("apply_runtime_patches", True,
                        "no patch scripts under tools/patches/")
            return
        output: list[str] = []
        skipped: list[str] = []
        failed: list[str] = []
        for script in scripts:
            remote = f"/tmp/{script.name}"
            command = (
                f"{self.runtime.env_prefix()}; "
                f"{push_snippet(script, remote)} && python3 {remote}"
            )
            result = self.adapter.exec(self.pod or "", command, timeout=600)
            output.append(f"$ {script.name} (exit {result.returncode})\n"
                          f"{result.stdout}{result.stderr}")
            if result.returncode == PATCH_NOT_APPLICABLE:
                skipped.append(script.name)
                output.append(
                    f">>> {script.name}: SKIPPED — the patch set is anchored to "
                    "one engine/plugin source shape and does not match this "
                    "install. Not fatal; the drift precheck below reports "
                    "what this environment actually needs.\n"
                )
            elif result.returncode != 0:
                failed.append(script.name)
                output.append(
                    f">>> {script.name}: FAILED — exit {result.returncode} is "
                    "not the explicit not-applicable result.\n"
                )
        self.write("runtime_patches.txt", "\n".join(output))
        if failed:
            detail = f"runtime patch execution failed: {', '.join(failed)}"
            self.record("apply_runtime_patches", False, detail)
            raise ActionFailed("RUNTIME_PATCH_FAILED", detail)
        detail = f"replayed {len(scripts)} patch script(s)"
        if skipped:
            detail += (f", {len(skipped)} skipped (non-matching pair): "
                       f"{', '.join(skipped)}")
        self.record("apply_runtime_patches", True, detail)

    def engine_core_drift_precheck(self) -> None:
        """Engine-core-init drift dry-run: seconds, not one reload per drift.

        MAT-027 proves the plugin imports; that gate passed while twelve
        call-time drifts still waited behind the 707 GiB weight load — each
        found only by a full server restart (run glm52-int-w8a8-p800-001,
        2026-09-14: ~15 minutes per drift, twelve times). The precheck
        replays the plugin's engine-facing references and calls against the
        installed engine — signature binding with placeholder arguments,
        the known init-path landmines, the vendor ops on the request path —
        with no weights and no server. It doubles as the "are the drift
        repairs applied on THIS pod" gate: a reinstalled pod that lost the
        site-packages repairs fails here in seconds instead of at the first
        EngineCore init after a full load.
        """
        probe = self.repo_root / "tools" / "probe" / "engine_core_drift_precheck.py"
        model_path = self.contract.get("context", {}).get("model", {}).get("path")
        model_config = f"{model_path}/config.json" if model_path else ""
        script = (
            f"{self.runtime.env_prefix()}; "
            + push_snippet(probe, "/tmp/kdp_drift_precheck.py")
            + " && python3 /tmp/kdp_drift_precheck.py"
            + (f" --model-config {shlex.quote(model_config)}" if model_config else "")
        )
        result = self.adapter.exec(self.pod or "", script, timeout=900)
        report = None
        for line in reversed((result.stdout or "").strip().splitlines()):
            if line.startswith("{"):
                try:
                    report = json.loads(line)
                    break
                except ValueError:
                    continue
        if report is None:
            raise ActionFailed(
                "RUNTIME_DRIFT",
                "the drift precheck produced no report: "
                f"{(result.stdout + result.stderr)[-500:]}",
            )
        self.write(
            "engine_core_drift_precheck.json",
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        )
        self.checks["engine_core_drift"] = report.get("state")
        drifts = [
            c for c in report.get("checks", [])
            if c.get("verdict") == "DRIFT"
            and c.get("scope", "path") == "path"
            and c.get("gating", "gate") == "gate"
        ]
        if drifts:
            preview = "; ".join(f"{c['id']}: {c['detail']}" for c in drifts[:8])
            summary = report.get("summary", {})
            context = ""
            if summary.get("drift_report_only"):
                context += (f" ({summary['drift_report_only']} further "
                            "conditional-branch finding(s) reported, not gating)")
            if summary.get("drift_out_of_path"):
                context += (f" ({summary['drift_out_of_path']} drift(s) in other "
                            "models' files, reported but not gating)")
            raise ActionFailed(
                "RUNTIME_DRIFT",
                f"{len(drifts)} engine-core-init drift(s) on this deployment's "
                "init surface, found before any weights loaded; apply "
                "tools/patches/patch_vllm_kunlun_drift.py in the pod and "
                f"re-run{context}: {preview}",
            )
        self.record(
            "engine_core_drift_precheck", True, json.dumps(report.get("summary", {}))
        )

    def verify_backend(self, prefix: str = "") -> None:
        pod = self.pod or ""
        backend = self.contract["checks"].get("backend", {})
        runtime = self.contract.get("context", {}).get("runtime", {})
        expected = backend.get("expected") or runtime.get("backend") or "kunlun"
        log = self.adapter.exec(pod, f"cat {self.server_log_path()}", timeout=120).stdout
        self.write(f"{prefix}server_log.txt" if prefix else "server_log.txt", log)
        lowered = log.lower()
        self.checks["expected_backend"] = expected if expected in lowered else "unknown"
        runtime = getattr(self, "runtime", None)
        markers = getattr(runtime, "fallback_markers", lambda: FALLBACK_MARKERS)()
        fallbacks = [marker for marker in markers if marker in lowered]
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
            "KUBECONFIG={kubeconfig} python3 cli/deployment/proof.py {contract} --execute\n".format(
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
            "artifact_root": str(self.artifact_dir.resolve()),
            # Which in-pod log this attempt wrote; a reproof (triage rerun)
            # records its own .rerun-* path here, so the original attempt's
            # log can always be identified from the status that produced it.
            "server_log": self.server_log_path(),
            "checks": self.checks,
            "artifacts": sorted(str(p.name) for p in self.artifact_dir.iterdir()),
        }
        if reason:
            status["reason"] = reason
            self.write("diagnosis.json", json.dumps({"state": state, "error": reason, "pod": self.pod, "next_action": "DIAGNOSE_RUNTIME"}, indent=2) + "\n")
        self.write("execution_records.json", json.dumps(self.records, indent=2, ensure_ascii=False) + "\n")
        attempt = locate_attempt(self.artifact_dir)
        inventory = ArtifactStore(attempt.root if attempt else self.artifact_dir)
        identity = attempt.identity if attempt else {"task_id": status["task_id"], "attempt_id": self.attempt_id}
        status["manifest_path"] = str(inventory.root / "manifest.json")
        status["artifacts"] = sorted({
            *(p.name for p in self.artifact_dir.iterdir()), "status.json",
        })
        self.store.write_json("status.json", status)
        try:
            declared_paths = [self.store.path(name) for name in status["artifacts"]]
            inventory.register(
                identity=identity, outcome=state,
                required=[
                    path.relative_to(inventory.root).as_posix()
                    for path in declared_paths if not path.is_dir()
                ],
            )
        except (ValueError, OSError) as error:
            status.update(state="BLOCKED", reason=str(error), manifest_path=None)
            try:
                self.store.write_json("status.json", status, overwrite=True)
            except (ValueError, OSError) as publication_error:
                raise error from publication_error
            raise
        return status

    # ---- main loop --------------------------------------------------------
    def run(self) -> dict[str, Any]:
        if self.store.path("status.json").exists():
            raise WritePolicyError("deployment output already has a formal result; use a fresh directory")
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
            # Protocol hard rule (AGENTS.md): repairs must be replayable. The
            # repo's idempotent patch set is applied after any install or
            # attach, and the drift precheck that follows verifies the result
            # rather than trusting it — a reinstalled pod self-heals here.
            self.apply_runtime_patches()
            if self.phase == "environment":
                self.verify_runtime_importable()
                # Before the first server launch: the whole point of the
                # precheck is that a drifted surface is seen here, seconds
                # after install, instead of one 15-minute reload per drift.
                self.engine_core_drift_precheck()
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
            # Before any (re)launch, including the service phase on an
            # attached pod: a pod that was reinstalled since the environment
            # proof lost its site-packages repairs, and this is the cheap
            # place to learn that — not after another full weight load.
            self.engine_core_drift_precheck()
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
            f"{self.runtime.env_prefix()}; "
            f"{self.runtime.import_version_command()}"
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
            f"{self.runtime.env_prefix()}; "
            f"{self.runtime.environment_fingerprint_command(self.workdir)}; "
            "echo '## driver'; xpu_smi | sed -n '3p'; "
            "echo '## os'; . /etc/os-release && echo \"$PRETTY_NAME\"; gcc --version | head -1"
        )
        result = self.adapter.exec(pod, script, timeout=300)
        self.write("environment_fingerprint.txt", result.stdout + result.stderr)
        self.record("environment_fingerprint", result.returncode == 0)
