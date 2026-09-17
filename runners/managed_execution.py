"""Supervise cooperating local processes with durable identity and lease ownership.

This is not a sandbox. A remote transport's exit is not remote termination; a
remote driver must supply its own trusted observer before it can use this API.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import time

from core.storage import ensure_external, locate_attempt
from engine.execution import (
    begin_execution, canonical_resource, finish_execution, get_execution, heartbeat,
    mark_started, reconcile_execution,
)


def host_identity() -> str:
    """Host plus boot identity: another machine/reboot cannot certify PID absence."""
    boot = Path("/proc/sys/kernel/random/boot_id")
    if boot.is_file():
        value = boot.read_text().strip()
    elif platform.system() == "Darwin":
        value = subprocess.check_output(["sysctl", "-n", "kern.boottime"], text=True).strip()
    else:
        raise ValueError("managed execution requires a supported process-identity observer")
    return hashlib.sha256(f"{platform.node()}:{value}".encode()).hexdigest()


def process_identity(pid: int) -> str | None:
    """Read process birth identity, not just the reusable numeric PID."""
    proc = Path(f"/proc/{pid}/stat")
    if platform.system() == "Linux":
        try:
            # comm can contain spaces/parentheses; the remaining fields start at 3.
            fields = proc.read_text().rsplit(")", 1)[1].split()
            return f"linux:{pid}:{fields[19]}"
        except FileNotFoundError:
            return None
    result = subprocess.run(["ps", "-p", str(pid), "-o", "lstart="],
                            text=True, capture_output=True, check=False)
    if result.returncode == 1 and not result.stdout.strip():
        return None
    if result.returncode or not result.stdout.strip():
        raise ValueError("cannot observe process birth identity")
    return f"{platform.system()}:{pid}:{result.stdout.strip()}"


def _group_alive(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False


def _terminate_owned(process, identity: str | None) -> bool:
    """Never signal a recovered, unverified or reused PID."""
    if process.poll() is None:
        if identity is None or process_identity(process.pid) != identity:
            return False
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            if process_identity(process.pid) != identity:
                return False
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=2)
    return not _group_alive(process.pid)


def _execution_scope(scheduler, run_id, task_id, root, resource):
    """Check caller-selected paths and Pod identity before reserving or spawning."""
    run = scheduler.store.run(run_id)
    if run is None:
        raise ValueError(f"unknown run: {run_id}")
    owned = locate_attempt(root)
    run_root = run.metadata.get("artifact_root")
    if (owned is None or owned.identity["run_id"] != run_id or not run_root
            or not root.is_relative_to(ensure_external(run_root))):
        raise ValueError("execution output must belong to a managed attempt of this run")
    if not any(root.is_relative_to(directory) for directory in
               (owned.output, owned.scratch, owned.logs)):
        raise ValueError("execution output must be under attempt output, scratch, or logs")
    if task_id is not None:
        task = scheduler.store.get_task(task_id)
        if task is None or task.run_id != run_id:
            raise ValueError("execution task does not belong to run")
        workspace = task.input.get("workspace", {})
        if (owned.identity["task_id"] != task_id or not workspace.get("root")
                or owned.root != Path(workspace["root"]).resolve()):
            raise ValueError("execution output must belong to the current claimed task attempt")
        if task.stage in {"xpu", "integration"}:
            from runners.managed_boundary import require_supported_runtime
            require_supported_runtime(run)
    selected = canonical_resource(resource)
    if selected is not None:
        if run.metadata.get("evidence_mode", "real") != "simulation":
            raise ValueError("real Pod resource execution requires a trusted remote process driver")
        bound = canonical_resource(run.environment.get("environment_proof", {}).get("resource_identity"))
        if run.status != "ENVIRONMENT_READY" or bound != selected:
            raise ValueError("execution resource must match the currently ready environment proof")


def execute_managed(scheduler, run_id, execution_id, argv, *, cwd, output_dir,
                    task_id=None, worker=None, lease_token=None, resource=None,
                    timeout=300, remote=False, env_overrides=None) -> dict:
    """Run once, persist before spawn, heartbeat even while stdout is silent.

    Repeating an ID returns its recorded state; UNKNOWN never implicitly reruns.
    Callers grade durable output separately -- exit zero is not correctness.
    """
    if remote:
        raise ValueError("remote execution requires a trusted remote process/termination driver")
    if (not isinstance(argv, (list, tuple)) or not argv or not argv[0]
            or not all(isinstance(value, str) and "\0" not in value for value in argv)):
        raise ValueError("argv must be a nonempty string list")
    if type(timeout) not in (float, int) or not 0 < timeout < float("inf"):
        raise ValueError("timeout must be finite and positive")
    if env_overrides is not None and (not isinstance(env_overrides, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) or "\0" in key + value or "=" in key
        for key, value in env_overrides.items()
    )):
        raise ValueError("environment overrides must map string names to string values")
    if lease_token and lease_token in json.dumps(env_overrides or {}, sort_keys=True):
        raise ValueError("controller lease token must not be forwarded to the child environment")
    root = ensure_external(output_dir)
    workdir = Path(cwd).resolve(strict=True)
    if not workdir.is_dir():
        raise ValueError("execution cwd must be a directory")
    owner = {"worker": worker, "lease_token": lease_token}
    payload = {"argv": list(argv), "cwd": str(workdir), "output_dir": str(root),
               "timeout": timeout, "remote": False,
               "environment_sha256": hashlib.sha256(json.dumps(
                   env_overrides or {}, sort_keys=True).encode()).hexdigest()}
    # A completed/unknown historical invocation is a receipt lookup, not
    # permission to execute against whatever attempt happens to be current.
    with scheduler.store.transaction():
        existing = get_execution(scheduler.store, run_id, execution_id)
        if existing is None:
            _execution_scope(scheduler, run_id, task_id, root, resource)
        record, fresh = begin_execution(scheduler, run_id, execution_id, payload,
                                       task_id=task_id, resource=resource, **owner)
    if not fresh:
        return record
    log = root / "console.log"
    process = None
    identity = None
    error = None
    confirmed = False
    spawn_attempted = False
    try:
        if task_id is not None:
            # Even a nearly expired initial claim must cover process startup.
            scheduler.renew_lease(task_id, worker, lease_token, 300)
        boot_identity = host_identity()
        root.mkdir(parents=True, exist_ok=True)
        with log.open("x", encoding="utf-8") as stream:
            environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            environment.update(env_overrides or {})
            spawn_attempted = True
            process = subprocess.Popen(argv, cwd=workdir, env=environment,
                                       stdin=subprocess.DEVNULL, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            identity = process_identity(process.pid)
            # A very short-lived child may exit before observation. wait/poll
            # still establishes its status; mark_started records the birth gap.
            mark_started(scheduler, run_id, execution_id, pid=process.pid,
                         process_identity=identity or "exited-before-observation",
                         host_id=boot_identity, **owner)
            started = time.monotonic()
            last_renewal = started
            last_heartbeat = started - 1
            while process.poll() is None:
                if time.monotonic() - started >= timeout:
                    raise TimeoutError(f"managed execution exceeded {timeout}s")
                if task_id and time.monotonic() - last_renewal >= 1:
                    scheduler.renew_lease(task_id, worker, lease_token, 300)
                    last_renewal = time.monotonic()
                if time.monotonic() - last_heartbeat >= 1:
                    heartbeat(scheduler, run_id, execution_id, **owner)
                    last_heartbeat = time.monotonic()
                time.sleep(0.1)
            confirmed = not _group_alive(process.pid)
            if not confirmed:
                error = "leader exited but process group is still active; termination is unknown"
    except (Exception, KeyboardInterrupt) as exc:
        error = f"{type(exc).__name__}: {exc}"
        if process is None:
            # An interrupt during Popen can leave a child without returning its
            # handle. Preserve STARTING occupancy as UNKNOWN in that window.
            confirmed = not spawn_attempted
        else:
            try:
                confirmed = _terminate_owned(process, identity)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                confirmed = False
    returncode = process.poll() if process is not None else None
    state = "UNKNOWN" if not confirmed else "SUCCEEDED" if returncode == 0 and not error else "FAILED"
    return finish_execution(scheduler, run_id, execution_id, state=state,
                            returncode=returncode, log_path=str(log), error=error,
                            termination_confirmed=confirmed, **owner)


def reconcile_local_execution(scheduler, run_id, execution_id):
    """Read-only OS observation; never kills or accepts a user-authored PASS."""
    def observe(record):
        result = {"execution_id": execution_id, "terminal": False,
                  "termination_confirmed": False, "state": "UNKNOWN"}
        if record.get("host_id") != host_identity() or record.get("remote_handle"):
            return result
        pid = record.get("pid")
        if not isinstance(pid, int) or record.get("payload", {}).get("remote"):
            return result
        # Missing leader alone is insufficient: its children may still own XPU.
        if process_identity(pid) is None and not _group_alive(pid):
            result.update(terminal=True, termination_confirmed=True, state="FAILED",
                          error="controller interrupted; local process group confirmed absent")
        return result
    return reconcile_execution(scheduler, run_id, execution_id, observer=observe)
