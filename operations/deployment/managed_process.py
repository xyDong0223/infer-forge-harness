"""Fixed, standalone process supervision protocol; not a task acceptance gate.

Linux is the Pod target. Darwin support exercises the real process protocol in
local tests; its ps birth timestamp has lower precision. This is a cooperating
process-group boundary, not a sandbox: commands must not escape their session,
modify protocol files, or place credentials in persisted argv. No Pod identity
is attested here, and no execution ledger is updated by this module.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import stat
import subprocess
import sys
import time
import traceback
from uuid import uuid4

from core.paths import REPO_ROOT
from core.storage import ensure_external


TERMINAL = frozenset({"EXITED", "CANCELLED"})
_CHILD_ENV = {"PATH": os.defpath, "LANG": "C", "PYTHONDONTWRITEBYTECODE": "1",
              "PYTHONNOUSERSITE": "1"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _read(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "r", encoding="utf-8") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > 1024 * 1024:
            raise ValueError(f"protocol input must be a regular JSON file smaller than 1 MiB: {path}")
        return json.load(handle, object_pairs_hook=_object,
                         parse_constant=lambda value: (_ for _ in ()).throw(
                             ValueError(f"non-finite JSON constant: {value}")))


def _path(value, label):
    if not isinstance(value, (str, Path)) or not str(value) or "\0" in str(value):
        raise ValueError(f"{label} must be an absolute canonical external path")
    path = Path(value)
    if not path.is_absolute() or path != path.resolve() or path.is_symlink():
        raise ValueError(f"{label} must be an absolute canonical external path without symlinks")
    return ensure_external(path)


def load_request(path):
    """Read strict external JSON. The controller must never put secrets in argv."""
    return validate_request(_read(_path(path, "request")))


def validate_request(request):
    fields = {"schema_version", "execution_id", "nonce", "argv", "cwd", "timeout_seconds"}
    if not isinstance(request, dict) or set(request) != fields:
        raise ValueError(f"request must contain exactly: {', '.join(sorted(fields))}")
    if type(request["schema_version"]) is not int or request["schema_version"] != 1:
        raise ValueError("unsupported request schema_version")
    if (not isinstance(request["execution_id"], str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}", request["execution_id"])):
        raise ValueError("invalid execution_id")
    if (not isinstance(request["nonce"], str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{16,128}", request["nonce"])):
        raise ValueError("nonce must have 16-128 letters, digits, underscores or hyphens")
    argv = request["argv"]
    if (not isinstance(argv, list) or not 1 <= len(argv) <= 512
            or any(not isinstance(part, str) or "\0" in part or len(part) > 65536 for part in argv)
            or not argv[0].strip()):
        raise ValueError("argv must be a bounded nonempty list of NUL-free strings")
    timeout = request["timeout_seconds"]
    try:
        valid_timeout = type(timeout) in (int, float) and math.isfinite(timeout) and 0 < timeout <= 86400
    except OverflowError:
        valid_timeout = False
    if not valid_timeout:
        raise ValueError("timeout_seconds must be finite, positive, nonboolean, and at most 86400")
    result = {**request, "argv": list(argv), "cwd": str(_path(request["cwd"], "cwd"))}
    if len(json.dumps(result)) > 1024 * 1024:
        raise ValueError("request exceeds 1 MiB")
    return result


def _atomic(root, name, value):
    destination = root / name
    if destination.is_symlink():
        raise ValueError(f"protocol file cannot be a symlink: {destination}")
    pending = root / f".{name}.{uuid4().hex}.pending"
    try:
        with pending.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, destination)
        fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        pending.unlink(missing_ok=True)


@contextmanager
def _lock(root, *, create=False, name="protocol.lock", nonblocking=False):
    if create:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_RDWR | os.O_NOFOLLOW | (os.O_CREAT if create else 0)
    fd = os.open(root / name, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("protocol lock must be a regular file")
        fcntl.flock(fd, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        yield
    finally:
        os.close(fd)


def _binding(request):
    return {"schema_version": 1, "execution_id": request["execution_id"],
            "nonce": request["nonce"], "request_sha256": _digest(request)}


def _initialize(root, request):
    path = root / "request.json"
    if path.exists() or path.is_symlink():
        envelope = _read(path)
        if envelope != {**_binding(request), "request": request}:
            raise ValueError("execution root already belongs to a different request/hash/nonce")
        return False
    if set(path.name for path in root.iterdir()) != {"protocol.lock"}:
        raise ValueError("execution root must be fresh and empty")
    _atomic(root, "request.json", {**_binding(request), "request": request})
    _atomic(root, "state.json", {
        **_binding(request), "state": "STARTING", "created_at": time.time(),
        "updated_at": time.time(), "revision": 1, "monitor": None, "process": None,
        "monitor_started": False, "spawn_intent": False, "returncode": None,
        "terminal": False, "termination_confirmed": False,
    })
    return True


def _state(root, request):
    value = _read(root / "state.json")
    if not isinstance(value, dict) or any(value.get(k) != v for k, v in _binding(request).items()):
        raise ValueError("state is not bound to this request")
    return value


def _update(root, previous, **changes):
    result = {**previous, **changes, "updated_at": time.time(), "revision": previous["revision"] + 1}
    _atomic(root, "state.json", result)
    return result


def _boot_identity():
    if sys.platform == "linux":
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if sys.platform == "darwin":
        return subprocess.check_output(["/usr/sbin/sysctl", "-n", "kern.boottime"], text=True).strip()
    raise ValueError("managed process protocol supports Linux and local Darwin tests only")


def process_identity(pid):
    """Process birth, session and process group, never a numeric PID alone."""
    if type(pid) is not int or pid <= 0:
        raise ValueError("pid must be a positive integer")
    if sys.platform == "linux":
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        except FileNotFoundError:
            return None
        if fields[0] == "Z":
            return None
        birth, pgid, sid = fields[19], int(fields[2]), int(fields[3])
    elif sys.platform == "darwin":
        observed = subprocess.run(["/bin/ps", "-p", str(pid), "-o", "lstart=", "-o", "state="],
                                  capture_output=True, text=True, check=False)
        if observed.returncode == 1 and not observed.stdout.strip():
            return None
        if observed.returncode or not observed.stdout.strip():
            raise ValueError("cannot observe process birth timestamp")
        fields = observed.stdout.strip().split()
        if fields[-1].startswith("Z"):
            return None
        birth = " ".join(fields[:-1])
        try:
            pgid, sid = os.getpgid(pid), os.getsid(pid)
        except ProcessLookupError:
            return None
    else:
        raise ValueError("unsupported process identity platform")
    return {"pid": pid, "birth": birth, "pgid": pgid, "sid": sid,
            "boot_id": _boot_identity(), "platform": platform.system()}


def _group_alive(pid):
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # EPERM is not proof of absence. Keep occupancy and let subsequent
        # observations establish ESRCH rather than abandoning the monitor.
        return True


def _signal_owned(identity, signum):
    if (not identity or identity.get("pid") != identity.get("pgid")
            or identity.get("pid") != identity.get("sid")
            or process_identity(identity["pid"]) != identity):
        return False
    try:
        os.killpg(identity["pid"], signum)
    except ProcessLookupError:
        return False
    return True


def _tombstone(root, request, reason):
    path = root / "cancel.json"
    if path.exists() or path.is_symlink():
        value = _read(path)
        if any(value.get(k) != v for k, v in _binding(request).items()):
            raise ValueError("cancellation is not bound to this request")
        return value
    value = {**_binding(request), "requested_at": time.time(), "reason": reason}
    _atomic(root, "cancel.json", value)
    return value


def _finish(root, state, *, cancelled, returncode, reason):
    value = {**state, "state": "CANCELLED" if cancelled else "EXITED",
             "terminal": True, "termination_confirmed": True, "returncode": returncode,
             "reason": reason, "finished_at": time.time(), "updated_at": time.time(),
             "revision": state["revision"] + 1}
    log = root / "console.log"
    if log.is_symlink():
        raise ValueError("console log cannot be a symlink")
    value["log_sha256"] = None
    if log.exists():
        fd = os.open(log, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("console log must be a regular file")
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        value["log_sha256"] = digest.hexdigest()
    receipt = root / "exit_receipt.json"
    if receipt.exists():
        return _read(receipt)
    # Receipt is published first so a crash before state update loses no exit.
    _atomic(root, "exit_receipt.json", value)
    _atomic(root, "state.json", value)
    return value


def _observe_locked(root, request):
    envelope = _read(root / "request.json")
    if envelope != {**_binding(request), "request": request}:
        raise ValueError("execution root already belongs to a different request/hash/nonce")
    receipt = root / "exit_receipt.json"
    value = _read(receipt) if receipt.exists() or receipt.is_symlink() else _state(root, request)
    if (not isinstance(value, dict)
            or any(value.get(k) != v for k, v in _binding(request).items())):
        raise ValueError("observation is not bound to this request")
    if receipt.exists():
        if value.get("state") not in TERMINAL or value.get("termination_confirmed") is not True:
            raise ValueError("invalid exit receipt")
    else:
        value = {**value, "terminal": False, "termination_confirmed": False}
        monitor_identity = value.get("monitor")
        if (value.get("state") in TERMINAL or not monitor_identity
                or process_identity(monitor_identity["pid"]) != monitor_identity):
            value.update(state="UNKNOWN", reason="no live matching monitor or durable exit receipt")
    return {**value, "cancel_requested": (root / "cancel.json").exists(),
            "root": str(root), "log_path": str(root / "console.log"),
            "receipt_path": str(receipt) if receipt.exists() else None,
            "task_verdict": None, "ledger_integrated": False}


def inspect(root, request):
    """Read-only observation. Missing state never proves that nothing started."""
    root, request = _path(root, "root"), validate_request(request)
    try:
        with _lock(root):
            return _observe_locked(root, request)
    except FileNotFoundError:
        return {**_binding(request), "state": "UNKNOWN", "terminal": False,
                "termination_confirmed": False, "reason": "protocol state is missing",
                "root": str(root), "task_verdict": None, "ledger_integrated": False}


def start(root, request):
    """Reserve before detached monitor launch; an identical request never respawns."""
    root, request = _path(root, "root"), validate_request(request)
    _boot_identity()  # Unsupported platforms must fail before reserving.
    with _lock(root, create=True):
        if not (root / "request.json").exists() and not Path(request["cwd"]).is_dir():
            raise ValueError("cwd must be an existing external directory")
        fresh = _initialize(root, request)
        if fresh:
            command = [sys.executable, "-B", str(REPO_ROOT / "cli/deployment/managed_process.py"),
                       "_monitor", "--root", str(root)]
            try:
                with (root / "monitor.log").open("x", encoding="utf-8") as stream:
                    child = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                             stdout=stream, stderr=subprocess.STDOUT,
                                             start_new_session=True, close_fds=True,
                                             cwd=REPO_ROOT, env=_CHILD_ENV)
                identity = process_identity(child.pid)
                _update(root, _state(root, request), monitor=identity)
            except (Exception, KeyboardInterrupt) as error:
                _update(root, _state(root, request), state="UNKNOWN",
                        reason=f"monitor startup uncertain: {type(error).__name__}: {error}")
    return inspect(root, request)


def cancel(root, request):
    """Persist cancellation before signalling; even a delayed start must obey it."""
    root, request = _path(root, "root"), validate_request(request)
    with _lock(root, create=True):
        _initialize(root, request)
        if (root / "exit_receipt.json").exists():
            return _observe_locked(root, request)
        state = _state(root, request)
        marker = _tombstone(root, request, "requested")
        identity = state.get("process")
        if not state.get("spawn_intent"):
            _finish(root, state, cancelled=True, returncode=None, reason="cancelled before process spawn")
        elif identity:
            actual = process_identity(identity["pid"])
            if actual is None and not _group_alive(identity["pid"]):
                _finish(root, state, cancelled=True, returncode=None,
                        reason="cancellation observed original process group absent")
            # The original monitor may be dead. Repeated explicit cancellation
            # must still escalate, without resetting the durable grace period
            # or signalling any process whose birth/session/group has changed.
            elif not _signal_owned(
                identity, signal.SIGKILL if time.time() - marker["requested_at"] >= 2 else signal.SIGTERM,
            ):
                _update(root, state, state="UNKNOWN",
                        reason="cannot signal without matching original process birth/session/group")
        else:
            _update(root, state, state="UNKNOWN", reason="spawn was attempted without a durable process identity")
    return inspect(root, request)


def monitor(root):
    """Internal detached entrypoint. Owns one child session, never replays it."""
    root = _path(root, "root")
    envelope = _read(root / "request.json")
    request = validate_request(envelope["request"])
    process = None
    try:
        with _lock(root, create=True, name="monitor.lock", nonblocking=True):
            with _lock(root):
                _initialize(root, request)
                state = _state(root, request)
                if state.get("monitor_started") or state.get("spawn_intent") or state.get("state") in TERMINAL:
                    return 0
                state = _update(root, state, monitor_started=True,
                                monitor=process_identity(os.getpid()))
                if (root / "cancel.json").exists():
                    _tombstone(root, request, "requested")
                    _finish(root, state, cancelled=True, returncode=None, reason="cancelled before process spawn")
                    return 0
                state = _update(root, state, spawn_intent=True)
                with (root / "console.log").open("x", encoding="utf-8") as stream:
                    process = subprocess.Popen(request["argv"], cwd=request["cwd"], env=_CHILD_ENV,
                                               stdin=subprocess.DEVNULL, stdout=stream,
                                               stderr=subprocess.STDOUT, start_new_session=True,
                                               close_fds=True)
                identity = process_identity(process.pid)
                # A child may exit before /proc/ps observes its birth. Only the
                # live Popen handle may then certify wait + absent process group.
                state = _update(root, state, state="RUNNING", process=identity,
                                spawned_pid=process.pid)
            begun, stopping = time.monotonic(), None
            while True:
                returncode = process.poll()
                with _lock(root):
                    state = _state(root, request)
                    if (root / "exit_receipt.json").exists():
                        return 0
                    expired = time.monotonic() - begun >= request["timeout_seconds"]
                    cancellation = ((root / "cancel.json").exists() or expired)
                    if cancellation:
                        marker = _tombstone(root, request, "timeout" if expired else "requested")
                    if returncode is not None and not _group_alive(process.pid):
                        _finish(root, state, cancelled=cancellation, returncode=returncode,
                                reason=marker["reason"] if cancellation else "observed child exit and absent group")
                        return 0
                    if cancellation:
                        stopping = stopping if stopping is not None else time.monotonic()
                        signum = signal.SIGKILL if time.monotonic() - stopping >= 2 else signal.SIGTERM
                        if not _signal_owned(identity, signum):
                            if state.get("state") != "UNKNOWN":
                                _update(root, state, state="UNKNOWN",
                                        reason="leader identity unavailable; process group is still occupied")
                            if time.monotonic() - stopping >= 3:
                                return 2  # Never release an unidentified/still-live group.
                    elif returncode is not None and state.get("state") != "UNKNOWN":
                        _update(root, state, state="UNKNOWN", returncode=returncode,
                                reason="leader exited but process group is still occupied")
                time.sleep(0.1)
    except BlockingIOError:
        return 0  # An existing monitor already owns the execution.
    except (Exception, KeyboardInterrupt) as error:
        # The detached CLI's stderr is the private monitor.log, not a shareable
        # report. Preserve the stack for diagnosis while keeping state concise.
        traceback.print_exc()
        with _lock(root):
            if not (root / "exit_receipt.json").exists():
                state = _state(root, request)
                _update(root, state, state="UNKNOWN", reason=f"monitor interrupted: {type(error).__name__}: {error}")
        return 2
