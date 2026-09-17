"""Cooperating local Graph controllers serialize per state database and run.

This is not a Pod lock or an OS sandbox. A kernel-held advisory lock prevents
two live local controllers from accepting/executing decisions simultaneously;
durable decision receipts separately prevent replay after a process crash.
"""

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import threading

from core.storage import ensure_external


_local = threading.local()


@contextmanager
def control_run(state, run_id):
    if not run_id:
        raise ValueError("Graph control requires a run_id")
    state = ensure_external(state)
    key = (os.getpid(), str(state), run_id)
    held = getattr(_local, "held", None)
    if held is None:
        held = _local.held = set()
    if key in held:
        yield
        return
    try:
        import fcntl
    except ImportError as exc:
        raise ValueError("local Graph control currently requires POSIX advisory locks") from exc
    digest = hashlib.sha256(f"{state}\0{run_id}".encode()).hexdigest()
    directory = ensure_external(Path(state).parent / ".infer-forge-control")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}.lock"
    with path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"another local Graph controller is active for run {run_id}") from exc
        held.add(key)
        try:
            yield
        finally:
            held.remove(key)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
