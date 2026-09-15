"""Crash-first evidence snapshots.

A harness that calls itself evidence-driven must treat a dying subprocess's
output as evidence, not console noise. Run glm52-int-w8a8-p800-001 (2026-09-14)
lost two kinds of proof that later diagnosis needed:

* the graph runner let node commands inherit its own stdout, so a crashing
  node's traceback scrolled past and was never persisted anywhere;
* every re-entry — a recovery rerun, the MAT-006 triage's service reproof, a
  manual relaunch — rewrote the same artifact names and the same in-pod
  ``server.log`` path, so the attempt that had just failed was overwritten
  seconds after it died: one engine stack trace was destroyed 40 s after the
  crash, by the triage that existed to explain it.

The rule this module implements: when a process dies, its log is archived to a
path that cannot be overwritten, immediately, before any failure edge or rerun
can touch the live files. Paths produced here only ever gain a new suffixed
file (``name.1``, ``name.2``, ...); an existing file is never replaced.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


def unique_path(path: Path) -> Path:
    """The first path at or after ``path`` that does not exist yet."""
    if not path.exists():
        return path
    n = 1
    while True:
        candidate = path.with_name(f"{path.name}.{n}")
        if not candidate.exists():
            return candidate
        n += 1


def write_unique(path: Path, content: str) -> Path:
    """Write evidence no later attempt can overwrite.

    ``write`` replaces; this snapshots. Crash evidence named through here
    survives every rerun that writes the same base name again.
    """
    target = unique_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


def archive_before_truncate(remote_log: str) -> str:
    """Shell snippet: park ``remote_log`` at the next free ``.prev-N``.

    Runs before a relaunch truncates the live log. A single fixed ``.prev``
    name (the old behaviour) only protected one generation: a third attempt
    destroyed the second's evidence while "archiving" it.
    """
    return (
        f"n=1; while [ -e '{remote_log}.prev-$n' ]; do n=$((n+1)); done; "
        f"cp '{remote_log}' '{remote_log}.prev-$n' 2>/dev/null || true; "
    )


def archive_crash_remote(remote_log: str) -> str:
    """Shell snippet: copy ``remote_log`` to the next free ``.crash-N``.

    Used at the moment a crash is detected, so the pod-side copy survives even
    a manual relaunch someone performs inside the pod before the executor gets
    another look.
    """
    return (
        f"n=1; while [ -e '{remote_log}.crash-$n' ]; do n=$((n+1)); done; "
        f"cp '{remote_log}' '{remote_log}.crash-$n' 2>/dev/null || true; "
    )


@dataclass
class LoggedResult:
    returncode: int
    console_log: Path
    crash_log: Path | None


def run_logged(
    command: Iterable[str],
    cwd: Path,
    log_path: Path,
    crash_tag: str | None = None,
    echo: Callable[[str], None] = print,
    watch=None,
) -> LoggedResult:
    """Run a node command with its output teed live to console and file.

    Live streaming matters: a 15-minute 707 GiB bring-up must show progress,
    not go silent behind a captured pipe (the same run's monitoring gap).
    PYTHONUNBUFFERED is forced because a child Python block-buffers stdout the
    moment it is a pipe, which would defeat the streaming.

    On a non-zero exit the captured log is immediately copied to a
    non-overwritable crash snapshot — before any failure edge, recovery rerun
    or manual relaunch can replace the live log file.

    ``watch`` (runners.watch.LogWatch) is signalled on every output line; its
    heartbeat thread keeps journaling even when the console is legitimately
    quiet, so "still running" is never a silent state.
    """
    command = list(command)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        text=True,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    chunks: list[str] = []
    with log_path.open("w", encoding="utf-8") as handle:
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            echo(line, end="")
            chunks.append(line)
            if watch is not None:
                watch.observe()
    returncode = process.wait()
    crash_log = None
    if returncode != 0 and crash_tag:
        crash_dir = log_path.parent / "crash"
        crash_log = write_unique(crash_dir / f"{crash_tag}.crash.log", "".join(chunks))
    return LoggedResult(returncode=returncode, console_log=log_path, crash_log=crash_log)
