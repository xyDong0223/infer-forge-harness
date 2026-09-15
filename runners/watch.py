"""Background watch: heartbeat journal, so "still running" is never silent.

Run glm52-int-w8a8-p800-001 had four "any progress?" interruptions during
15-minute silent stretches — a 707 GiB load, a health poll answering 503, a
node whose console legitimately prints nothing until it finishes. Nothing
durable existed in between: no record that the runner was alive, how long it
had been running, or whether the thing being watched was moving.

Two watches close that gap:

* ``LogWatch`` (local): a daemon thread that journals a heartbeat every
  interval while a child runs — elapsed, watched-log bytes, last line —
  plus a final entry carrying the outcome. A quiet console is *expected*
  during a long load, so the local watch records heartbeats, it does not
  claim a hang.
* The pod-side startup watch (runners/deployment_proof.py): each health
  poll journals the server log's size and last line, and flags a *stall*
  when the log stops growing well before the deadline — a vLLM loader
  that is loading prints; a frozen log is the cheap, early signal that
  the wait will not end well.

The journal is append-only JSONL, one line per beat, in the node's own
artifact directory — post-hoc "what happened during the silent 15 minutes"
without touching any live file.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

_LAST_LINE_CHARS = 160


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log_digest(path: Path) -> tuple[int, str]:
    """(bytes, last line) of the watched log, cheap and failure-tolerant."""
    try:
        size = path.stat().st_size
    except OSError:
        return -1, ""
    try:
        with path.open("rb") as handle:
            handle.seek(max(0, size - 8192))
            tail = handle.read().decode("utf-8", errors="replace")
        last = tail.strip().splitlines()[-1] if tail.strip() else ""
        return size, last[-_LAST_LINE_CHARS:]
    except OSError:
        return size, ""


class LogWatch:
    """Heartbeat journal for one running child process.

    ``observe()`` is called by the reader loop on every output line; the
    daemon thread writes a heartbeat every ``interval`` seconds regardless,
    because a silent console is the normal shape of a long load, and the
    journal's job is to prove the runner is alive and how far along the
    watched log is — not to guess intent from silence.
    """

    def __init__(self, name: str, log_path: Path, journal_path: Path,
                 interval: float = 30.0, echo: Callable[[str], None] | None = print,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.name = name
        self.log_path = Path(log_path)
        self.journal_path = Path(journal_path)
        self.interval = float(interval)
        self._echo = echo
        self._sleep = sleep
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0
        self._observations = 0
        self._beats = 0
        self._stopped = False
        self._summary: dict | None = None

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> "LogWatch":
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self._started = time.monotonic()
        self._thread = threading.Thread(
            target=self._loop, name=f"watch:{self.name}", daemon=True
        )
        self._thread.start()
        return self

    def stop(self, outcome: str) -> dict:
        """Final entry and summary. Idempotent."""
        if self._stopped and self._summary is not None:
            return self._summary
        self._stopped = True
        if self._thread is not None and self._thread.is_alive():
            self._stop.set()
            self._thread.join(timeout=max(1.0, self.interval * 2))
        self._thread = None
        duration = round(time.monotonic() - self._started, 1) if self._started else 0.0
        bytes_now, last = _log_digest(self.log_path)
        self._journal({
            "phase": "final", "outcome": outcome, "log_bytes": bytes_now,
            "last_line": last, "observations": self._observations,
            "beats": self._beats, "elapsed_s": duration,
        })
        self._summary = {"beats": self._beats,
                         "observations": self._observations,
                         "duration_s": duration,
                         "journal": str(self.journal_path)}
        return self._summary

    # -- progress signalling ------------------------------------------------
    def observe(self) -> None:
        """The watched activity produced output; a heartbeat lands now."""
        self._observations += 1
        self._journal({
            "phase": "progress", "observations": self._observations,
            "elapsed_s": round(time.monotonic() - self._started, 1),
        })

    # -- internals ----------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self._beat()

    def _beat(self) -> None:
        bytes_now, last = _log_digest(self.log_path)
        self._journal({
            "phase": "beat", "beats": self._beats,
            "elapsed_s": round(time.monotonic() - self._started, 1),
            "log_bytes": bytes_now, "last_line": last,
        })
        if self._echo:
            elapsed = int(round(time.monotonic() - self._started))
            self._echo(
                f"[watch] {self.name}: {elapsed}s elapsed, "
                f"console {bytes_now} B, {self._observations} output lines"
            )

    def _journal(self, entry: dict) -> None:
        if entry.get("phase") == "beat":
            self._beats += 1
            entry["beats"] = self._beats
        entry = {"at": now_iso(), "watch": self.name, **entry}
        try:
            with self.journal_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:  # never let a journal problem kill the watched child
            pass
