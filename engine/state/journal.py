"""Journal: make evidence retrievable across Tasks.

Until now each Task wrote an isolated directory under the artifact root, so the
only way to feed one Task's output into the next was to hand it a path. That makes
two things impossible: knowing whether a fact already exists, and knowing whether
it is still about the current world.

A fact is (kind, subject, environment fingerprint) -> artifact bundle. The
fingerprint is what makes a hit trustworthy: the same model on the same stack
commit is the same fact, and a different stack commit is a different one, even if
the model is identical. Without it, a cached fact would silently answer for an
environment nobody validated.
"""

from __future__ import annotations

import hashlib
import json
import fcntl
import os
from contextlib import contextmanager
from pathlib import Path

from core.paths import REPO_ROOT

from core.storage import default_state_root, ensure_external

DEFAULT_JOURNAL = default_state_root() / "journal.jsonl"
from core.task_execution import default_execution_catalog

# Task definitions, not a parallel hand-maintained fact-name registry.
KINDS = {name: task.produces for name, task in default_execution_catalog().items()}


def fingerprint(environment: dict[str, str]) -> str:
    """Stable digest of the facts a conclusion is only true of."""
    canonical = json.dumps(environment, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def record(
    journal: Path,
    kind: str,
    subject: str,
    state: str,
    artifacts: Path,
    environment: dict[str, str],
    extra: dict | None = None,
    *,
    _locked: bool = False,
) -> dict:
    journal = ensure_external(journal)
    entry = {
        "kind": kind,
        "subject": subject,
        "state": state,
        "artifacts": str(artifacts),
        "environment": environment,
        "fingerprint": fingerprint(environment),
    }
    if extra:
        entry["detail"] = extra
    if _locked:
        _append(journal, entry)
    else:
        with locked(journal):
            _append(journal, entry)
    return entry


@contextmanager
def locked(journal: Path):
    """Serialize cooperating read/append transactions, including projection updates.

    Readers do not acquire this write lock or create a sidecar. The separate lock
    inode remains stable across an interrupted writer; the Journal is append-only.
    """
    journal = ensure_external(journal)
    journal.parent.mkdir(parents=True, exist_ok=True)
    lock_path = journal.with_name(journal.name + ".lock")
    with lock_path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield journal
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _append(journal: Path, entry: dict) -> None:
    with journal.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    # A newly created Journal must survive a crash along with its contents.
    directory = os.open(journal.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def load(journal: Path) -> list[dict]:
    if not journal.exists():
        return []
    return [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines() if line.strip()]


def query(
    journal: Path,
    kind: str,
    subject: str | None = None,
    environment: dict[str, str] | None = None,
    states: tuple[str, ...] = (),
) -> list[dict]:
    """Most recent first. An environment argument restricts to matching facts.

    A fact recorded under a different fingerprint is not returned even when the
    subject matches: reusing it would answer a question about this environment
    with evidence from another one. None is unfiltered inspection; {} returns
    no hits because it provides no runtime identity.
    """
    # None is an explicitly unfiltered inspection query. An empty runtime
    # context is not permission to reuse facts from every environment.
    if environment is not None and not environment:
        return []
    wanted = fingerprint(environment) if environment is not None else None
    hits = [
        entry
        for entry in load(journal)
        if entry["kind"] == kind
        and (subject is None or entry["subject"] == subject)
        and (wanted is None or (
            entry.get("environment") == environment
            and entry.get("fingerprint") == wanted
        ))
        and (not states or entry["state"] in states)
    ]
    return list(reversed(hits))


def latest(journal: Path, kind: str, **kwargs) -> dict | None:
    hits = query(journal, kind, **kwargs)
    return hits[0] if hits else None


def execute(args) -> int:

    def environment(pairs: list[str]) -> dict[str, str]:
        return dict(pair.split("=", 1) for pair in pairs)

    if args.command == "record":
        print(json.dumps(record(args.journal, args.kind, args.subject, args.state,
                                args.artifacts, environment(args.env)), indent=2))
        return 0
    if args.command == "query":
        hits = query(args.journal, args.kind, args.subject,
                     environment(args.env) or None, tuple(args.state))
        print(json.dumps(hits, indent=2, ensure_ascii=False))
        return 0 if hits else 1
    for entry in load(args.journal):
        print(f"{entry['state']:22} {entry['kind']:18} {entry['subject']:34} {entry['artifacts']}")
    return 0
