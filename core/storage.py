"""External run paths and artifact inventories for managed harness entry points.

These checks govern cooperating tools; they do not sandbox arbitrary subprocesses.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable
from uuid import uuid4


from core.paths import REPO_ROOT


class WritePolicyError(ValueError):
    """A requested runtime write crosses an ownership boundary."""


def ensure_external(path: str | Path, repo_root: str | Path | None = None) -> Path:
    candidate = Path(path).expanduser().resolve()
    source = Path(repo_root or REPO_ROOT).resolve()
    if candidate == source or source in candidate.parents or candidate in source.parents:
        raise WritePolicyError(f"runtime path overlaps the source repository: {candidate}")
    if candidate == Path.home().resolve() or candidate == Path(candidate.anchor):
        raise WritePolicyError(f"runtime path must be a dedicated directory: {candidate}")
    return candidate


def default_state_root() -> Path:
    configured = os.environ.get("INFER_FORGE_STATE_ROOT")
    if configured:
        return ensure_external(configured)
    base = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")
    return ensure_external(base / "infer-forge")


def safe_component(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WritePolicyError("run and task identities must be nonempty strings")
    if re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}", value):
        return value
    stem = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip(".-")[:64] or "id"
    return f"{stem}-{hashlib.sha256(value.encode()).hexdigest()[:12]}"


def _atomic_json(path: Path, payload: Any) -> None:
    content = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    _atomic_text(path, content, overwrite=True)


def _atomic_text(path: Path, content: str, *, overwrite: bool) -> None:
    pending = path.with_name(f".{path.name}.{uuid4().hex}.pending")
    try:
        with pending.open("x", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(pending, path)
        else:
            os.link(pending, path)
    finally:
        pending.unlink(missing_ok=True)


@dataclass(frozen=True)
class AttemptPaths:
    root: Path
    identity: dict[str, str]

    @property
    def input(self) -> Path:
        return self.root / "input"

    @property
    def scratch(self) -> Path:
        return self.root / "scratch"

    @property
    def output(self) -> Path:
        return self.root / "output"

    @property
    def logs(self) -> Path:
        return self.root / "logs"


class RunPaths:
    def __init__(self, root: str | Path, run_id: str | None = None):
        self.root = ensure_external(root)
        self.run_id = run_id
        if run_id is not None:
            safe_component(run_id)

    @classmethod
    def for_run(cls, run_id: str) -> "RunPaths":
        return cls(default_state_root() / "runs" / safe_component(run_id), run_id)

    @property
    def journal(self) -> Path:
        return self.root / "journal.jsonl"

    @property
    def memory(self) -> Path:
        return self.root / "task_memory.json"

    @property
    def state(self) -> Path:
        return self.root / "state.sqlite"

    def initialize(self) -> "RunPaths":
        ensure_external(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        marker = self.root / "run.json"
        if marker.is_symlink():
            raise WritePolicyError(f"run identity cannot be a symlink: {marker}")
        if marker.exists():
            try:
                identity = json.loads(marker.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                raise WritePolicyError(f"cannot read run identity: {marker}") from exc
            existing = identity.get("run_id") if isinstance(identity, dict) else None
            if not isinstance(identity, dict) or identity.get("schema_version") != 1:
                raise WritePolicyError(f"unsupported run identity: {marker}")
            if not isinstance(existing, str) or not existing.strip() or (
                self.run_id is not None and existing != self.run_id
            ):
                raise WritePolicyError(f"run identity conflicts with {marker}")
            self.run_id = existing
        else:
            run_id = self.run_id or uuid4().hex
            pending = marker.with_name(f".run.{uuid4().hex}.pending")
            # Publish a complete file exclusively; another launcher cannot read
            # partially written JSON or replace the accepted run identity.
            try:
                with pending.open("x", encoding="utf-8") as handle:
                    json.dump({"schema_version": 1, "run_id": run_id}, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.link(pending, marker)
                self.run_id = run_id
            except FileExistsError:
                return self.initialize()
            finally:
                pending.unlink(missing_ok=True)
        return self

    def allocate_attempt(self, task_id: str) -> AttemptPaths:
        self.initialize()
        assert self.run_id is not None
        parent = self.root / "tasks" / safe_component(task_id) / "attempts"
        ensure_external(parent)
        if any(path.is_symlink() for path in (parent, parent.parent, parent.parent.parent)):
            raise WritePolicyError(f"task directories cannot be symlinks: {parent}")
        if self.root not in parent.resolve().parents:
            raise WritePolicyError("task path escapes its run")
        parent.mkdir(parents=True, exist_ok=True)
        number = 1
        while True:
            root = parent / f"{number:06d}"
            try:
                root.mkdir()
                break
            except FileExistsError:
                number += 1
        identity = {"run_id": self.run_id, "task_id": task_id, "attempt_id": root.name}
        attempt = AttemptPaths(root, identity)
        for directory in (attempt.input, attempt.scratch, attempt.output, attempt.logs):
            directory.mkdir()
        _atomic_json(root / ".attempt.json", {"schema_version": 1, **identity})
        return attempt


def locate_attempt(path: str | Path) -> AttemptPaths | None:
    current = ensure_external(path)
    for directory in (current, *current.parents):
        marker = directory / ".attempt.json"
        if marker.is_symlink():
            raise WritePolicyError(f"attempt identity cannot be a symlink: {marker}")
        if not marker.exists():
            continue
        if not marker.is_file():
            raise WritePolicyError(f"invalid attempt marker: {marker}")
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise WritePolicyError(f"invalid attempt marker: {marker}") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != 1 or not all(
            isinstance(payload.get(key), str) and payload[key]
            for key in ("run_id", "task_id", "attempt_id")
        ):
            raise WritePolicyError(f"invalid attempt identity: {marker}")
        if (len(directory.parents) < 4 or directory.name != payload["attempt_id"]
                or directory.parent.name != "attempts"
                or directory.parents[1].name != safe_component(payload["task_id"])
                or directory.parents[2].name != "tasks"):
            raise WritePolicyError(f"attempt identity does not match its directory: {marker}")
        run_marker = directory.parents[3] / "run.json"
        if run_marker.is_symlink():
            raise WritePolicyError(f"run identity cannot be a symlink: {run_marker}")
        try:
            run_identity = json.loads(run_marker.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise WritePolicyError(f"cannot read owning run identity: {run_marker}") from exc
        if (not isinstance(run_identity, dict) or run_identity.get("schema_version") != 1
                or run_identity.get("run_id") != payload["run_id"]):
            raise WritePolicyError(f"attempt has a conflicting run identity: {marker}")
        return AttemptPaths(ensure_external(directory), {
            key: payload[key] for key in ("run_id", "task_id", "attempt_id")
        })
    return None


class ArtifactStore:
    def __init__(self, root: str | Path):
        self.root = ensure_external(root)

    def path(self, relative: str | Path) -> Path:
        relative = Path(relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise WritePolicyError(f"artifact must be relative to its owner: {relative}")
        lexical = self.root / relative
        for component in (lexical, *lexical.parents):
            if component == self.root:
                break
            if component.is_symlink():
                raise WritePolicyError(f"artifact paths cannot contain symlinks: {component}")
        path = ensure_external(lexical)
        if self.root not in path.parents:
            raise WritePolicyError(f"artifact escapes its owner: {relative}")
        return path

    def write_text(self, relative: str | Path, content: str, *, overwrite: bool = False) -> Path:
        path = self.path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_text(path, content, overwrite=overwrite)
        return path

    def write_json(self, relative: str | Path, payload: Any, *, overwrite: bool = False) -> Path:
        content = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
        return self.write_text(relative, content, overwrite=overwrite)

    def register(
        self, *, identity: dict[str, Any], outcome: str, required: Iterable[str] = (),
        include: Iterable[str] | None = None, manifest_name: str = "manifest.json",
    ) -> Path:
        """Inventory formal files without promoting their correctness verdict."""
        ensure_external(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        manifest = self.path(manifest_name)
        artifacts = []
        selected = None if include is None else {
            self.path(item).relative_to(self.root).as_posix() for item in include
        }

        def unreadable(error: OSError) -> None:
            raise error

        for directory, dirs, files in os.walk(self.root, followlinks=False, onerror=unreadable):
            base = Path(directory)
            if "scratch" in base.relative_to(self.root).parts:
                dirs[:] = []
                continue
            if selected is not None:
                dirs[:] = [
                    name for name in dirs
                    if any(
                        (base / name).relative_to(self.root) in Path(item).parents
                        for item in selected
                    )
                ]
            for name in dirs:
                if (base / name).is_symlink():
                    raise WritePolicyError(f"artifact directories cannot be symlinks: {base / name}")
            dirs[:] = sorted(name for name in dirs if name != "scratch")
            for name in sorted(files):
                file = base / name
                if file == manifest or name == ".attempt.json":
                    continue
                if selected is not None and file.relative_to(self.root).as_posix() not in selected:
                    continue
                if file.is_symlink():
                    raise WritePolicyError(f"artifact files cannot be symlinks: {file}")
                if not file.is_file():
                    raise WritePolicyError(f"artifact must be a regular file: {file}")
                relative = file.relative_to(self.root)
                self.path(relative)
                digest = hashlib.sha256()
                with file.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                artifacts.append({
                    "path": relative.as_posix(), "size_bytes": file.stat().st_size,
                    "sha256": digest.hexdigest(),
                    "kind": "log" if file.suffix == ".log" else "json" if file.suffix == ".json" else "file",
                })
        available = {entry["path"] for entry in artifacts}
        missing = sorted((set(required) | (selected or set())) - available)
        if missing:
            raise WritePolicyError(f"missing required artifacts: {', '.join(missing)}")
        _atomic_json(manifest, {
            "schema_version": 1, "identity": identity, "outcome": outcome,
            "artifacts": artifacts,
        })
        return manifest
