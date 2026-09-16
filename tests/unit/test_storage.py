import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from core.storage import (
    ArtifactStore,
    RunPaths,
    WritePolicyError,
    default_state_root,
    ensure_external,
    locate_attempt,
    safe_component,
)
from engine.scheduler import EventStore
from engine.state import journal, task_memory


def test_runtime_roots_are_external_and_resolved(tmp_path):
    repo = tmp_path / "source"
    repo.mkdir()
    for path in (repo, repo / "artifacts", repo.parent, Path.home(), Path("/")):
        with pytest.raises(WritePolicyError):
            ensure_external(path, repo_root=repo)
    alias = tmp_path / "alias"
    alias.symlink_to(repo, target_is_directory=True)
    with pytest.raises(WritePolicyError):
        ensure_external(alias / "output", repo_root=repo)
    assert ensure_external(tmp_path / "runtime", repo_root=repo) == tmp_path / "runtime"


def test_external_state_defaults(monkeypatch, tmp_path):
    monkeypatch.delenv("INFER_FORGE_STATE_ROOT", raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    assert default_state_root() == tmp_path / "infer-forge"
    monkeypatch.setenv("INFER_FORGE_STATE_ROOT", str(tmp_path / "custom"))
    assert default_state_root() == tmp_path / "custom"
    assert not (tmp_path / "custom").exists()


def test_safe_identifiers_cannot_traverse_or_collide():
    assert safe_component("model-v1") == "model-v1"
    assert safe_component("a/b") != safe_component("a:b")
    for value in ("../escape", ".", "..", "/absolute", "a" * 500):
        component = safe_component(value)
        assert "/" not in component and component not in (".", "..")
        assert len(component) < 120
    with pytest.raises(WritePolicyError):
        safe_component("")


def test_run_identity_is_stable_and_constructor_does_not_write(tmp_path):
    root = tmp_path / "run"
    paths = RunPaths(root, "run-1")
    assert not root.exists()
    paths.initialize()
    assert RunPaths(root).initialize().run_id == "run-1"
    with pytest.raises(WritePolicyError):
        RunPaths(root, "run-2").initialize()
    assert json.loads((root / "run.json").read_text())["run_id"] == "run-1"


def test_attempt_identity_and_directory_aliases_are_rejected(tmp_path):
    paths = RunPaths(tmp_path / "run", "run")
    attempt = paths.allocate_attempt("one")
    alias = paths.root / "tasks" / "two"
    alias.symlink_to(attempt.root.parents[1], target_is_directory=True)
    with pytest.raises(WritePolicyError, match="symlink"):
        paths.allocate_attempt("two")
    marker = attempt.root / ".attempt.json"
    payload = json.loads(marker.read_text())
    payload["run_id"] = "another-run"
    marker.write_text(json.dumps(payload))
    with pytest.raises(WritePolicyError, match="conflicting run"):
        locate_attempt(attempt.output)
    marker.unlink()
    marker.symlink_to(paths.root / "run.json")
    with pytest.raises(WritePolicyError, match="symlink"):
        locate_attempt(attempt.output)


def test_concurrent_initialization_and_attempts(tmp_path):
    root = tmp_path / "run"
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: RunPaths(root).initialize().run_id, range(16)))
    assert len(set(results)) == 1
    with ThreadPoolExecutor(max_workers=8) as pool:
        attempts = list(pool.map(lambda _: RunPaths(root).allocate_attempt("torch/op"), range(16)))
    assert len({attempt.root for attempt in attempts}) == 16
    assert sorted(a.identity["attempt_id"] for a in attempts) == [
        f"{number:06d}" for number in range(1, 17)
    ]
    assert all(locate_attempt(a.output / "result.json") == a for a in attempts)


def test_retry_preserves_previous_attempt(tmp_path):
    paths = RunPaths(tmp_path / "run", "r")
    first = paths.allocate_attempt("node")
    ArtifactStore(first.output).write_json("status.json", {"state": "FAILED"})
    original = (first.output / "status.json").read_bytes()
    second = paths.allocate_attempt("node")
    ArtifactStore(second.output).write_json("status.json", {"state": "READY"})
    assert first.root != second.root
    assert (first.output / "status.json").read_bytes() == original


def test_guarded_paths_and_exclusive_atomic_writes(tmp_path):
    store = ArtifactStore(tmp_path / "output")
    for name in ("../escape.json", "/absolute.json", "."):
        with pytest.raises(WritePolicyError):
            store.path(name)
    store.write_text("result.txt", "first")
    with pytest.raises(FileExistsError):
        store.write_text("result.txt", "second")
    assert store.path("result.txt").read_text() == "first"
    store.write_text("result.txt", "final", overwrite=True)
    assert store.path("result.txt").read_text() == "final"
    destination = tmp_path / "outside"
    destination.mkdir()
    (store.root / "linked").symlink_to(destination, target_is_directory=True)
    with pytest.raises(WritePolicyError):
        store.write_text("linked/escaped.txt", "bad")
    assert not (destination / "escaped.txt").exists()
    (store.root / "result-alias.txt").symlink_to(store.root / "result.txt")
    with pytest.raises(WritePolicyError, match="symlink"):
        store.write_text("result-alias.txt", "overwrite", overwrite=True)
    assert store.path("result.txt").read_text() == "final"


def test_manifest_hashes_formal_artifacts_not_scratch(tmp_path):
    attempt = RunPaths(tmp_path / "run", "run").allocate_attempt("task")
    store = ArtifactStore(attempt.root)
    store.write_text("output/result.txt", "evidence")
    store.write_text("logs/task.log", "log")
    store.write_text("scratch/exploration.txt", "temporary")
    path = store.register(identity=attempt.identity, outcome="FAILED",
                          required=["output/result.txt"])
    manifest = json.loads(path.read_text())
    assert manifest["identity"] == attempt.identity
    assert manifest["outcome"] == "FAILED"
    assert {item["path"] for item in manifest["artifacts"]} == {
        "output/result.txt", "logs/task.log",
    }
    result = next(item for item in manifest["artifacts"] if item["path"].startswith("output"))
    assert result["sha256"] == hashlib.sha256(b"evidence").hexdigest()
    assert result["size_bytes"] == len(b"evidence")
    original = path.read_bytes()
    with pytest.raises(WritePolicyError, match="missing"):
        store.register(identity=attempt.identity, outcome="PASS", required=["output/missing"])
    assert path.read_bytes() == original


def test_manifest_rejects_symlinks_and_nonregular_files(tmp_path):
    store = ArtifactStore(tmp_path / "output")
    result = store.write_text("result.txt", "ok")
    (store.root / "linked.txt").symlink_to(result)
    with pytest.raises(WritePolicyError, match="symlink"):
        store.register(identity={}, outcome="FAILED")
    (store.root / "linked.txt").unlink()
    import os
    os.mkfifo(store.root / "pipe")
    with pytest.raises(WritePolicyError, match="regular"):
        store.register(identity={}, outcome="FAILED")


def test_single_file_manifest_does_not_inventory_unrelated_inputs(tmp_path):
    store = ArtifactStore(tmp_path / "output")
    store.write_text("result.json", "{}")
    store.write_text("input.json", "not an output")
    path = store.register(identity={}, outcome="COMPLETED", include=["result.json"],
                          required=["result.json"], manifest_name="result.json.manifest.json")
    manifest = json.loads(path.read_text())
    assert [item["path"] for item in manifest["artifacts"]] == ["result.json"]


def test_legacy_state_writers_reject_source_tree():
    repo = Path(__file__).resolve().parents[2]
    with pytest.raises(WritePolicyError):
        journal.record(repo / "artifacts/journal.jsonl", "test", "subject",
                       "FAILED", repo, {})
    with pytest.raises(WritePolicyError):
        task_memory.save(repo / "artifacts/task_memory.json", {})
    with pytest.raises(WritePolicyError):
        EventStore(repo / "artifacts/state.sqlite")
    store = EventStore(":memory:")
    store.close()
