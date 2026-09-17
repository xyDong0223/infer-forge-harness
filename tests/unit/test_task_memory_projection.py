"""Task Memory is a disposable view, never a second source of evidence truth."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from core.paths import REPO_ROOT
from engine.state import journal, task_memory as memory


@pytest.fixture
def files(tmp_path):
    return tmp_path / "task_memory.json", tmp_path / "journal.jsonl"


def read(files, run_id="run", task_id="workflow", subject="model"):
    path, log = files
    return memory.load(path, task_id, subject, journal=log, run_id=run_id)


def save(files, view, run_id="run"):
    path, log = files
    memory.save(path, view, journal=log, run_id=run_id)


def snapshot(root):
    return {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_projection_preserves_blocks_claims_routing_and_delivery(files):
    view = read(files)
    assert not files[0].exists() and not files[1].exists()
    memory.start_block(view, "scan:1", "scan", {"state": "PASS"}, {"mode": "execution"})
    save(files, view)
    assert read(files) == view
    # The final child binding is only known after actual fan-out execution.
    view["current_loop_block"]["routing"]["child_skills"] = [{"id": "child", "sha256": "abc"}]
    memory.finish_block(view, "FAILED", ["/external/attempt/output"], {"sub_target": "triage"})
    memory.record_observed_issue(view, "command_failure", ["failure.json"],
                                 {"hardware": "simulation-cpu"}, source="scan")
    memory.record_claim(view, "measured assertion", "CONFIRMED", ["measurement.json"],
                        {"revision": "pinned"}, supersedes="old hypothesis")
    save(files, view)
    memory.start_block(view, "delivery:2", "delivery", {}, {"mode": "scheduler_delivery"})
    memory.finish_block(view, "DELIVERED", ["receipt.json"])
    save(files, view)
    assert read(files) == view
    assert read(files)["status"] == "COMPLETED"
    events = journal.load(files[1])
    assert len(events) == 3
    assert all(item["kind"] == memory.PROJECTION_KIND for item in events)
    operations = events[1]["detail"]["operations"]
    assert [item["type"] for item in operations] == [
        "block_completed", "claim_recorded", "claim_recorded", "fields_updated",
    ]
    assert "completed_loop_blocks" not in json.dumps(operations)
    assert len(journal.query(files[1], memory.PROJECTION_KIND, subject="model")) == 3


@pytest.mark.parametrize("damage", [None, "not JSON", '{"claims": ["forged"]}'])
def test_missing_or_corrupt_view_rebuilds_only_from_journal(files, damage):
    view = read(files)
    memory.record_observed_issue(view, "failed", ["evidence"], {"evidence_mode": "simulation"})
    save(files, view)
    if damage is None:
        files[0].unlink()
    else:
        files[0].write_text(damage)
    before = snapshot(files[0].parent)
    assert read(files) == view
    assert snapshot(files[0].parent) == before
    log_bytes = files[1].read_bytes()
    assert memory.rebuild(files[0], "workflow", "model", journal=files[1], run_id="run") == view
    assert json.loads(files[0].read_text()) == view
    assert files[1].read_bytes() == log_bytes


def test_legacy_seed_is_readonly_until_mutation_and_preserves_original_bytes(files):
    legacy = memory.empty_memory("workflow", "model")
    memory.record_claim(legacy, "not revalidated", "CONFIRMED", ["historic.json"],
                        {"old": "environment"}, supersedes="prior idea")
    legacy["custom_history"] = {"retain": "verbatim as old data"}
    original = json.dumps(legacy, separators=(",", ":")) + "\r\n\r\n"
    files[0].write_bytes(original.encode())
    before = snapshot(files[0].parent)
    view = read(files)
    assert snapshot(files[0].parent) == before
    memory.start_block(view, "restart:1", "triage", {}, {})
    save(files, view)
    events = journal.load(files[1])
    assert len(events) == 1  # Initialization and its first delta are one append.
    source = events[0]["detail"]["operations"][0]["source"]
    assert source["raw_text"] == original
    assert source["sha256"] == hashlib.sha256(original.encode()).hexdigest()
    assert source["revalidated"] is False
    assert read(files) == view
    save(files, view)
    assert len(journal.load(files[1])) == 1


def test_failure_between_append_and_cache_write_replays_without_duplication(files, monkeypatch):
    view = read(files)
    memory.record_observed_issue(view, "failed", ["logs"], {})
    original_write = memory._write_view

    def failed_write(*args):
        raise OSError("simulated cache write interruption")

    monkeypatch.setattr(memory, "_write_view", failed_write)
    with pytest.raises(OSError, match="interruption"):
        save(files, view)
    assert not files[0].exists()
    assert read(files) == view
    before = files[1].read_bytes()
    monkeypatch.setattr(memory, "_write_view", original_write)
    save(files, view)
    assert files[1].read_bytes() == before
    assert json.loads(files[0].read_text()) == view


def test_stale_writer_cannot_erase_or_replace_another_writer(files):
    first, stale = read(files), read(files)
    memory.start_block(first, "first", "scan", {}, {})
    save(files, first)
    memory.start_block(stale, "second", "scan", {}, {})
    before = snapshot(files[0].parent)
    with pytest.raises(ValueError, match="reload"):
        save(files, stale)
    assert snapshot(files[0].parent) == before


def test_run_identity_and_shared_journal_scope(files):
    first = read(files)
    save(files, first)
    before = snapshot(files[0].parent)
    for kwargs in ({"run_id": "other"}, {"subject": "other"}, {"task_id": "other"}):
        with pytest.raises(ValueError, match="another"):
            read(files, **kwargs)
    with pytest.raises(ValueError, match="another"):
        save(files, first, run_id="other")
    assert snapshot(files[0].parent) == before
    second = (files[0].parent / "another-view.json", files[1])
    view = read(second, run_id="other")
    memory.record_claim(view, "other", "OBSERVED", [], {})
    save(second, view, run_id="other")
    assert read(second, run_id="other") == view
    assert not read(files)["claims"]


def test_history_rewrite_and_derived_record_forgery_are_rejected(files):
    view = read(files)
    memory.start_block(view, "one", "scan", {})
    memory.finish_block(view, "FAILED", ["evidence"])
    save(files, view)
    before = snapshot(files[0].parent)
    for mutate in (lambda value: value["completed_loop_blocks"].clear(),
                   lambda value: value["execution_records"].append({"state": "PASS"})):
        changed = read(files)
        mutate(changed)
        with pytest.raises(ValueError):
            save(files, changed)
    assert snapshot(files[0].parent) == before


def test_bad_legacy_or_journal_is_not_silently_discarded(files):
    files[0].write_text("[]")
    with pytest.raises(ValueError, match="object"):
        read(files)
    assert not files[1].exists()
    files[0].unlink()
    save(files, read(files))
    entries = journal.load(files[1])
    entries[0]["detail"]["operations"] = [{"type": "forged"}]
    files[1].write_text(json.dumps(entries[0]) + "\n")
    with pytest.raises(ValueError, match="digest"):
        read(files)


def test_partial_journal_append_stays_fail_closed(files):
    view = read(files)
    save(files, view)
    with files[1].open("a") as handle:
        handle.write('{"kind":"TaskMemoryProjection"')
    before = snapshot(files[0].parent)
    with pytest.raises(ValueError):
        read(files)
    memory.record_claim(view, "retry must not append", "OBSERVED", [], {})
    with pytest.raises(ValueError):
        save(files, view)
    assert snapshot(files[0].parent) == before


def test_rebuild_requires_authority_and_does_not_seed_or_create_paths(files):
    before = snapshot(files[0].parent)
    with pytest.raises(ValueError, match="no authoritative"):
        memory.rebuild(files[0], "workflow", "model", journal=files[1], run_id="run")
    assert snapshot(files[0].parent) == before
    with pytest.raises(ValueError, match="run_id"):
        memory.load(files[0], "workflow", "model", journal=files[1])


def test_legacy_api_and_duplicate_event_replay(files):
    view = memory.load(files[0], "workflow", "model")
    memory.save(files[0], view)
    assert memory.load(files[0], "workflow", "model") == view
    save(files, read(files))
    item = journal.load(files[1])[0]
    journal.record(files[1], item["kind"], item["subject"], item["state"],
                   Path(item["artifacts"]), item["environment"], extra=item["detail"])
    assert read(files) == view


def test_concurrent_journal_writers_preserve_whole_records(files):
    def write(index):
        for iteration in range(10):
            journal.record(files[1], "Fact", "model", "OBSERVED", files[0].parent,
                           {}, extra={"writer": index, "iteration": iteration})

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(write, range(4)))
    assert len(journal.load(files[1])) == 40
    assert len({(entry["detail"]["writer"], entry["detail"]["iteration"])
                for entry in journal.load(files[1])}) == 40


def test_cli_show_is_readonly_and_rebuild_is_explicit(files):
    view = read(files)
    save(files, view)
    files[0].unlink()
    command = [sys.executable, str(REPO_ROOT / "cli/state/task_memory.py"),
               "--path", str(files[0]), "--task-id", "workflow", "--subject", "model",
               "--journal", str(files[1]), "--run-id", "run"]
    before = snapshot(files[0].parent)
    shown = subprocess.run([*command, "--show"], capture_output=True, text=True)
    assert shown.returncode == 0, shown.stderr
    assert json.loads(shown.stdout) == view
    assert snapshot(files[0].parent) == before
    built = subprocess.run([*command, "--rebuild"], capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    assert json.loads(files[0].read_text()) == view
    rejected = subprocess.run([*command[:-1], "wrong", "--rebuild"],
                              capture_output=True, text=True)
    assert rejected.returncode == 2
