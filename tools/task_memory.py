"""Durable Task Loop memory for agent-driven execution.

The Task remains the stable goal. Each graph node or manual investigation is a
Loop Block with a local target, exit condition, and execution record. This
module intentionally stores only coordination state; evidence stays in the
artifact directories and the Journal remains the source of fact provenance.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


def empty_memory(task_id: str, subject: str) -> dict[str, Any]:
    return {
        "version": 1,
        "task_id": task_id,
        "subject": subject,
        "status": "IN_PROGRESS",
        "current_loop_block": None,
        "completed_loop_blocks": [],
        "next_loop_block": None,
        "execution_records": [],
    }


def load(path: Path, task_id: str, subject: str) -> dict[str, Any]:
    if not path.exists():
        return empty_memory(task_id, subject)
    memory = json.loads(path.read_text(encoding="utf-8"))
    if memory.get("task_id") != task_id or memory.get("subject") != subject:
        raise ValueError(f"task memory belongs to another task: {path}")
    memory.setdefault("completed_loop_blocks", [])
    memory.setdefault("execution_records", [])
    memory.setdefault("next_loop_block", None)
    memory.setdefault("current_loop_block", None)
    return memory


def save(path: Path, memory: dict[str, Any]) -> None:
    """Atomically replace memory so an interrupted Agent cannot leave bad JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(memory, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def start_block(
    memory: dict[str, Any],
    block_id: str,
    sub_target: str,
    exit_condition: dict[str, Any],
    routing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    block = {
        "block_id": block_id,
        "sub_target": sub_target,
        "exit_condition": exit_condition,
        "routing": routing or {},
        "started_at": time.time(),
    }
    memory["current_loop_block"] = block
    memory["next_loop_block"] = None
    return block


def finish_block(
    memory: dict[str, Any],
    state: str,
    artifacts: list[str] | None = None,
    next_block: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current = memory.get("current_loop_block")
    if not current:
        raise ValueError("cannot finish a Task Loop without a current block")
    record = {
        **current,
        "state": state,
        "artifacts": artifacts or [],
        "finished_at": time.time(),
    }
    memory["completed_loop_blocks"].append(record)
    memory["execution_records"].append(
        {
            "block_id": record["block_id"],
            "state": state,
            "artifacts": record["artifacts"],
        }
    )
    memory["current_loop_block"] = None
    memory["next_loop_block"] = next_block
    if next_block is None and state in {"DELIVERED", "TASK_COMPLETE"}:
        memory["status"] = "COMPLETED"
    return record


def set_next_block(memory: dict[str, Any], next_block: dict[str, Any]) -> None:
    memory["next_loop_block"] = next_block


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    memory = load(args.path, args.task_id, args.subject)
    if args.show:
        print(json.dumps(memory, indent=2, ensure_ascii=False))
        return 0
    save(args.path, memory)
    print(json.dumps(memory, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
