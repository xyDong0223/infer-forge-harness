"""Managed output ownership shared by task command-line entry points."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Callable
from uuid import uuid4

from core.storage import ArtifactStore, WritePolicyError, ensure_external, locate_attempt


def run_managed_tool(main: Callable[[], int | None], *, task_id: str) -> int | None:
    """Guard a task CLI's explicit output and inventory even failed executions.

    This is a cooperating-entrypoint convention, not a subprocess sandbox.
    Graph runners own attempt allocation; standalone tools retain their exact
    output path for compatibility and may only claim a fresh directory.
    """
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--out")
    parser.add_argument("-h", "--help", action="store_true")
    if task_id == "mat-008-capability-evaluation":
        parser.add_argument("--list-dimensions", action="store_true")
    args, unknown = parser.parse_known_args()
    if args.help or getattr(args, "list_dimensions", False):
        return main()
    # The underlying legacy parsers accept abbreviations. Do not let one
    # bypass this guard while still selecting an output directory downstream.
    if any(token in ("--o", "--ou") or token.startswith(("--o=", "--ou="))
           for token in unknown):
        raise WritePolicyError("use the complete --out option for managed output")
    if args.out is None:
        return main()

    lexical = Path(args.out).expanduser().absolute()
    out = ensure_external(lexical)
    attempt = locate_attempt(out)
    if attempt is None:
        # A symlink below an attempt must not disguise an escaped output as
        # an unrelated standalone invocation.
        for parent in lexical.parents:
            if (parent / ".attempt.json").exists():
                attempt = locate_attempt(parent)
                break
    store = ArtifactStore(out)
    if attempt is not None:
        output = attempt.output
        if out != output and output not in out.parents:
            raise WritePolicyError(f"--out must be inside the owning attempt output: {output}")
        identity = {**attempt.identity, "tool_id": task_id}
    else:
        if out.exists() and (not out.is_dir() or any(out.iterdir())):
            raise WritePolicyError(f"--out must be a fresh or empty external directory: {out}")
        identity = {
            "run_id": f"local-{uuid4().hex}",
            "task_id": task_id,
            "attempt_id": uuid4().hex,
        }
        try:
            store.write_json(".tool-invocation.json", {"schema_version": 1, **identity})
        except FileExistsError as exc:
            raise WritePolicyError(f"--out is already owned by another invocation: {out}") from exc

    outcome = "ERROR"
    error: BaseException | None = None
    original_argv = sys.argv
    normalized_argv = original_argv.copy()
    index = 1
    while index < len(normalized_argv):
        token = normalized_argv[index]
        if token == "--":
            break
        if token == "--out":
            index += 1
            normalized_argv[index] = str(out)
        elif token.startswith("--out="):
            normalized_argv[index] = f"--out={out}"
        index += 1
    try:
        sys.argv = normalized_argv
        result = main()
        outcome = "COMPLETED" if result in (None, 0) else "FAILED"
        return result
    except BaseException as exc:
        error = exc
        raise
    finally:
        sys.argv = original_argv
        try:
            store.register(identity=identity, outcome=outcome)
        except Exception as exc:
            if error is None:
                raise
            message = f"artifact registration also failed: {exc}"
            add_note = getattr(error, "add_note", None)
            if callable(add_note):
                add_note(message)
            else:
                print(message, file=sys.stderr)
