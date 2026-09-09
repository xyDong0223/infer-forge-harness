"""MAT-029 in-pod probe: find torch shims that silently replaced vendor kernels.

The failure this looks for has a name and a shape. During GLM-5.2's bring-up,
three functions were written to stand in for upstream triton/CUDA kernels --
kv_spans_from_batches, kunlun_convert_req_index_to_global_index,
kunlun_concat_and_cache_mla -- and none of them ever became an operator
request. The plugin served, the shims served, and the fact that three pieces of
hot-path arithmetic were running as un-optimised torch was recorded nowhere.

The probe is a net, not ground truth. It scans the installed plugin package for
signals that a function is standing in for a kernel:

- a top-level function whose name carries the ``kunlun_`` shim prefix, or
- a function whose docstring (or the comments directly above it) says it
  replaces, reimplements or shims a triton/CUDA kernel.

Every signal must be accounted for in the shim registry by the MAT-029 tool:
declared, dispatched to operator development, or waived with a reason. A signal
the registry cannot explain is exactly how the GLM-5.2 gap stayed invisible.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

# Matched case-insensitively against a function's docstring and the comment
# lines immediately above its definition. Deliberately narrow: the probe's job
# is to surface candidates for the registry, not to adjudicate them.
REPLACEMENT_MARKERS = (
    "triton",
    "cuda kernel",
    "reimplement",
    "re-implement",
    "shim",
    "垫片",
    "顶掉",
    "torch 版",
    "torch实现",
    "torch 实现",
)
SHIM_PREFIX = "kunlun_"


def _comment_text(node: ast.AST, source_lines: list[str]) -> str:
    """Comment lines immediately above the definition.

    Walks upward over comments (skipping blank lines) and stops at the first
    line of code, so a comment inside the previous function's body is never
    attributed to this one.
    """
    lines: list[str] = []
    index = node.lineno - 2
    while index >= 0:
        stripped = source_lines[index].strip()
        if stripped.startswith("#"):
            lines.append(stripped)
        elif stripped:
            break
        index -= 1
    return "\n".join(reversed(lines))


def _doc_text(node: ast.AST) -> str:
    return ast.get_docstring(node) or ""


def scan_file(path: Path, module: str) -> tuple[list[dict], bool]:
    """Signals in one file, plus whether the file parsed at all."""
    try:
        source = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return [], False
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [], False
    source_lines = source.splitlines()
    signals = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        doc = _doc_text(node)
        comments = _comment_text(node, source_lines)
        if node.name.startswith(SHIM_PREFIX):
            signals.append(
                {
                    "kind": "KUNLUN_PREFIX",
                    "file": module,
                    "symbol": node.name,
                    "line": node.lineno,
                    "excerpt": (doc or comments).strip().splitlines()[-1][:160]
                    if (doc or comments).strip()
                    else "",
                }
            )
            continue
        haystack = f"{doc}\n{comments}".lower()
        marker = next((m for m in REPLACEMENT_MARKERS if m in haystack), None)
        if marker:
            signals.append(
                {
                    "kind": "KERNEL_REPLACEMENT_DOC",
                    "file": module,
                    "symbol": node.name,
                    "line": node.lineno,
                    "excerpt": next(
                        (line.strip()[:160] for line in (doc or comments).splitlines()
                         if marker in line.lower()),
                        "",
                    ),
                }
            )
    return signals, True


def main() -> int:
    plugin = sys.argv[1] if len(sys.argv) > 1 else "vllm_kunlun"
    root = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    if root is None:
        __import__(plugin)
        root = Path(sys.modules[plugin].__file__).resolve().parent
    report = {
        "plugin": plugin,
        "plugin_path": str(root),
        "version": getattr(sys.modules.get(plugin), "__version__", None)
        if plugin in sys.modules
        else None,
        "files_scanned": 0,
        "files_unparsable": 0,
        "signals": [],
    }
    for path in sorted(root.rglob("*.py")):
        module = ".".join(path.relative_to(root).with_suffix("").parts)
        signals, parsed = scan_file(path, module)
        report["signals"].extend(signals)
        report["files_scanned"] += 1
        if not parsed:
            report["files_unparsable"] += 1
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
