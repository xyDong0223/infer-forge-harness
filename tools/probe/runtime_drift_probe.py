"""In-pod probe: import every plugin module against the installed engine.

Answers the question a registry scan cannot: not "is this architecture
registered" but "does the code that is registered still load here". The plugin
and the engine version it targets drift apart, and every drifted symbol is a
separate failure that only appears when something imports that module -- which,
for a backend module, means after a full model load.

Emits one JSON object on the last stdout line:

    {"engine": {...}, "plugin": {...}, "failures": [...], "resolved": {...}}

Each failure carries the module, the exception type, the message, and -- when the
message names a symbol or module -- candidate new locations found by indexing the
engine's own source. The candidates are what turn a failure list into a fix list.
"""

from __future__ import annotations

import ast
import importlib
import importlib.metadata
import json
import os
import pkgutil
import re
import sys
import traceback

# "cannot import name 'X' from 'pkg.mod'" / "No module named 'pkg.mod'"
_MISSING_NAME = re.compile(r"cannot import name '([^']+)' from '([^']+)'")
_MISSING_MODULE = re.compile(r"No module named '([^']+)'")


def _index_definitions(root: str, package: str) -> dict[str, list[str]]:
    """Map every top-level definition name in `package` to the modules defining it.

    AST rather than import: indexing by importing would run the very code whose
    imports are in question, and would cost minutes on a package this size.
    """
    index: dict[str, list[str]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            path = os.path.join(dirpath, filename)
            rel = os.path.relpath(path, root)[: -len(".py")]
            parts = [p for p in rel.split(os.sep) if p != "__init__"]
            module = ".".join([package] + parts)
            try:
                with open(path, "rb") as handle:
                    tree = ast.parse(handle.read())
            except (SyntaxError, ValueError):
                continue
            for node in tree.body:
                names: list[str] = []
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    names = [node.name]
                elif isinstance(node, ast.Assign):
                    names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    names = [node.target.id]
                for name in names:
                    index.setdefault(name, []).append(module)
    return index


def _search_order(root: str, available: list[str]) -> list[str]:
    """The package the failed import named first, the others after it.

    Order matters both ways. Searching only the named package turned a correct
    answer for `ParallelLMHead` (re-exported by the plugin, defined by the engine)
    into "gone". Searching everything at once matched a missing
    `vllm_kunlun.ops.quantization` to `vllm.config.quantization` on the tail name --
    a different package and a wrong lead. Nearest first, then widen.
    """
    return [root] + [name for name in available if name != root]


def _resolve(message: str, indexes: dict[str, dict[str, list[str]]],
             modules: dict[str, set[str]]) -> dict:
    """Suggest where a missing symbol or module lives now, or say it is gone.

    `gone_upstream` is a claim the probe is entitled to make: every indexed
    package's source was searched and nothing defines the name. It is the
    difference between "repoint this import" and "reimplement this behaviour", so
    it is stated rather than left as an empty list for the next Task to interpret.
    """
    match = _MISSING_NAME.search(message)
    if match:
        name, old_module = match.group(1), match.group(2)
        root = old_module.split(".", 1)[0]
        for package in _search_order(root, list(indexes)):
            candidates = sorted(
                m for m in indexes.get(package, {}).get(name, []) if m != old_module
            )
            if candidates:
                return {"kind": "MISSING_SYMBOL", "symbol": name, "old_module": old_module,
                        "found_in_package": package, "candidates": candidates[:8],
                        "gone_upstream": False}
        return {"kind": "MISSING_SYMBOL", "symbol": name, "old_module": old_module,
                "candidates": [], "gone_upstream": True}

    match = _MISSING_MODULE.search(message)
    if match:
        missing = match.group(1)
        root = missing.split(".", 1)[0]
        tail = missing.rsplit(".", 1)[-1]
        for package in _search_order(root, list(modules)):
            candidates = sorted(m for m in modules.get(package, set())
                                if m.rsplit(".", 1)[-1] == tail and m != missing)
            if candidates:
                return {"kind": "MISSING_MODULE", "module": missing,
                        "found_in_package": package, "candidates": candidates[:8],
                        "gone_upstream": False}
        return {"kind": "MISSING_MODULE", "module": missing, "candidates": [],
                "gone_upstream": True}

    return {"kind": "UNCLASSIFIED"}


def main(argv: list[str]) -> int:
    plugin_name = argv[1] if len(argv) > 1 else "vllm_kunlun"
    engine_name = argv[2] if len(argv) > 2 else "vllm"

    engine = importlib.import_module(engine_name)
    plugin = importlib.import_module(plugin_name)
    engine_root = os.path.dirname(engine.__file__)
    plugin_root = os.path.dirname(plugin.__file__)

    index = _index_definitions(engine_root, engine_name)
    engine_modules = {
        m.name for m in pkgutil.walk_packages([engine_root], prefix=f"{engine_name}.")
    }
    plugin_modules = {
        m.name for m in pkgutil.walk_packages([plugin_root], prefix=f"{plugin_name}.")
    }
    # Both packages are indexed: a plugin module can go missing from the plugin,
    # and looking for it in the engine only produces coincidental tail matches.
    indexes = {engine_name: index, plugin_name: _index_definitions(plugin_root, plugin_name)}
    modules = {engine_name: engine_modules, plugin_name: plugin_modules}

    failures = []
    scanned = 0
    for module in pkgutil.walk_packages([plugin_root], prefix=f"{plugin_name}."):
        scanned += 1
        try:
            importlib.import_module(module.name)
        except BaseException as error:  # noqa: BLE001 - a probe reports, it does not judge
            message = str(error)[:400]
            failures.append({
                "module": module.name,
                "error_type": type(error).__name__,
                "message": message,
                "resolution": _resolve(message, indexes, modules),
                "traceback_tail": traceback.format_exc().strip().splitlines()[-3:],
            })

    def _version(dist: str) -> str | None:
        try:
            return importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            return None

    print(json.dumps({
        "engine": {"name": engine_name, "version": getattr(engine, "__version__", None),
                   "path": engine_root},
        "plugin": {"name": plugin_name, "version": _version(plugin_name.replace("_", "-")),
                   "path": plugin_root, "modules_scanned": scanned},
        "failures": failures,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
