"""In-pod precheck: engine-core-init drift, seconds instead of a reload.

MAT-027 proves every plugin module *imports*. That gate passed while twelve
call-time drifts still waited behind the 707 GiB weight load — each one a full
server restart to discover (run glm52-int-w8a8-p800-001, 2026-09-14: one
~15-minute load per drift, twelve times). This probe replays the plugin's
engine-facing code surface against the installed engine *without any weights,
server or XPU state*:

1. resolve — every engine-rooted symbol the plugin's source references
   (attribute chains, whether called or not) must exist in the installed
   engine;
2. bind — every engine-rooted *call* is dry-run against the engine function's
   signature with placeholder arguments: an arity or keyword mismatch is
   exactly the crash EngineCore would raise, seen now instead of after load;
3. curated — the known init-path landmines from the 0.25.1 drift map, the
   plugin-state the drift repairs produce (prefill stub, flashmla_sparse
   port, VllmModel protocol methods, is_cuda_alike), and the vendor ops on
   the request path.

Scoping: with ``--model-config <path>/config.json`` the gate is limited to
the plugin files this deployment actually loads — the model module the
engine's own ModelRegistry resolves for the config's architectures, plus
the init surface below. Drift in other models' files is reported as
``drift_out_of_path`` — real, but not this deployment's blocker.

Gating: the generic sweep cannot prove a reference *executes* — the healthy
glm52 pod carries 23 genuinely-missing symbols in branches that never run on
XPU (HPU/NEURON platform arms, flashinfer/cudnn prefill options, CP-attention
merge ops). Every one of the run's twelve blocking drifts was either a
call-site failure or is covered by a curated check; the bare attribute reads
that remain are conditional-branch arms. So the sweep hard-gates call-site
failures on the init surface, and reports everything else:

    models/<focus>           the module the registry resolves for the config
    __init__.py, platforms/  plugin bootstrap and platform selection
    v1/attention/backends/mla/{indexer,flashmla_sparse,prefill_xpu}.py
                             the MLA/DSA attention init surface

Every other finding keeps its DRIFT verdict but is ``gating: report-only``
— surfaced for triage, never a launch blocker. Bare attribute reads are
report-only even inside the gate prefixes (load-bearing ones — envs flags
the engine removed — are curated explicitly). ``--gate-prefixes`` overrides
the prefix list.

Ordering: engine modules are warmed before the plugin import, mirroring the
server's engine-before-plugin order. Registering the plugin's custom ops
first makes later engine module imports raise spurious "Duplicate op name"
errors; such import-raised resolutions are WARN (cannot verify here), never
DRIFT. A reference guarded by ``except AttributeError`` is WARN for the same
reason: the plugin is allowed to probe for an optional symbol.

Verdicts: PASS / DRIFT (gate fails) / WARN (recorded, gate passes) /
UNVERIFIABLE (signature opaque or variadic — honest, not silently green).
PASS entries are deduplicated per symbol; every non-PASS site is reported.

Emits one JSON object on the last stdout line:

    {"state": "PASS"|"DRIFT", "engine": {...}, "plugin": {...},
     "scope": {...}, "checks": [...], "summary": {...}}

Not covered (and honestly not coverable statically): return-shape changes
(FusedMoE's combine list->tensor), subscript protocol changes (kv_cache[0]),
and behaviour that only diverges with real tensors. Those still belong to the
toy bring-up and the correctness gates.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.metadata
import inspect
import json
import subprocess
import sys
from pathlib import Path

_FILLER = None  # bind() inspects shape, never values
_MAX_REPORTED = 1500

# Engine modules imported before the plugin, mirroring the server's order
# (see the module docstring's Ordering paragraph).
WARMUP_ENGINE_MODULES = [
    "vllm.model_executor.layers.logits_processor",
    "vllm.model_executor.layers.fused_moe",
    "vllm.model_executor.layers.fused_moe.layer",
    "vllm.model_executor.layers.rotary_embedding",
    "vllm.model_executor.layers.mla",
    "vllm.v1.attention.backends.mla.indexer",
    "vllm.v1.attention.backends.mla.prefill.registry",
    "vllm.v1.attention.ops.flashmla",
    "vllm.utils",
]

# Known init-path landmines from openwiki/harness/vllm-0251-drift-map.md.
# Each is a symbol resolve; the "why" names the run failure that proved the
# check earns its place.
CURATED_SYMBOLS = [
    ("vllm.model_executor.layers.mla.MultiHeadLatentAttentionWrapper",
     "drift 1: MHA class removed, Wrapper is the same constructor"),
    ("vllm.v1.attention.backends.mla.indexer.split_indexer_prefill_chunks",
     "drift 3: split_prefill_chunks was renamed and re-parameterised"),
    ("vllm.v1.attention.ops.flashmla.is_flashmla_dense_supported",
     "drift 6/13: flashmla moved and is_flashmla_supported was split"),
    ("vllm.v1.attention.backends.mla.prefill.registry.register_mla_prefill_backend",
     "drift 8: the MLA prefill registry the plugin's stub registers into"),
    ("vllm.v1.attention.backends.mla.prefill.registry.MLAPrefillBackendEnum",
     "drift 8: enum side of the same registry"),
    ("torch.ops._C.concat_and_cache_mla",
     "request path: the MLA KV-cache update the ported backend calls"),
]

# Optional symbols: absent means WARN + operator debt, never a blocked gate.
CURATED_OPTIONAL = [
    ("torch.ops.xspeedgate_ops.kv_spans_from_batches",
     "vendor op absent in pinned xspeedgate_ops 1.5.1: torch fallback is "
     "exact-equal, upgrade remains operator debt"),
    ("vllm.envs.VLLM_ATTENTION_BACKEND",
     "drift 5: removed from vllm.envs; a plugin still reading it crashes at "
     "VllmConfig creation — the patched plugin reads os.getenv instead"),
]

# Plugin-state the drift repairs produce; absent = repairs not applied here.
CURATED_PLUGIN_STATE = [
    ("vllm_kunlun.v1.attention.backends.mla.prefill_xpu.XPUMLAPrefillStub",
     "drift 8: XPU MLA prefill stub registered at plugin activation"),
    ("vllm_kunlun.v1.attention.backends.mla.flashmla_sparse",
     "the ported sparse MLA backend (new SparseMLAAttentionImpl API)"),
]

# The unconditional engine-core-init surface (see the module docstring's
# Gating paragraph): where run glm52-int-w8a8-p800-001's twelve blocking
# drifts actually executed. Findings outside these prefixes are real but
# conditional-branch risk: reported, never the launch blocker.
GATE_PREFIXES = (
    "__init__.py",
    "models/",
    "platforms/",
    "v1/attention/backends/mla/indexer.py",
    "v1/attention/backends/mla/flashmla_sparse.py",
    "v1/attention/backends/mla/prefill_xpu.py",
)


def resolve(path: str, cache: dict | None = None) -> tuple[object | None, str, str]:
    """Import the longest module prefix of ``path`` and getattr the rest.

    Returns ``(obj, detail, kind)`` with kind one of:

    * ``resolved``      — the symbol exists, ready to bind;
    * ``missing``       — no prefix imports and the tail attr exists nowhere:
                          the drift this probe exists to find;
    * ``import-raised`` — a module import crashed for a non-missing reason
                          (op registration order, half-initialised state).
                          Honest uncertainty, not proof of drift.
    """
    cache = cache if cache is not None else {}
    if path in cache:
        return cache[path]
    parts = path.split(".")
    result: tuple[object | None, str, str] | None = None

    def keep(candidate: tuple[object | None, str, str]) -> None:
        nonlocal result
        if result is None or (result[2] != "import-raised"
                              and candidate[2] == "import-raised"):
            # An import that RAISED (op registration, half-initialised
            # state) is strictly more informative than a later plain
            # "missing" from a shorter prefix: keep the stronger finding.
            result = candidate

    for split in range(len(parts), 0, -1):
        module_name = ".".join(parts[:split])
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue  # try a shorter importable prefix
        except ImportError as error:
            # The module path exists but its import failed — symbol-level
            # breakage inside that chain is exactly "missing".
            keep((None, f"import {module_name} failed: {error}", "missing"))
            continue
        except Exception as error:  # noqa: BLE001 - the failure IS the answer
            keep((None, f"import {module_name} raised {type(error).__name__}: "
                        f"{error}", "import-raised"))
            continue
        obj: object = module
        detail, kind = "", "resolved"
        for attr in parts[split:]:
            try:
                obj = getattr(obj, attr)
            except AttributeError:
                obj, detail, kind = None, (
                    f"{module_name} has no attribute {'.'.join(parts[split:])}"
                ), "missing"
                break
        if kind == "resolved":
            cache[path] = (obj, detail, kind)
            return cache[path]
    if result is None:
        result = (None, f"no importable prefix resolves {path}", "missing")
    cache[path] = result
    return result


def bind_call(func, call: ast.Call) -> tuple[str, str]:
    """Dry-run a call node against the engine function's real signature."""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return "UNVERIFIABLE", "engine callable exposes no introspectable signature"
    if any(
        p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        for p in signature.parameters.values()
    ):
        return "UNVERIFIABLE", f"engine signature is variadic: {signature}"
    positional, keywords = [], {}
    for arg in call.args:
        if isinstance(arg, ast.Starred):
            return "UNVERIFIABLE", "call site unpacks *args"
        positional.append(_FILLER)
    for kw in call.keywords:
        if kw.arg is None:
            return "UNVERIFIABLE", "call site unpacks **kwargs"
        keywords[kw.arg] = _FILLER
    try:
        signature.bind(*positional, **keywords)
    except TypeError as error:
        return "DRIFT", str(error)
    return "PASS", f"binds: {signature}"


def _attr_chain(node: ast.AST) -> tuple[str, list[str]] | None:
    """``a.b.c`` -> ("a", ["b", "c"]); plain ``a`` -> ("a", [])."""
    attrs: list[str] = []
    while isinstance(node, ast.Attribute):
        attrs.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return node.id, list(reversed(attrs))
    return None


class Scanner:
    """Walk one plugin file's AST for engine-rooted references and calls.

    An attribute chain is reported once at its head and not descended into:
    ``a.b.c`` reports ``a.b.c``, never the shorter ``a.b``. A chain rooted at
    a call or subscript (``f().g``, ``x[i].b``) descends so nested calls are
    still found. Store targets are skipped: assigning ``engine.attr = x`` is
    monkeypatching, not a reference. Only the engine package is swept —
    torch's C internals are a monkeypatching surface for the plugin and a
    false-positive factory; the torch ops that matter are curated.
    """

    def __init__(self, roots: tuple[str, ...], aliases: dict[str, str]) -> None:
        self.roots = roots
        self.aliases = aliases
        self.filename = "?"
        self.checks: list[dict] = []
        self.stats = {"references": 0, "pass": 0, "drift": 0,
                      "warn": 0, "unverifiable": 0, "files": 0, "truncated": False}
        self._pass_entries: dict[str, dict] = {}
        self._try_stack: list[ast.Try] = []

    # -- verdict plumbing ---------------------------------------------------
    def _guarded(self) -> bool:
        for node in self._try_stack:
            for handler in node.handlers:
                if handler.type is None:
                    return True
                names = []
                if isinstance(handler.type, ast.Name):
                    names = [handler.type.id]
                elif isinstance(handler.type, ast.Tuple):
                    names = [e.id for e in handler.type.elts
                             if isinstance(e, ast.Name)]
                if "AttributeError" in names:
                    return True
        return False

    def _rooted(self, root: str, attrs: list[str]) -> str | None:
        base = self.aliases.get(root)
        if base is None:
            return None
        path = ".".join([base, *attrs])
        if any(path == r or path.startswith(r + ".") for r in self.roots):
            return path
        return None

    def report(self, path: str, line: int, call: ast.Call | None) -> None:
        self.stats["references"] += 1
        obj, error, kind = resolve(path)
        guarded = self._guarded()
        if obj is None:
            if kind == "import-raised" or guarded:
                verdict = "WARN"
            else:
                verdict = "DRIFT"
            detail = error
        elif call is None:
            verdict, detail = "PASS", "symbol resolves"
        else:
            verdict, detail = bind_call(obj, call)
            if verdict == "DRIFT" and guarded:
                verdict = "WARN"
        self.stats[verdict.lower()] += 1
        if verdict == "PASS":
            prior = self._pass_entries.get(path)
            if prior is not None:
                prior["sites"] += 1
                return
        if len(self.checks) >= _MAX_REPORTED:
            self.stats["truncated"] = True
            return
        entry = {"kind": "bind" if call is not None else "resolve",
                 "id": f"{self.filename}:{line} {path}",
                 "verdict": verdict, "detail": detail,
                 "symbol": path, "sites": 1,
                 "resolution": kind if obj is None else "resolved"}
        if verdict == "PASS":
            self._pass_entries[path] = entry
        self.checks.append(entry)

    # -- traversal ----------------------------------------------------------
    def scan(self, node: ast.AST) -> None:
        if isinstance(node, ast.Attribute):
            if isinstance(node.ctx, ast.Store):
                return
            chain = _attr_chain(node)
            if chain:
                root, attrs = chain
                path = self._rooted(root, attrs)
                if path:
                    self.report(path, node.lineno, None)
                return  # the chain is reported as one reference
            for child in ast.iter_child_nodes(node):
                self.scan(child)
            return
        if isinstance(node, ast.Call):
            chain = _attr_chain(node.func)
            path = None
            if chain:
                root, attrs = chain
                path = self._rooted(root, attrs)
            if path:
                self.report(path, node.lineno, node)
            else:
                self.scan(node.func)
            for arg in node.args:
                self.scan(arg)
            for kw in node.keywords:
                self.scan(kw.value)
            return
        if isinstance(node, ast.Try):
            self._try_stack.append(node)
            for child in ast.iter_child_nodes(node):
                self.scan(child)
            self._try_stack.pop()
            return
        for child in ast.iter_child_nodes(node):
            self.scan(child)


def collect_aliases(tree: ast.AST, roots: tuple[str, ...]) -> dict[str, str]:
    """Local name -> dotted module/symbol path, for engine-rooted imports.

    ``import a.b.c`` without an asname binds ``a`` — recording the deep path
    under ``a`` made every later full chain double itself
    (``a.b.c.a.b.c...``). Only asnamed plain imports are recorded; the bare
    top-level name already resolves through the default root. File-global
    approximation: local shadowing is rare and at worst produces one false
    entry whose detail line names the site for a human to dismiss.
    """
    aliases: dict[str, str] = {}

    def rooted(path: str) -> bool:
        return any(path == r or path.startswith(r + ".") for r in roots)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                if name.asname and rooted(name.name):
                    aliases[name.asname] = name.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for name in node.names:
                target = f"{node.module}.{name.name}"
                if rooted(target):
                    aliases[name.asname or name.name] = target
    return aliases


def package_root(package: str) -> Path | None:
    try:
        module = importlib.import_module(package)
    except Exception:  # noqa: BLE001 - surfaced as a DRIFT check by caller
        return None
    if module.__file__ is None:
        return None
    return Path(module.__file__).parent


def focus_suffixes(model_config: str | None, engine: str,
                   plugin: str) -> tuple[list[str], list[str] | None, list[dict]]:
    """Plugin ``models/`` modules this deployment loads, from the registry.

    The engine's own ModelRegistry maps architecture -> module_name; the
    plugin's corresponding file is the same suffix under the plugin package
    (that is where the drift repairs for GlmMoeDsaForCausalLM live). Returns
    (architectures, suffixes, notes). ``suffixes is None`` means the scope
    could NOT be resolved — the fail-safe direction: every model file is
    then treated as on this deployment's path, because guessing "not my
    file" the other way would silently ungated real drift.
    """
    if not model_config:
        return [], None, [{"kind": "scope", "id": "model-config",
                           "verdict": "WARN", "detail": "no --model-config given: "
                           "gate covers the whole plugin package",
                           "symbol": "scope", "sites": 1,
                           "gating": "report-only"}]
    try:
        architectures = json.loads(
            Path(model_config).read_text(encoding="utf-8")
        ).get("architectures", [])
    except (OSError, ValueError) as error:
        return [], None, [{"kind": "scope", "id": "model-config",
                           "verdict": "WARN", "detail": f"unreadable: {error}",
                           "symbol": "scope", "sites": 1,
                           "gating": "report-only"}]
    notes: list[dict] = []
    suffixes: list[str] = []
    resolved = False
    try:
        from vllm.model_executor.models import registry as reg  # noqa: PLC0415
        mapping = getattr(getattr(reg, "ModelRegistry", None), "models", None)
        if not isinstance(mapping, dict):
            raise RuntimeError("ModelRegistry.models is not a dict")
        resolved = True
        for arch in architectures:
            entry = mapping.get(arch)
            module = getattr(entry, "module_name", None)
            if not module:
                notes.append({"kind": "scope", "id": f"arch:{arch}",
                              "verdict": "WARN",
                              "detail": "not in the registry: gate covers the "
                                        "whole plugin package",
                              "symbol": "scope", "sites": 1,
                              "gating": "report-only"})
                resolved = False
                continue
            # The engine registers model modules as
            # <engine>.model_executor.models.<name>, while the plugin's
            # corresponding file is models/<name>.py — that is where the
            # drift repairs for GlmMoeDsaForCausalLM live.
            engine_models_prefix = f"{engine}.model_executor.models."
            if module.startswith(engine_models_prefix):
                suffixes.append(f"models.{module[len(engine_models_prefix):]}")
            elif module.startswith(f"{engine}.models."):
                suffixes.append(
                    f"models.{module[len(f'{engine}.models.'):]}")
            elif module.startswith(f"{plugin}."):
                suffixes.append(module[len(plugin) + 1:])
    except Exception as error:  # noqa: BLE001 - fail safe, gate everything
        notes.append({"kind": "scope", "id": "registry",
                      "verdict": "WARN",
                      "detail": f"architecture scoping unavailable ({error}): "
                                "gate covers the whole plugin package",
                      "symbol": "scope", "sites": 1,
                      "gating": "report-only"})
    return architectures, (suffixes if resolved else None), notes


def sweep_plugin(plugin: str, engine: str, checks: list[dict],
                 model_suffixes: list[str] | None,
                 gate_prefixes: tuple[str, ...]) -> dict:
    root = package_root(plugin)
    if root is None:
        checks.append({"kind": "resolve", "id": f"import {plugin}",
                       "verdict": "DRIFT", "symbol": plugin, "sites": 1,
                       "scope": "path", "gating": "gate",
                       "detail": "the plugin package itself does not import"})
        return {"references": 0, "pass": 0, "drift": 1, "warn": 0,
                "unverifiable": 0, "files": 0, "truncated": False,
                "drift_out_of_path": 0, "drift_report_only": 0}
    roots = (engine,)
    merged: list[dict] = []
    total = {"references": 0, "pass": 0, "drift": 0, "warn": 0,
             "unverifiable": 0, "files": 0, "truncated": False,
             "drift_out_of_path": 0, "drift_report_only": 0}
    pass_index: dict[str, dict] = {}
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_bytes())
        except (SyntaxError, ValueError) as error:
            merged.append({"kind": "parse", "id": str(path),
                           "verdict": "WARN", "detail": str(error),
                           "symbol": str(path), "sites": 1, "scope": "path",
                           "gating": "report-only"})
            total["warn"] += 1
            continue
        relative = str(path.relative_to(root))
        # Scoping: model files are on this deployment's path when they are
        # the module the registry maps the config's architectures to. An
        # unresolved scope (None) keeps every model file on-path — fail
        # safe, because un-gating real drift silently is the worse error.
        is_model_file = relative.startswith("models/")
        module_suffix = relative[:-len(".py")].replace("/", ".")
        on_path = (not is_model_file
                   or model_suffixes is None
                   or module_suffix in model_suffixes)
        # Gating: only the unconditional init surface blocks the launch.
        gating = "gate" if relative.startswith(gate_prefixes) else "report-only"
        if is_model_file and not on_path:
            gating = "report-only"  # another model's file, never ours
        scanner = Scanner(roots, collect_aliases(tree, roots))
        scanner.filename = relative
        scanner.scan(tree)
        total["files"] += 1
        for key in ("references", "pass", "drift", "warn", "unverifiable"):
            total[key] += scanner.stats[key]
        total["truncated"] = total["truncated"] or scanner.stats["truncated"]
        for entry in scanner.checks:
            entry["scope"] = "path" if on_path else "other-models"
            # Call-site failures gate on the init surface; bare attribute
            # reads are always findings-only (conditional arms dominate the
            # false-positive class, and the load-bearing reads are curated).
            entry["gating"] = gating if entry["kind"] == "bind" else "report-only"
            if entry["verdict"] == "DRIFT":
                if not on_path:
                    total["drift_out_of_path"] += 1
                elif gating == "report-only":
                    total["drift_report_only"] += 1
            if entry["verdict"] == "PASS":
                prior = pass_index.get(entry["symbol"])
                if prior is not None:
                    prior["sites"] += entry["sites"]
                    continue
                pass_index[entry["symbol"]] = entry
            merged.append(entry)
    if len(merged) > _MAX_REPORTED:
        total["truncated"] = True
    checks.extend(merged[:_MAX_REPORTED])
    return total


def curated_checks(engine: str, plugin: str, checks: list[dict]) -> dict:
    """The known landmines and the repaired plugin-state, by name.

    Only meaningful for the real package pair; a fixture sweep in tests passes
    other names and exercises the generic machinery alone.
    """
    stats = {"pass": 0, "drift": 0, "warn": 0, "unverifiable": 0}
    if (engine, plugin) != ("vllm", "vllm_kunlun"):
        return stats

    def record(kind: str, path: str, verdict: str, why: str) -> None:
        checks.append({"kind": kind, "id": path, "verdict": verdict,
                       "detail": why, "symbol": path, "sites": 1,
                       "scope": "path", "gating": "gate"})
        stats[verdict.lower()] += 1

    for path, why in CURATED_SYMBOLS:
        obj, error, _kind = resolve(path)
        record("curated-symbol", path,
               "PASS" if obj is not None else "DRIFT",
               why if obj is not None else f"{why} — AND {error}")
    for path, why in CURATED_OPTIONAL:
        obj, _error, _kind = resolve(path)
        record("curated-optional", path, "PASS" if obj is not None else "WARN", why)
    for path, why in CURATED_PLUGIN_STATE:
        obj, error, _kind = resolve(path)
        record("plugin-state", path, "PASS" if obj is not None else "DRIFT",
               why if obj is not None else f"{why} — drift repairs not applied: {error}")

    # Executable: the platform must answer is_cuda_alike truthily — an OOT
    # enum whose real contract is CUDA-alike is exactly landmine 11.
    try:
        platform_module = importlib.import_module("vllm_kunlun.platforms.kunlun")
        platform_cls = getattr(platform_module, "KunlunPlatform")
        ok = platform_cls().is_cuda_alike()
        verdict = "PASS" if ok else "DRIFT"
        detail = ("is_cuda_alike() answers the platform's real contract"
                  if ok else
                  "torch_xmlir maps XPU onto the torch.cuda API; is_cuda_alike "
                  "must answer True even for the OOT PlatformEnum")
    except Exception as error:  # noqa: BLE001 - an init-time crash IS drift
        verdict, detail = "DRIFT", f"platform probe failed: {error}"
    record("exec", "vllm_kunlun.platforms.kunlun.KunlunPlatform.is_cuda_alike",
           verdict, detail)

    # Executable: the VllmModel protocol (drift 4) on the model classes the
    # registry will instantiate — absent methods kill "--runner generate"
    # before any weights load. An import failure of the module is MAT-027's
    # domain (gated there); it is reported here as WARN, not double-gated.
    try:
        models = importlib.import_module("vllm_kunlun.models.deepseek_v2")
        missing = [
            f"{name}.{method}"
            for name, obj in vars(models).items()
            if inspect.isclass(obj) and name.endswith("ForCausalLM")
            for method in ("embed_input_ids", "compute_logits")
            if not callable(getattr(obj, method, None))
        ]
        verdict, detail = (
            ("PASS", "model classes implement the VllmModel protocol")
            if not missing else ("DRIFT", f"missing: {', '.join(missing)}")
        )
    except Exception as error:  # noqa: BLE001
        verdict, detail = "WARN", (
            f"model module import failed (MAT-027's domain): {error}"
        )
    record("exec", "vllm_kunlun.models.deepseek_v2.*ForCausalLM VllmModel-protocol",
           verdict, detail)
    return stats


def package_identity(name: str) -> dict:
    try:
        return {"name": name,
                "version": importlib.metadata.distribution(name).version}
    except importlib.metadata.PackageNotFoundError:
        return {"name": name, "version": "unknown"}


_SETTLE_CODE = """
import importlib, json, sys
paths = json.loads(sys.stdin.read())
out = {}
for path in paths:
    parts = path.split('.')
    for split in range(len(parts), 0, -1):
        try:
            module = importlib.import_module('.'.join(parts[:split]))
        except ModuleNotFoundError:
            continue
        except Exception:
            out[path] = 'import-raised'
            break
        obj = module
        try:
            for attr in parts[split:]:
                obj = getattr(obj, attr)
        except AttributeError:
            out[path] = 'missing'
            break
        out[path] = 'resolved'
        break
    else:
        out[path] = 'missing'
print(json.dumps(out))
"""


def settle_disputes(paths: list[str]) -> dict[str, str]:
    """Re-resolve disputed paths in a clean, engine-first interpreter.

    The probe process imports the plugin, which mutates engine module state
    (op registration, compat rewrites); a symbol that resolves cleanly
    engine-first was never missing — the probe's own import order broke the
    lookup, and that is a WARN, not drift. One batched subprocess, so the
    cost is one interpreter warm-up, not one per path.
    """
    if not paths:
        return {}
    try:
        result = subprocess.run(
            [sys.executable, "-c", _SETTLE_CODE],
            input=json.dumps(sorted(set(paths))),
            text=True, capture_output=True, timeout=600,
        )
        for line in reversed(result.stdout.strip().splitlines()):
            if line.startswith("{"):
                return json.loads(line)
    except (OSError, subprocess.SubprocessError):
        pass
    # Settlement unavailable: keep the probe's own verdict (fail safe).
    return {}


def warmup_engine(engine: str, checks: list[dict]) -> None:
    """Import the colliding engine modules before the plugin, server-order.

    The server imports the engine before activating the plugin; importing the
    plugin first makes later engine imports raise "Duplicate op name" and
    poisons resolution for the rest of the sweep.
    """
    if engine != "vllm":
        return
    for module_name in WARMUP_ENGINE_MODULES:
        obj, error, kind = resolve(module_name)
        if obj is None and kind == "missing":
            checks.append({"kind": "warmup", "id": module_name,
                           "verdict": "DRIFT", "detail": error,
                           "symbol": module_name, "sites": 1, "scope": "path",
                           "gating": "gate"})
        elif obj is None:
            checks.append({"kind": "warmup", "id": module_name,
                           "verdict": "WARN", "detail": error,
                           "symbol": module_name, "sites": 1, "scope": "path",
                           "gating": "report-only"})


def _reclassify_disputed(checks: list[dict],
                         settlements: dict[str, str]) -> int:
    """Apply the clean-interpreter verdict to would-gate missing symbols.

    Returns how many entries were downgraded DRIFT -> WARN.
    """
    downgraded = 0
    for entry in checks:
        if (entry.get("verdict") == "DRIFT"
                and entry.get("kind") == "bind"
                and entry.get("resolution") == "missing"
                and entry.get("scope", "path") == "path"
                and entry.get("gating", "gate") == "gate"):
            verdict = settlements.get(entry.get("symbol"))
            if verdict == "resolved":
                entry["verdict"] = "WARN"
                entry["detail"] += (" — resolves in a clean engine-first "
                                    "interpreter (probe import-order artifact)")
                downgraded += 1
            elif verdict == "import-raised":
                entry["verdict"] = "WARN"
                entry["detail"] += (" — import raised in a clean interpreter "
                                    "too (environment, not drift)")
                downgraded += 1
    return downgraded


def build_report(plugin: str, engine: str, model_config: str | None = None,
                 gate_prefixes: tuple[str, ...] = GATE_PREFIXES,
                 disputer=settle_disputes) -> dict:
    checks: list[dict] = []
    # Engine modules first, then the plugin — the server's order (see the
    # warmup docstring). The plugin's startup stages install attributes the
    # engine's own modules do not ship, and later resolution must see them.
    warmup_engine(engine, checks)
    try:
        importlib.import_module(plugin)
    except Exception as error:  # noqa: BLE001 - recorded, not fatal here
        checks.append({"kind": "resolve", "id": f"import {plugin}",
                       "verdict": "DRIFT", "detail": str(error),
                       "symbol": plugin, "sites": 1, "scope": "path",
                       "gating": "gate"})

    architectures, suffixes, scope_notes = focus_suffixes(
        model_config, engine, plugin)
    checks.extend(scope_notes)
    sweep = sweep_plugin(plugin, engine, checks, suffixes, gate_prefixes)
    curated = curated_checks(engine, plugin, checks)
    # A would-gate "missing" symbol may be an artifact of this probe's own
    # import order: settle those in a clean engine-first interpreter before
    # any verdict becomes the launch decision.
    disputed = sorted({
        c["symbol"] for c in checks
        if c.get("verdict") == "DRIFT"
        and c.get("kind") == "bind"
        and c.get("resolution") == "missing"
        and c.get("scope", "path") == "path"
        and c.get("gating", "gate") == "gate"
    })
    _reclassify_disputed(checks, disputer(disputed))

    def sites(verdict: str, **filters) -> int:
        return sum(c.get("sites", 1) for c in checks
                   if c.get("verdict") == verdict
                   and all(c.get(key) == value for key, value in filters.items()))

    gated_drift = sites("DRIFT", scope="path", gating="gate")
    return {
        "state": "DRIFT" if gated_drift else "PASS",
        "engine": package_identity(engine),
        "plugin": {
            **package_identity(plugin),
            "files_scanned": sweep.get("files", 0),
            "engine_rooted_references": sweep.get("references", 0),
            "truncated": sweep.get("truncated", False),
        },
        "scope": {
            "model_config": model_config,
            "architectures": architectures,
            "model_modules": suffixes,
            "gate_prefixes": list(gate_prefixes),
        },
        "checks": checks,
        "summary": {
            "pass": sites("PASS"),
            "drift": sites("DRIFT"),
            "warn": sites("WARN"),
            "unverifiable": sites("UNVERIFIABLE"),
            "drift_gated": gated_drift,
            "drift_report_only": sites("DRIFT", scope="path",
                                       gating="report-only"),
            "drift_out_of_path": sites("DRIFT", scope="other-models"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin", default="vllm_kunlun")
    parser.add_argument("--engine", default="vllm")
    parser.add_argument("--model-config", default=None,
                        help="path to the target model's config.json; scopes "
                             "the gate to the modules this deployment loads")
    parser.add_argument("--gate-prefixes", default=",".join(GATE_PREFIXES),
                        help="comma-separated plugin file prefixes that hard-"
                             "gate; sweep findings elsewhere are report-only")
    args = parser.parse_args()
    prefixes = tuple(
        p for p in (part.strip() for part in args.gate_prefixes.split(",")) if p
    ) or GATE_PREFIXES
    report = build_report(args.plugin, args.engine, args.model_config, prefixes)
    # One line: the launcher parses the last stdout line, exactly like
    # MAT-027's probe. Indentation belongs to the persisted artifact copy.
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["state"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
