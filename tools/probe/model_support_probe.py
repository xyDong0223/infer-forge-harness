"""In-pod model support probe. Prints JSON to stdout.

Answers one question with six possible answers, because "supported / not
supported" is the wrong shape: an architecture missing from the Kunlun registry
is usually *not* a gap — it means the upstream generic implementation is used and
any failure lies elsewhere, typically in an operator. Conflating the two sends
someone off to write a model file that is not needed.

Escalates only as far as it must: installed stack first, then vLLM main, then
open pull requests. Most adaptation work happens on models already merged into
main but not yet released, so stopping at the installed version would report a
gap that is really a version lag.

The sixth answer exists because "resolves" is not "implemented for this hardware".
Upstream ships some models once per accelerator, and a registry lookup cannot see
which variant a platform lands on or what that variant hard-depends on.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request

MAIN_REGISTRY_URL = (
    "https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/registry.py"
)
PR_SEARCH_URL = (
    "https://api.github.com/search/issues?q=repo:vllm-project/vllm+is:pr+is:open+in:title+{arch}"
)
REGISTER_CALL = re.compile(r'register_model\(\s*"([A-Za-z0-9_]+)"')
TIMEOUT = 30

# Directory names upstream uses when it ships one model per accelerator. The point is
# not the list but the shape: a resolved architecture can still be an implementation
# written for someone else's hardware.
BACKEND_DIRS = {"nvidia", "amd", "cuda", "rocm", "xpu", "cpu", "tpu", "hpu", "neuron"}
HARD_IMPORT = re.compile(r"^\s*(?:from|import)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
CUSTOM_OP_CALL = re.compile(r"\bops\.([a-z_][a-z0-9_]*)\(")
# Modules every implementation imports; listing them as findings would bury the real ones.
UNINTERESTING = {
    "torch", "vllm", "typing", "collections", "dataclasses", "math", "os", "sys",
    "functools", "itertools", "enum", "abc", "copy", "json", "re", "warnings", "logging",
    "numpy", "transformers", "einops", "regex",
}


def fetch(url: str) -> tuple[int, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "kunlun-inference-agent"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        return error.code, error.reason or ""
    except Exception as error:  # network, DNS, proxy
        return 0, f"{type(error).__name__}: {error}"


def installed_state(arch: str) -> dict:
    import vllm
    import vllm_kunlun  # noqa: F401 - activates the platform plugin and its registry
    from vllm import ModelRegistry

    import inspect

    from vllm_kunlun.models import register_model

    oot = sorted(set(REGISTER_CALL.findall(inspect.getsource(register_model))))
    supported = sorted(ModelRegistry.get_supported_archs())
    return {
        "vllm_version": vllm.__version__,
        "kunlun_oot_archs": oot,
        "in_kunlun_oot": arch in oot,
        "in_installed_vllm": arch in supported,
        "installed_arch_count": len(supported),
    }


def upstream_state(arch: str) -> dict:
    status, body = fetch(MAIN_REGISTRY_URL)
    if status != 200:
        # An unreachable upstream is unknown, not absent. Reporting ABSENT here
        # would send someone to write a model that may already exist.
        return {"main_lookup": "UNKNOWN", "detail": f"{MAIN_REGISTRY_URL} -> {status} {body[:200]}"}
    return {"main_lookup": "FOUND" if arch in body else "NOT_FOUND", "detail": f"{len(body)} bytes"}


def pull_request_state(arch: str) -> dict:
    status, body = fetch(PR_SEARCH_URL.format(arch=arch))
    if status != 200:
        return {"pr_lookup": "UNKNOWN", "detail": f"search -> {status} {body[:200]}"}
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as error:
        return {"pr_lookup": "UNKNOWN", "detail": str(error)}
    items = [
        {"number": item.get("number"), "title": item.get("title"), "url": item.get("html_url")}
        for item in payload.get("items", [])
    ]
    return {"pr_lookup": "FOUND" if items else "NOT_FOUND", "pull_requests": items}


def _missing_dependencies(sources: list[str]) -> dict:
    """Imports that do not import here, and custom ops that are not registered here."""
    import importlib

    import torch

    joined = "\n".join(sources)
    missing_modules = []
    for name in sorted(set(HARD_IMPORT.findall(joined))):
        if name in UNINTERESTING or name.startswith("_"):
            continue
        try:
            importlib.import_module(name)
        except Exception as error:
            missing_modules.append({"module": name, "error": f"{type(error).__name__}: {error}"})

    namespace = getattr(torch.ops, "_C", None)
    missing_ops = []
    if namespace is not None:
        for name in sorted(set(CUSTOM_OP_CALL.findall(joined))):
            try:
                getattr(namespace, name)
            except Exception:
                missing_ops.append(name)
    return {"unimportable_modules": missing_modules, "unregistered_custom_ops": missing_ops}


def _read_variant_sources(directory: str) -> list[str]:
    import os

    sources = []
    for root, _, files in os.walk(directory):
        for name in files:
            if name.endswith(".py"):
                try:
                    with open(os.path.join(root, name), encoding="utf-8", errors="replace") as handle:
                        sources.append(handle.read())
                except OSError:
                    continue
    return sources


def backend_variant(arch: str) -> dict:
    """Which implementation of a resolved architecture will actually run here.

    "Registered" is not the same as "written for this hardware". Upstream ships some
    models once per accelerator — vllm/models/minimax_m3/{nvidia,amd,common} — and the
    package selects by platform predicate. vLLM-Kunlun answers False to every predicate
    upstream tests, so it lands on the nvidia variant and inherits its hard
    dependencies. A registry lookup cannot see that, and finding it one launch at a
    time cost four launches.

    The directory is inspected through `find_spec`, which does not execute the package,
    so a model whose selection itself fails on this platform can still be reported.
    """
    import importlib.util
    import os

    from vllm import ModelRegistry

    result: dict = {"inspected": True}
    entry = (getattr(ModelRegistry, "models", None) or {}).get(arch)
    if entry is None:
        return {"inspected": True, "lookup_error": "not in ModelRegistry.models",
                "vendored_per_backend": "UNKNOWN"}

    module_name = getattr(entry, "module_name", "") or ""
    result["registry_module"] = module_name
    try:
        spec = importlib.util.find_spec(module_name)
    except Exception as error:
        return {**result, "lookup_error": f"{type(error).__name__}: {error}",
                "vendored_per_backend": "UNKNOWN"}
    origin = getattr(spec, "origin", None) if spec else None
    if not origin:
        return {**result, "lookup_error": "no file origin", "vendored_per_backend": "UNKNOWN"}

    package_dir = os.path.dirname(origin)
    siblings = sorted(
        name for name in os.listdir(package_dir)
        if name in BACKEND_DIRS and os.path.isdir(os.path.join(package_dir, name))
    )
    result["backend_variants_present"] = siblings
    result["vendored_per_backend"] = len(siblings) > 1
    if len(siblings) < 2:
        # A single implementation is just an implementation; a failure in it is an
        # operator problem, which is what UPSTREAM_GENERIC already says.
        return result

    try:
        model_cls = entry.load_model_cls()
        selected = next(
            (part for part in getattr(model_cls, "__module__", "").split(".") if part in BACKEND_DIRS),
            None,
        )
        result["selected_variant"] = selected
    except Exception as error:
        # The selection itself failing here is the strongest possible answer to the
        # question, so it is recorded rather than swallowed.
        result["selection_error"] = f"{type(error).__name__}: {error}"
        result["selected_variant"] = "UNRESOLVED"
        result["variant_hard_dependencies"] = _missing_dependencies(
            [source for name in siblings if name != "common"
             for source in _read_variant_sources(os.path.join(package_dir, name))]
        )
        result["variant_is_runnable_here"] = False
        return result

    scan_dirs = [package_dir] if not selected else [os.path.join(package_dir, selected)]
    if "common" in os.listdir(package_dir):
        scan_dirs.append(os.path.join(package_dir, "common"))
    sources = [source for directory in scan_dirs for source in _read_variant_sources(directory)]
    result["scanned"] = scan_dirs
    result["variant_hard_dependencies"] = _missing_dependencies(sources)
    dependencies = result["variant_hard_dependencies"]
    result["variant_is_runnable_here"] = not (
        dependencies["unimportable_modules"] or dependencies["unregistered_custom_ops"]
    )
    return result


def classify(arch: str) -> dict:
    result: dict = {"architecture": arch}
    result.update(installed_state(arch))

    if result["in_kunlun_oot"]:
        result["verdict"] = "KUNLUN_OOT"
        result["meaning"] = "a Kunlun-specific implementation is registered and will be used"
        return result
    if result["in_installed_vllm"]:
        variant = backend_variant(arch)
        result["backend_variant"] = variant
        if variant.get("vendored_per_backend") is True and variant.get("variant_is_runnable_here") is False:
            result["verdict"] = "UPSTREAM_VENDORED_VARIANT"
            result["meaning"] = (
                "the architecture resolves, but upstream ships it once per accelerator and the "
                "variant selected here was written for other hardware: its hard dependencies are "
                "listed and are missing, so this is adaptation work, not a ready implementation"
            )
            return result
        result["verdict"] = "UPSTREAM_GENERIC"
        result["meaning"] = (
            "the installed vLLM implementation is used; the absence of a Kunlun OOT model is "
            "not a gap, so a runtime failure points at an operator or backend, not at networking"
        )
        return result

    result.update(upstream_state(arch))
    if result.get("main_lookup") == "FOUND":
        result["verdict"] = "MAIN_ONLY"
        result["meaning"] = "merged upstream but not in the installed version: upgrade or cherry-pick"
        return result

    result.update(pull_request_state(arch))
    if result.get("pr_lookup") == "FOUND":
        result["verdict"] = "PR_PENDING"
        result["meaning"] = "an open upstream pull request adds it: wait for merge or cherry-pick"
        return result
    if "UNKNOWN" in (result.get("main_lookup"), result.get("pr_lookup")):
        result["verdict"] = "UNKNOWN_UPSTREAM"
        result["meaning"] = "upstream could not be reached; absence cannot be concluded"
        return result

    result["verdict"] = "ABSENT"
    result["meaning"] = "no implementation anywhere: a model file has to be written"
    return result


def main(archs: list[str]) -> int:
    out = {
        "state": "SCAN_READY",
        "proxy": os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY") or "",
        "results": [],
    }
    try:
        for arch in archs:
            out["results"].append(classify(arch))
    except Exception as error:  # an unusable runtime is not a scan result
        print(json.dumps({"state": "SCAN_FAILED", "reason": f"{type(error).__name__}: {error}"}))
        return 0
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"state": "CONTRACT_INVALID", "reason": "usage: probe <Arch> [Arch ...]"}))
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1:]))
