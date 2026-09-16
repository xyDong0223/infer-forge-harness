"""Runtime registry: get a runtime by canonical name.

Support relationships are declared in `catalog/runtime_catalog.yaml`, not
implied by import paths. Declaring `sglang-kunlun` there is what makes it
exist for the phase-2 compatibility check; implementing it is what makes it
run. The registry is the only place that maps names to implementations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from runtimes.vllm_kunlun import VllmKunlunRuntime

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG = REPO_ROOT / "catalog" / "runtime_catalog.yaml"

RegistryError = ValueError

_LOADERS: dict[str, Callable[[], object]] = {
    "vllm-kunlun": VllmKunlunRuntime.load,
}

_DEFAULT: object | None = None


def _entries() -> list[dict]:
    import yaml

    raw = yaml.safe_load(CATALOG.read_text(encoding="utf-8")) or {}
    entries = raw.get("entries") or []
    if not entries:
        raise RegistryError(f"{CATALOG} declares no runtime entries")
    return entries


def get_runtime(name: str = "vllm-kunlun") -> object:
    """Return the runtime registered under canonical `name`.

    The catalog entry must exist (declared support) and a loader must be
    wired (implemented support); a declared-but-unimplemented runtime fails
    loudly instead of silently falling back to the default one.
    """
    global _DEFAULT
    entry = next((e for e in _entries() if e.get("name") == name), None)
    if entry is None:
        known = ", ".join(sorted(str(e.get("name")) for e in _entries()))
        raise RegistryError(f"unknown runtime {name!r}; catalog declares: {known}")
    loader = _LOADERS.get(name)
    if loader is None:
        raise RegistryError(
            f"runtime {name!r} is declared in the catalog but no implementation is wired"
        )
    return loader()


def default_runtime() -> object:
    """The cached default runtime (the only wired stack today)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = get_runtime()
    return _DEFAULT


def resolve_runtime(name: str) -> object:
    """Facade-friendly runtime resolver with no implicit fallback."""
    return get_runtime(name)
