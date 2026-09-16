"""Adapters isolate Kubernetes, Kunlun P800, and vLLM-Kunlun differences.

`get_hardware(name)` is the seam between the core and concrete adapters:
runners and tools reach the adapter through this factory, never by
importing `adapters.kunlun_p800` directly (tests/unit/test_runtimes.py
enforces that invariant). Phase 2 threads the name in from TargetContext;
    today the default is the only wired hardware.
"""

from __future__ import annotations

from adapters.kunlun_p800.adapter import (
    ClusterConfig,
    KunlunP800Adapter,
    SafetyViolation,
    push_snippet,
)
from adapters.kunlun_p800.adapter import KunlunP800Adapter as _P800

__all__ = [
    "ClusterConfig",
    "KunlunP800Adapter",
    "SafetyViolation",
    "get_hardware",
    "push_snippet",
]

_HARDWARE: dict[str, type] = {
    "kunlun-p800": _P800,
    "kunlun/p800": _P800,
}


def get_hardware(name: str = "kunlun-p800") -> type:
    """Return the adapter class registered under canonical hardware `name`."""
    try:
        return _HARDWARE[name]
    except KeyError:
        raise ValueError(
            f"no hardware adapter registered for {name!r}; known: {sorted(_HARDWARE)}"
        ) from None


def resolve_hardware(name: str) -> type:
    """Facade-friendly alias; keeps platform selection at the boundary."""
    return get_hardware(name)
