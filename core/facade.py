"""Single entry point for resolving platform adapters from a target."""

from dataclasses import dataclass
from typing import Any

from adapters import get_hardware
from core.contracts import TargetContext
from core.target import canonical_hardware, compatibility_status
from runtimes.registry import get_runtime


@dataclass(frozen=True)
class AdapterBundle:
    target: TargetContext
    hardware: type
    runtime: Any
    compatibility: dict[str, Any]


def resolve_adapters(target: TargetContext, *, require_supported: bool = False) -> AdapterBundle:
    if target.hardware != canonical_hardware(target.hardware):
        target = TargetContext(
            model=target.model,
            hardware=canonical_hardware(target.hardware),
            engine=target.engine,
            backend=target.backend,
            plugin=target.plugin,
            revisions=target.revisions,
        )
    status = compatibility_status(target)
    if require_supported and status["status"] != "supported":
        raise ValueError(
            f"cannot resolve unsupported target: {target.hardware} / "
            f"{target.engine} / {target.plugin or target.backend} "
            f"({status['status']})"
        )
    if status["status"] != "supported":
        return AdapterBundle(target, get_hardware(target.hardware), None, status)
    hardware = get_hardware(target.hardware)
    runtime_name = target.plugin or f"{target.engine}-{target.backend}"
    runtime = get_runtime(runtime_name)
    return AdapterBundle(target, hardware, runtime, status)
