"""Target resolution and compatibility gate for platform combinations."""

from pathlib import Path
from typing import Any
import yaml

from core.contracts import TargetContext

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "compatibility" / "matrix.yaml"
HARDWARE_ALIASES = {
    "Kunlunxin-3-P800": "kunlun/p800",
    "p800": "kunlun/p800",
    "kunlun-p800": "kunlun/p800",
}


def canonical_hardware(name: str) -> str:
    """Normalize legacy display names at the configuration boundary."""
    return HARDWARE_ALIASES.get(name, name)


def target_from_mapping(data: dict[str, Any]) -> TargetContext:
    target = data.get("target", data)
    runtime = target.get("runtime", {})
    return TargetContext(
        model=str(target.get("model", "<unspecified>")),
        hardware=canonical_hardware(str(target["hardware"])),
        engine=str(runtime["engine"]),
        backend=str(runtime["backend"]),
        plugin=runtime.get("plugin"),
        revisions=dict(runtime.get("revisions", {})),
    )


def load_target(path: str | Path) -> TargetContext:
    with Path(path).open(encoding="utf-8") as stream:
        return target_from_mapping(yaml.safe_load(stream) or {})


def compatibility_status(context: TargetContext) -> dict[str, Any]:
    entries = yaml.safe_load(MATRIX.read_text(encoding="utf-8")).get("entries", [])
    match = next(
        (
            entry for entry in entries
            if entry.get("hardware") == context.hardware
            and entry.get("engine") == context.engine
            and entry.get("backend") == context.backend
            and entry.get("plugin") == context.plugin
        ),
        None,
    )
    if match is None:
        return {"status": "unknown", "reason": "combination is not declared"}
    return {
        "status": match["status"],
        "id": match.get("id"),
        "reason": match.get("reason", ""),
    }


def require_supported(context: TargetContext) -> dict[str, Any]:
    result = compatibility_status(context)
    if result["status"] != "supported":
        raise ValueError(
            f"target {context.hardware} + {context.engine} + "
            f"{context.plugin or context.backend} is {result['status']}: "
            f"{result.get('reason', '')}"
        )
    return result
