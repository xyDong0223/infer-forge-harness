"""Target resolution and compatibility gate for platform combinations."""

from pathlib import Path
from typing import Any
import yaml

from core.contracts import TargetContext

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "compatibility" / "matrix.yaml"
HARDWARE_ALIASES = {
    "Kunlunxin-3-P800": "kunlun/p800",
    "P800": "kunlun/p800",
    "p800": "kunlun/p800",
    "kunlun-p800": "kunlun/p800",
}


def canonical_hardware(name: str) -> str:
    """Normalize legacy display names at the configuration boundary."""
    return HARDWARE_ALIASES.get(name, name)


def target_from_mapping(data: dict[str, Any]) -> TargetContext:
    if not isinstance(data, dict):
        raise ValueError("target must be a mapping")
    target = data.get("target", data)
    if not isinstance(target, dict):
        raise ValueError("target must be a mapping")
    runtime = target.get("runtime", {})
    if not isinstance(runtime, dict):
        raise ValueError("target.runtime must be a mapping")
    for name, value in (("hardware", target.get("hardware")),
                        ("runtime.engine", runtime.get("engine")),
                        ("runtime.backend", runtime.get("backend"))):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"target.{name} must be a nonempty string")
    plugin = runtime.get("plugin")
    if plugin is not None and not isinstance(plugin, str):
        raise ValueError("target.runtime.plugin must be a string or null")
    model = target.get("model", "<unspecified>")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("target.model must be a nonempty string")
    return TargetContext(
        model=model.strip(),
        hardware=canonical_hardware(target["hardware"].strip()),
        engine=runtime["engine"].strip(),
        backend=runtime["backend"].strip(),
        plugin=plugin.strip() or None if plugin is not None else None,
        revisions=normalize_revisions(runtime.get("revisions", {})),
    )


def normalize_revisions(revisions: dict) -> dict[str, str]:
    if not isinstance(revisions, dict):
        raise ValueError("runtime.revisions must be a mapping")
    result = {}
    for key, value in revisions.items():
        if (not isinstance(key, str) or not key.strip()
                or not isinstance(value, (str, int)) or isinstance(value, bool)
                or not str(value).strip()):
            raise ValueError("runtime.revisions requires nonempty names and values")
        result[key.strip()] = str(value).strip()
    return result


def bind_subject(target: TargetContext, subject: str) -> TargetContext:
    """Examples may leave the model unbound; a concrete identity must agree."""
    if target.model not in ("<unspecified>", "<model-id>", subject):
        raise ValueError(f"target model {target.model!r} does not match subject {subject!r}")
    return TargetContext(subject, target.hardware, target.engine, target.backend,
                         target.plugin, normalize_revisions(target.revisions))


def target_environment(target: TargetContext) -> dict[str, str]:
    return {
        "model": target.model,
        "hardware": canonical_hardware(target.hardware),
        "engine": target.engine,
        "backend": target.backend,
        "plugin": target.plugin or "",
        **{f"{name}_revision": value
           for name, value in normalize_revisions(target.revisions).items()},
    }


def contract_target(contract: dict, requested: TargetContext | None = None) -> TargetContext:
    """Resolve legacy defaults without letting an explicit target override a contract."""
    context = contract.get("context", {})
    if not isinstance(context, dict) or not isinstance(context.get("target", {}), dict):
        raise ValueError("contract context and context.target must be mappings")
    model = context.get("model", {})
    runtime = context.get("runtime", {})
    if not isinstance(model, dict) or not isinstance(runtime, dict):
        raise ValueError("contract context.model and context.runtime must be mappings")
    actual = target_from_mapping({
        "model": model.get("name", "<unspecified>"),
        "hardware": context.get("target", {}).get("hardware", "kunlun/p800"),
        "runtime": {"engine": "vllm", "backend": "kunlun", "plugin": "vllm-kunlun", **runtime},
    })
    environment_only = contract.get("metadata", {}).get("task_type") == "environment_proof"
    if not environment_only and model.get("revision") is not None:
        revision = normalize_revisions({"model": model["revision"]})["model"]
        if actual.revisions.get("model", revision) != revision:
            raise ValueError("contract model revision conflicts with runtime.revisions.model")
        actual = TargetContext(
            actual.model, actual.hardware, actual.engine, actual.backend, actual.plugin,
            {**actual.revisions, "model": revision},
        )
    if requested is None:
        return actual
    require_supported(requested)
    for axis in ("hardware", "engine", "backend", "plugin"):
        expected = getattr(requested, axis)
        if axis == "hardware":
            expected = canonical_hardware(expected)
        if getattr(actual, axis) != expected:
            raise ValueError(f"contract target {axis}={getattr(actual, axis)!r} "
                             f"does not match requested {expected!r}")
    # The environment-only contract deliberately runs a base-model smoke test.
    if not environment_only and requested.model not in ("<unspecified>", "<model-id>"):
        if actual.model != requested.model:
            raise ValueError(f"contract model {actual.model!r} does not match "
                             f"requested subject {requested.model!r}")
    for name, revision in normalize_revisions(requested.revisions).items():
        if name in actual.revisions and actual.revisions[name] != revision:
            raise ValueError(f"contract {name} revision does not match requested target")
    return TargetContext(
        actual.model, actual.hardware, actual.engine, actual.backend, actual.plugin,
        {**requested.revisions, **actual.revisions},
    )


def load_target(path: str | Path) -> TargetContext:
    try:
        with Path(path).open(encoding="utf-8") as stream:
            return target_from_mapping(yaml.safe_load(stream) or {})
    except yaml.YAMLError as error:
        raise ValueError(f"invalid target YAML: {error}") from error


def compatibility_status(context: TargetContext) -> dict[str, Any]:
    entries = yaml.safe_load(MATRIX.read_text(encoding="utf-8")).get("entries", [])
    match = next(
        (
            entry for entry in entries
            if entry.get("hardware") == canonical_hardware(context.hardware)
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
