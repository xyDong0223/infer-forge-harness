"""Small, dependency-light tensor diff primitives and JSON CLI.

The comparator is intentionally independent of any accelerator framework. It
accepts JSON arrays so probes can dump tensors from PyTorch, XPU, or a vendor
binding and grade them with one implementation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from core.paths import REPO_ROOT

from core.storage import ArtifactStore, WritePolicyError, ensure_external, locate_attempt


def _flat(value: Any) -> list[float]:
    if isinstance(value, list):
        result: list[float] = []
        for item in value:
            result.extend(_flat(item))
        return result
    return [float(value)]


def _shape(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return [len(value), *_shape(value[0])] if value else [0]


def _has_irregular_shape(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    shapes = [_shape(item) for item in value]
    return len(set(map(tuple, shapes))) != 1 or any(
        _has_irregular_shape(item) for item in value
    )


def compare(candidate: Any, reference: Any, atol: float = 0.0) -> dict[str, Any]:
    left, right = _flat(candidate), _flat(reference)
    if len(left) != len(right):
        raise ValueError(f"tensor sizes differ: {len(left)} != {len(right)}")
    if _has_irregular_shape(candidate) or _has_irregular_shape(reference):
        raise ValueError("tensor values must be rectangular arrays")
    if _shape(candidate) != _shape(reference):
        raise ValueError(f"tensor shapes differ: {_shape(candidate)} != {_shape(reference)}")
    diffs = [abs(a - b) for a, b in zip(left, right)]
    non_finite = any(not math.isfinite(value) for value in left + right)
    ref_norm = math.sqrt(sum(value * value for value in right))
    rel_l2 = math.sqrt(sum(value * value for value in diffs)) / max(ref_norm, atol, 1e-30)
    if non_finite:
        rel_l2 = math.inf
    finite_candidate = sum(math.isfinite(value) for value in left)
    finite_reference = sum(math.isfinite(value) for value in right)
    return {
        "shape_candidate": _shape(candidate),
        "shape_reference": _shape(reference),
        "numel": len(left),
        "relative_l2": rel_l2,
        "max_abs_error": max(diffs, default=0.0),
        "nan_candidate": sum(math.isnan(value) for value in left),
        "nan_reference": sum(math.isnan(value) for value in right),
        "inf_candidate": len(left) - finite_candidate - sum(math.isnan(value) for value in left),
        "inf_reference": len(right) - finite_reference - sum(math.isnan(value) for value in right),
    }


def grade(
    candidate: Any,
    reference: Any,
    control: Any | None = None,
    max_relative_l2: float | None = None,
) -> dict[str, Any]:
    result = compare(candidate, reference)
    if control is not None:
        result["control"] = compare(control, reference)
        result["control_discriminates"] = max_relative_l2 is not None and (
            result["control"]["relative_l2"] > max_relative_l2
        )
    if max_relative_l2 is not None:
        result["pass"] = result["relative_l2"] <= max_relative_l2
    return result


def execute(args) -> int:
    lexical = args.out.expanduser().absolute()
    args.out = ensure_external(lexical)
    attempt = locate_attempt(args.out)
    if attempt is None:
        for parent in lexical.parents:
            if (parent / ".attempt.json").exists():
                attempt = locate_attempt(parent)
                break
    if attempt is not None and attempt.output not in args.out.parents:
        raise WritePolicyError("tensor report must be inside the owning attempt output")
    store = ArtifactStore(args.out.parent)
    if args.out.exists() or args.out.with_name(args.out.name + ".manifest.json").exists():
        raise WritePolicyError(f"tensor report output is already owned: {args.out}")
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    control = json.loads(args.control.read_text(encoding="utf-8")) if args.control else None
    result = grade(candidate, reference, control, args.max_relative_l2)
    store.write_text(args.out.name, json.dumps(result, indent=2) + "\n")
    store.register(
        identity={**(attempt.identity if attempt is not None else {}), "tool_id": "tensor_diff"},
        outcome="COMPLETED" if result.get("pass", True) else "FAILED",
        include=[args.out.name], required=[args.out.name],
        manifest_name=args.out.name + ".manifest.json",
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("pass", True) else 6
