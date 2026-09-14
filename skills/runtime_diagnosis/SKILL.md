---
name: runtime-diagnosis
description: Localize inference failures with reproducible traces and discriminating controls.
---

# Runtime Diagnosis Skill

Use this Skill when a model loads, imports, or serves incorrectly and the
failure boundary is not yet established.

## Method

1. Pin the engine, plugin, model revision, device, and serving command.
2. Capture the smallest failing input and the complete call-site arguments.
3. Trace layer or operator boundaries with bounded logging and optional tensor dumps.
4. Bisect from input to output, separating model, API, runtime-state, plugin, and vendor layers.
5. Reproduce the suspected cause with an independent control before assigning ownership.
6. Record a repair hypothesis and a concrete next action; do not silently retry.

## Rules

- Traces are evidence artifacts, not source files; write them under the external artifact root.
- Bound forward count, rank scope, token count, and tensor dump size.
- Distinguish an import failure, dispatch failure, numerical mismatch, and service failure.
- A successful health check does not prove numerical correctness.

## Exit

The Skill ends with `TRIAGE_READY`, `REDISCOVER_REQUIRED`, `REWORK`, or `NEEDS_HUMAN`.
