---
name: fallback-validation
description: Validate a reversible reference fallback without hiding unsupported cases.
---

# Fallback Validation Skill

Use this Skill when a vendor path is unavailable or fails and a reference
implementation is proposed as a temporary compatibility path.

## Method

1. Define the supported tensor layouts, dtypes, cache format, and execution phase.
2. Implement the smallest readable reference with explicit refusal conditions.
3. Compare output against an independent reference using shape checks and relative L2.
4. Add a negative control that would distinguish the suspected defect from a shared mistake.
5. Verify the real service dispatches through the fallback and run service regression checks.
6. Record performance and coverage limitations; route a durable operator request when optimization is required.

## Rules

- A fallback is not an XPU implementation and must never be reported as one.
- Unsupported features must raise an explicit error or remain on the vendor path.
- Do not validate a fallback only against its own arithmetic.
- Keep installation reversible and outside the installed package when possible.

## Exit

The Skill ends with `FALLBACK_VALIDATED`, `FALLBACK_REJECTED`, or `NEEDS_HUMAN`.
