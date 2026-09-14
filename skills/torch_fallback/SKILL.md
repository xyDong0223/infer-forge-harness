---
name: torch-fallback
description: Validate a reversible PyTorch fallback for an unavailable accelerator path.
---

# Torch Fallback Skill

Use this method only after the failing path and tensor contract are evidenced.
Define supported layouts, dtypes, cache formats, and execution phases; implement
the smallest fallback in `tools/torch/`; compare it with an independent reference
and a discriminating control; then verify real service dispatch and regression.

Unsupported inputs must be refused explicitly. A PyTorch fallback is compatibility
evidence, not an accelerator implementation or a performance result.

Exit with `FALLBACK_VALIDATED`, `FALLBACK_REJECTED`, or `NEEDS_HUMAN` and cite
external artifact paths for all runtime evidence.
