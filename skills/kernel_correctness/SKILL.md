---
name: kernel-correctness
description: Locate P800 numerical mismatches with independent references and discriminating controls.
---

# Kernel Correctness Skill

Use this Skill when a served model produces a plausible response but numerical
evidence disagrees with an independent reference.

## Method

1. Pin the model, runtime, image, and serving process.
2. Capture the smallest failing shape and record the actual call arguments.
3. Compare against a reference that the candidate implementation did not write.
4. Use a negative control that would fail if the suspected bug were absent.
5. Sweep structural explanations before assigning ownership to a platform kernel.
6. Apply the narrowest reversible workaround and rerun both service and isolated checks.

## Rules

- Norm and cosine alone are insufficient; use relative L2 for tensor outputs.
- Check warmup, empty tensors, metadata fields, cache axes, and causal boundaries.
- A successful HTTP response is not accuracy evidence.
- Record before/after values, control behavior, exact code locations, and limitations.

## Exit

The Skill ends with `ACCURACY_PASS`, an evidence-backed `TRIAGE_READY`, or
`NEEDS_HUMAN`; it must never silently convert an unreachable probe into a pass.
