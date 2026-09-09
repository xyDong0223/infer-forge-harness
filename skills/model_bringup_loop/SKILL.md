---
name: model-bringup-loop
description: Bring a model up on P800 through evidence-driven failure classification and correction.
---

# Model Bring-up Loop

Keep the model goal stable while iterating over small evidence-producing blocks:

1. Observe the smallest failing request and capture the actual call path.
2. Classify the failure as missing capability, bypassed capability, unlaunchable
   implementation, executable-but-wrong implementation, runtime-state failure,
   or API incompatibility.
3. Choose the cheapest discriminating experiment.
4. Apply only a reversible workaround, then grade it against an independent reference.
5. Re-run the integrated service path and record the next block from evidence.

Never infer correctness from startup completion, HTTP 200, or symbol presence.
Every block needs an exit condition and its artifacts must include the runtime,
model revision, and platform revision.
