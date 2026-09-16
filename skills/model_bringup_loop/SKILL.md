---
name: model-bringup-loop
description: Bring a model up on P800 through evidence-driven failure classification and correction.
---

# Model Bring-up Loop

Keep the model goal stable while iterating over small evidence-producing blocks:

1. Resume the recorded Pod and runtime; never roll/recreate it for a model error.
   First prove the environment using the configured MiniMax-M2.5 base smoke.
   Do not run the historical Kunlun drift patch as an automatic preparation step.
2. Run dummy-weight toy bring-up before loading target weights, and again after
   every repair. Require engine construction, prefill and two-token decode.
3. Observe the smallest failing request and capture the actual call path.
4. Classify the failure as missing capability, bypassed capability, unlaunchable
   implementation, executable-but-wrong implementation, runtime-state failure,
   or API incompatibility.
5. Choose the cheapest discriminating experiment.
6. Apply only a reversible workaround, then grade it against an independent reference.
7. Re-run the integrated service path and record the next block from evidence.

Never infer correctness from startup completion, HTTP 200, or symbol presence.
Every block needs an exit condition and its artifacts must include the runtime,
model revision, and platform revision.
