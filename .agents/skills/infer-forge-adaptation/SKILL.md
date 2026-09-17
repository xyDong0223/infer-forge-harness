---
name: infer-forge-adaptation
description: Start or continue an Infer-Forge model adaptation run through the durable harness, including operator work and failure decisions. Use for executing an adaptation, not ordinary repository development or architecture review.
---

# Infer-Forge adaptation

Use Codex as the main controller and the existing harness as the authority for
tasks, claims, evidence, and delivery. This skill adds no worker service or
alternate state machine.

## Recover the run before acting

Read the repository [agent protocol](../../../AGENTS.md) and the
[interaction protocol](../../../docs/migration/codex-interaction.zh-CN.md).
Resolve the state database and run ID from the user's request or configured
state location. Query first:

```bash
python3 cli/adaptation.py --state /external/adaptation.db context --run-id RUN
```

The context is a read-only projection, not a claim or fresh evidence validation.
Inspect identity, evidence mode, environment, task progress, and saved graph
execution context. A missing run/configuration is a missing input: do not create
a replacement run or reconstruct settings from an old chat. For an explicitly
new adaptation, collect the model/plugin revisions, backend, external state and
artifact locations, and required environment inputs; use the existing create
and graph setup described in the interaction protocol.

Repository changes, a plan, or a simulation request do not authorize real cluster
or XPU execution. For an authorized real adaptation, require the user's resource
owner ID before environment provisioning; do not derive it from login names or
old resource names. Keep simulation evidence separate from real readiness.

## Advance to a meaningful boundary

Use `advance --run-id RUN` with the same state database after checking its saved
configuration and resolving any uncertain execution/validation boundary below.
Independent eligible worker tasks may still proceed on disjoint resources.
Consume the returned boundary and referenced artifacts rather
than repeatedly issuing the command:

- Missing inputs or a blocked outcome: report the exact missing input/evidence
  or unsupported action. Do not spin, invent values, or weaken a gate.
- Worker work: inspect task context, then dispatch a scoped assignment as below.
- A decision handoff: read the request, failed attempt, original error, input
  references, allowed actions, and remaining budget. Select a conclusion from
  evidence, delegating bounded diagnosis where helpful. Submit the existing
  Decision format through `submit-decision`, using the returned handoff identity
  and expected version; see the interaction protocol for the envelope.
- Completion: inspect the persisted delivery receipt and its evidence. A zero
  exit code, a submitted decision, or an operator finishing is not delivery.

Keep the same decision ID and identical payload when retrying an uncertain
submission. Inspect the recorded receipt first; stale/conflicting decisions need
fresh context, not a new ID to force execution. An accepted decision with unknown
execution state is a stop-and-investigate boundary. Do not run the headless
automatic recovery loop alongside this interactive controller or wait in a
shell for a decision that this same Codex session must provide.

## Dispatch operator work

Use the project roles `infer-forge-implementer`, `infer-forge-diagnoser`, and
`infer-forge-validator` when supported, or equivalent bounded subagent prompts.
They inherit the user's model and permissions. Do not create a role for every
graph node or assume that a subagent has a separate worktree.

Before an assignment, read `context --run-id RUN --task-id TASK`. For
`worker_protocol: managed-v2`, read the [managed worker protocol](../../../docs/migration/managed-worker.zh-CN.md)
before freezing, validating, or recovering. Its real Pod/device drivers are not
yet supported: that is a blocker, not permission to downgrade the protocol.

Claim with both
`--run-id RUN --task-id TASK`, the matching stage, and a worker identity that
identifies the actual lease holder. For legacy results preserve the producer's
original worker identity when submitting on their behalf. Managed runs separately
record controller, producer, and validator; do not relabel one as another.
Explicitly name who renews/submits. Only the
current claim token authorizes submission; the frozen `input/task.json` does not
contain it. Keep the token with the designated holder.

Give the child the full claim-time task packet, relevant upstream evidence,
authorized source paths, and current attempt workspace. Read the selected
`input/skill.json` method snapshot when supplied. Otherwise consult
[the existing method catalog](../../../catalog/skill_catalog.yaml) and read only
the applicable method under [skills](../../../skills/README.md); do not duplicate
those methods here. Before choosing an implementation path, read the relevant
`openwiki/vllm-core/`, `openwiki/vllm-kunlun/`, and `openwiki/harness/` references.

Implementers write candidates and focused tests. A separate validator must
actually execute the required checks against the fixed candidate and independent
reference, recording commands, candidate identity, environment, logs, and results
in its assigned current-attempt output directory. A different validator name or
another conversation is not evidence of independent testing. Do not invent a
validator scheduler stage: validation is part of the claimed stage's result.
Managed-v2 uses `freeze-candidate` and `validate-worker`; only the persisted Runner
receipt supports `complete`. Do not author a substitute PASS receipt.

Use the existing `complete`, `fail`, and diagnosis commands with the current
lease. Consume persisted results and evidence, not only child summaries. On a
missing semantic contract, failed test, or lease loss, preserve evidence and
return the appropriate failure/blocked outcome rather than manufacture success.

## Resource and restart limits

One controller owns a run. Independent source work and CPU investigations may
proceed in parallel when write paths are disjoint. The controller must serialize
shared Pod changes, candidate installation, device tests, toy probes, and service
regression. Multiple task leases do not grant concurrent Pod mutation rights.
Legacy/P1 execution has no automatic resource/process guarantees. Managed-v2
adds same-database occupancy and local execution supervision, not a sandbox or a
complete remote Pod driver. Read the supported boundary before using it.

Before re-claiming after interruption, inspect accepted results and whether the
old local/remote execution is still running. Lease expiry is not process death.
If execution ownership is uncertain, retain the Pod and evidence and stop new
mutations to the affected task/attempt and shared resource until resolved. This
does not stop independent eligible source/CPU tasks on disjoint paths/resources.
Re-claims use fresh attempts; never overwrite an older
attempt or recreate the proven Pod simply to retry.
For managed runs inspect `execution-status`; `reconcile-execution` only confirms
known local process-group absence, never force-unlocks unknown remote work.

Report the run/task IDs, actual verdict, durable evidence paths, and next action.
Keep `SIMULATION_PASS` explicitly distinct from hardware readiness.
