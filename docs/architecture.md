# Architecture

## Layers

The platform has explicit boundaries:

- **Workflow** describes orchestration and dependencies. It must not contain shell commands.
- **Task** describes one verifiable engineering objective, its context, acceptance gates, artifacts, and terminal states.
- **Skill** describes reusable engineering method and decision rules.
- **Tool** performs one deterministic action and returns structured evidence.
- **Runner** executes a Task and coordinates state, retries, and artifacts.
- **Adapter** isolates Kubernetes, Kunlun P800, and vLLM-Kunlun runtime differences.
- **Validator** independently decides whether evidence satisfies acceptance.
- **Catalog** stores facts about models, backends, and support status.

## Runtime boundary

Repository contents are versioned source of truth. Runtime state, logs, traces, model caches, Pod descriptions, benchmark raw data, and temporary worktrees belong in an external artifact root. A final Artifact Manifest may be committed or attached to a Draft PR, but large raw files should remain outside Git.

## First vertical slice

KDP-001 is the first vertical slice. It must reach `DEPLOYMENT_READY` only after Pod readiness, `/health`, Chat API, expected Kunlun backend, no unexpected fallback, and evidence collection all pass. A Kubernetes `Running` state alone is never sufficient.

## Capability promotion

New Skills enter the catalog only after a Golden Task and an independent Validator pass. Failures are classified before a Skill is changed. Contract and status semantics require migration notes and backward-compatibility tests.
