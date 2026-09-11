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

## Environment gate

Runtime investigation starts only after the deployment environment proof has
passed. The proof owns one prepared Pod and records four facts: the Pod is
Ready, the installed stack imports, the pinned vLLM-Kunlun code worktree is
present, and the target XPU devices are visible. Model scans, toy bring-up,
shim discovery, and operator tasks reuse that Pod and its fingerprint. A run
created through `tools/run_adaptation.py` remains `WAITING_FOR_ENVIRONMENT`
until this proof is bound; discovery is rejected before that transition.

## First vertical slice

KDP-001 is the first vertical slice. It must reach `DEPLOYMENT_READY` only after Pod readiness, `/health`, Chat API, expected Kunlun backend, no unexpected fallback, and evidence collection all pass. A Kubernetes `Running` state alone is never sufficient.

## Capability promotion

New Skills enter the catalog only after a Golden Task and an independent Validator pass. Failures are classified before a Skill is changed. Contract and status semantics require migration notes and backward-compatibility tests.
