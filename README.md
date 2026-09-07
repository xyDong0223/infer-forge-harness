# kunlun-inference-agent

An agentic, contract-driven inference engineering platform for vLLM-Kunlun model adaptation and performance optimization.

## Design principle

> Workflow orchestrates, Task defines acceptance, Skill provides engineering method, Tool performs deterministic actions, Runner executes, Adapter isolates platform differences, Validator decides, and Catalog stores capability facts.

The repository separates reusable engineering rules from runtime state. Task contracts, schemas, workflows, skills, tools, adapters, validators, tests, and documentation are versioned here. Logs, traces, benchmark outputs, model caches, Pod state, and temporary worktrees belong in an external artifact root.

## v0.1 scope

The first milestone is **KDP-001 Kunlun Deployment Proof**: from a fixed Deployment Manifest, prepare a Kunlun P800 environment, start a vLLM-Kunlun service, verify readiness and a chat completion, and produce a reproducible artifact manifest. The initial repository also defines the first model-adaptation workflow stages: model intake and static model scan.

The runner is intentionally conservative. Plan-only mode is the default; cluster-side execution requires explicit authorization and an environment-specific adapter.

## Repository map

| Directory | Responsibility |
| --- | --- |
| `contracts/` | Stable machine-readable schemas |
| `workflows/` | Business orchestration and Task Graphs |
| `tasks/` | Verifiable task contracts and task-local guidance |
| `skills/` | Domain methods, rules, and human engineering knowledge |
| `tools/` | Deterministic action interfaces |
| `runners/` | Execution, state machine, and artifact coordination |
| `adapters/` | Kubernetes, Kunlun P800, and vLLM-Kunlun differences |
| `validators/` | Independent acceptance gates |
| `catalog/` | Model, backend, and support facts |
| `tests/` | Unit, contract, fake-runner, and P800 integration tests |
| `docs/` | Architecture and contribution guidance |

## Planned execution path

```text
Task Contract
  -> Deployment Manifest
  -> Runner
  -> Tool / Adapter
  -> Validator
  -> Artifact Manifest
  -> PASS / REWORK / NO_GO / NEEDS_HUMAN
```

## Development

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python -m compileall runners validators tools
```

The P800 integration suite is opt-in and must never be run against a shared cluster without an explicit environment configuration.

## OpenWiki reference

The [`openwiki/`](openwiki/) directory contains the OpenWiki reference copied from the vLLM-Kunlun `v0.25.1-dev` branch. Its provenance and source revision are recorded in [`openwiki/SOURCE.md`](openwiki/SOURCE.md). Use it as runtime and architecture reference material; platform contracts and executable Task definitions in this repository remain the source of truth for the Agent platform.

## Status

This is an initial public scaffold. Runtime adapters and real P800 execution are intentionally added incrementally behind contracts, fake adapters, and independent validators.
