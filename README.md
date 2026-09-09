# kunlun-inference-agent

An agentic, contract-driven inference engineering platform for vLLM-Kunlun model adaptation and performance optimization.

## Design principle

> Workflow orchestrates, Task defines acceptance, Skill provides engineering method, Tool performs deterministic actions, Runner executes, Adapter isolates platform differences, Validator decides, and Catalog stores capability facts.

The repository separates reusable engineering rules from runtime state. Task contracts, schemas, workflows, skills, tools, adapters, validators, tests, and documentation are versioned here. Logs, traces, benchmark outputs, model caches, Pod state, and temporary worktrees belong in an external artifact root.

## v0.1 scope

The first milestone is **KDP-001 Kunlun Deployment Proof**: from a fixed Deployment Manifest, prepare a Kunlun P800 environment, start a vLLM-Kunlun service, verify readiness and a chat completion, and produce a reproducible artifact manifest. The initial repository also defines the first model-adaptation workflow stages: model intake and static model scan.

The runner is intentionally conservative. Plan-only mode is the default; cluster-side execution requires explicit authorization and an environment-specific adapter.

## Agent execution

The graph runner keeps the Task Graph for cross-task routing and stores the
current and completed Loop Blocks in Task Memory. Successful facts can be reused
only when their Journal environment fingerprint matches:

```bash
python3 runners/graph_runner.py \
  --subject Qwen3-8B \
  --artifact-root /path/to/artifacts \
  --journal /path/to/journal.jsonl \
  --env hardware=P800 \
  --env stack_commit=<commit> \
  --resume --json
```

`--resume` skips nodes whose successful artifact and state file are still
available. `--json` emits a compact result for each decision with `status`,
`reason_code`, `next_task`, and artifact paths. Task Memory is written to
`<artifact-root>/task_memory.json` by default and can be overridden with
`--loop-state`.

The tool capability index is in `catalog/tool_catalog.yaml`. It is the first
lookup for an Agent choosing a deterministic tool; task-specific contracts and
validators remain authoritative for inputs and acceptance.

The Skill registry is in `catalog/skill_catalog.yaml`. It maps every workflow
`task_type` to one method unit with preconditions, tools, verification, exit
conditions, and rules learned from prior P800 adaptations. `graph_runner` records
the selected Skill in Task Memory and JSON summaries.

When a Task exposes a more specific fact, the resolver selects a narrower
method automatically. For example:

```bash
--set issue=cache_layout       # cache-layout-validation
--set dimension=quantization   # quantization-differential
--set issue=runtime_state      # runtime-state-triage
--set issue=p800_kernel        # p800-fallback-selection
```

The generic Skill remains the fallback when no specialized fact is present.
Specialized Skills are deliberately activated by observed context, not by a
model name or an unverified guess.

### Asynchronous operator loop

`model_adaptation` dispatches confirmed `CAPABILITY_MISSING` gaps to durable
`xpu-op-gen` requests and continues model bring-up. After service and independent
accuracy both pass, it freezes a baseline. Generated candidates are then tested
one at a time against that baseline; a failed kernel, dispatch, service, or
accuracy gate is rejected and must be rolled back before the next candidate.

The lifecycle tool can also be driven directly:

```bash
python3 tools/operator_lifecycle.py integrate \
  --baseline /path/to/baseline_manifest.json \
  --candidate /path/to/candidate_manifest.json \
  --subject Qwen3-8B \
  --out /path/to/integration
```

The candidate manifest must reference independent reports for `KERNEL_PASS`,
`DISPATCH_CONFIRMED`, service regression, and accuracy regression. The tool
does not mutate the running service; integration and rollback remain explicit
adapter actions.

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
