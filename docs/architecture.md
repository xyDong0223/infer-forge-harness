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

## Target axes (Runtime / Hardware / Capability)

The layer list above describes *building blocks*. Three orthogonal axes
describe *what a piece of code is about*, and every component must be able to
answer which axis it belongs to:

| Axis | Meaning | Today | Planned |
| --- | --- | --- | --- |
| **Runtime** | The inference framework stack a service runs on. | `vllm-kunlun` (the only wired one; `config/clusters/b200-cluster.yaml` declares `runtime.engine: sglang` but no adapter exists for it) | `sglang-kunlun` and others, one entry per stack under `runtimes/` |
| **Hardware** | The accelerator and cluster a target runs on. | `kunlun-p800` (wired); `b200` (cluster profile template only, no adapter) | other XPU parts and clusters |
| **Capability** | The engineering capability being exercised. | model adaptation (the only fully equipped vertical) | performance, memory analysis, release testing — one workflow each |

A run's **TargetContext** is the tuple that pins all three plus the subject:

```text
TargetContext = {
    model,          # e.g. GLM5.2-Int-W8A8 (from the task instance)
    runtime,        # e.g. vllm-kunlun      (from the runtime profile)
    hardware,       # e.g. kunlun-p800      (from the cluster config)
    deployment,     # launch parameters, manifest, revisions
    artifact_root,  # where evidence for this run lives
}
```

Today this tuple exists but is scattered: the model comes from the task
instance, hardware-ish values are read from the cluster config (through a
field confusingly named `--backend`), and the runtime is implicit — 19
hardcoded `/opt/vllm_kunlun` strings in Python (`runners/` 8 + `tools/` 11;
2 more inside `tools/install_vllm_kunlun.sh`), a serve command assembled
inside `runners/task_runner.py`, and 17 direct imports of the concrete
`KunlunP800Adapter` (runners 5 + tools 12). The refactor plan (phases 1-2)
makes the axes explicit: a runtime registry, a hardware split out of the
cluster adapter, and a compatibility check that fails unsupported
combinations before any work starts.

### Component ownership map

Every component, classified by the axis it *should* carry (components that
import the concrete `KunlunP800Adapter` carry a Hardware leak by that fact
alone, whatever else they mix in):

| Component | Class | Notes |
| --- | --- | --- |
| `engine/` (scheduler, contracts, discovery, recovery, brain, fake_agents) | Core | verified zero platform leakage (no vllm/kunlun/p800/kubectl references) |
| `runners/graph_runner.py` | Core | one leak: reads `environment.get("hardware", "p800")` into a field named `backend` |
| `runners/evidence.py`, `runners/watch.py`, `tools/journal.py`, `tools/task_memory.py` | Core | crash-first snapshots, heartbeats, fingerprint-scoped facts |
| `runners/task_runner.py` | Core + Runtime + Hardware | renders contracts; assembles the `python -m vllm.entrypoints.openai.api_server` serve command itself; imports `KunlunP800Adapter` (task_runner.py:57) |
| `runners/deployment_proof.py` | Core + Runtime + Hardware | install/replay/precheck/readiness; the largest mixed site (876 lines); imports `KunlunP800Adapter` (deployment_proof.py:19) |
| `runners/{triage,patch,correctness}_executor.py` | Core + Runtime + Hardware | one direct `KunlunP800Adapter` import each (3 sites) |
| `adapters/kunlun_p800/` | Hardware + Cluster | kubectl primitives, safety gates, `xpu_smi` device probes — cluster and hardware still fused; split planned |
| `tools/` (task CLIs + `probe/`) | Core + Runtime + Hardware | 8 CLIs embed the venv prefix string; 12 import `KunlunP800Adapter` directly; probes are mostly stack-aware by nature |
| `tools/patches/` | Runtime | repair content for the vllm-kunlun stack — stack-specific on purpose |
| `validators/` | Core (5 files carry platform concepts) | functional: `deployment_validator.py` (`expected_backend: kunlun`); concept-level: intake, scan, handoff, triage |
| `tasks/`, `workflows/`, `skills/`, `catalog/`, `contracts/`, `config/clusters/` | Capability + declarative | catalog already separates device facts (`xpu_specs`) from support facts (`support_matrix`); `b200-cluster.yaml` is a second-cluster template with no adapter wired |
| `openwiki/harness/experiences/` | Capability | capability-axis experience homes (attention-mla, moe, ...) |

### Canonical naming

Names are compared by string, so one spelling per concept:

| Concept | Canonical | Deprecated aliases (still in tree) |
| --- | --- | --- |
| Runtime | `vllm-kunlun` | `sglang` (only inside the unwired `b200-cluster.yaml` template's `runtime.engine`) |
| Hardware | `kunlun-p800` | `Kunlunxin-3-P800` (display name: 22 task.yaml files, 3 instance yamls, `task_runner.py:84`, `support_matrix.yaml`, `model_intake.py:169` and `update_support_matrix.py:83` default args), `p800` (graph_runner default, `memory_budget.py:328` `--device` default), `kunlun` (hardware wearing a runtime-sounding name: `checks.backend.expected`, `deployment_validator.py:17`), `Kunlunxin-3 P800` (spaced, cluster yaml description) |

The CLI flag `--backend` currently carries *hardware* values; splitting it
into `--runtime` / `--hardware` lands with the phase-2 compatibility check,
together with alias cleanup — the alias column above is the cleanup ledger.
Until then, new code uses the canonical spellings only.

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
