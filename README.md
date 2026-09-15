<p align="center">
  <img src="assets/readme-hero.png" alt="Hand-drawn engineering flow from a task contract through a compute runner and validation to a reproducible artifact" width="100%">
</p>

# infer-forge-harness

> **An agentic harness for bringing up, validating, and optimizing inference models across heterogeneous accelerators.**

**Python 3.10+** · **Real-cluster execution behind explicit authorization** · **416 unit tests green** (scheduler and run-adaptation suites deselected pending pre-existing fixes)

`infer-forge-harness` turns inference-engineering work into explicit, verifiable loops. It keeps reusable engineering rules in version control, separates them from runtime state, and requires independent validation before an outcome becomes a reusable fact. Kunlun P800 with vLLM-Kunlun is the first target stack. Three model bring-ups have run through it on real clusters: Qwen3-8B is `validated` in the support matrix with full evidence; GLM5.2-Int-W8A8's service proof reached `DEPLOYMENT_READY` (accuracy differential still open); MiniMax-M2.5/M3 bring-ups completed, with M3's evidence archived.

## Contents

- [Quick start](#quick-start)
- [What it solves](#what-it-solves)
- [Design model](#design-model)
- [Execution path](#execution-path)
- [MiniMax-M3 real adaptation loop](#minimax-m3-real-adaptation-loop)
- [Execution model](#execution-model)
- [Operator integration loop](#operator-integration-loop)
- [Repository map](#repository-map)
- [Run the checks](#run-the-checks)
- [Reference material](#reference-material)
- [Contributing](#contributing)
- [Status](#status)

## Quick start

Use Python 3.10 or later. Local checks resolve contracts and validate artifacts without creating or changing cluster resources — plan-only is what you get by *not* passing `--execute`, not a limitation of the scaffold. Real-cluster execution (pod creation, service bring-up, in-pod probes) is wired and E2E-proven; it requires an authorized Kubernetes context, a reachable cluster, the model volume and revision, and a vLLM-Kunlun image. Those environment-specific prerequisites are intentionally not created by this repository.

```bash
git clone https://github.com/xyDong0223/infer-forge-harness.git
cd infer-forge-harness

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pyyaml
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

python -m unittest discover -s tests -p 'test_*.py' -v
python -m compileall runners validators tools
```

The full unit suite uses **PyYAML** to load contracts and **PyTorch** for CPU reference arithmetic. A real P800 run additionally requires an authorized Kubernetes context, a reachable cluster, the model volume and revision, and a vLLM-Kunlun image. Those environment-specific prerequisites are intentionally not created by this repository.

## What it solves

Inference model adaptation needs more than a sequence of scripts. A reliable workflow must declare the expected input, select a constrained method, record environment-specific facts, execute only authorized platform actions, and prove that the result satisfies an independent acceptance gate.

**KDP-001 — Kunlun Deployment Proof** was the first vertical slice and is now fully equipped: from a fixed Deployment Manifest, the platform prepares a Kunlun P800 environment, installs and drift-checks the vLLM-Kunlun stack, starts a service, verifies readiness and a chat completion, and produces a reproducible artifact manifest. On top of it stand the complete model-adaptation chain (intake, scan, capability match, gap classification, evaluation, toy bring-up, shim handoff, accuracy differential, support matrix) and the operator-integration loop. See [docs/architecture.md](docs/architecture.md) for the target axes (Runtime / Hardware / Capability) and the component ownership map.

Cluster-side execution is intentionally not implicit. The runner stays in plan mode unless `--execute` is passed; an environment-specific Adapter and explicit authorization are required before actions reach a cluster.

## Design model

> **Workflow orchestrates, Task defines acceptance, Skill provides engineering method, Tool performs deterministic actions, Runner executes, Adapter isolates platform differences, Validator decides, and Catalog stores capability facts.**

| Building block | Responsibility |
| --- | --- |
| **Workflow** | Orchestrates business stages and cross-task routing through the Task Graph. |
| **Task** | Defines the verifiable contract, acceptance criteria, and task-local guidance. |
| **Skill** | Encodes the engineering method, including preconditions, verification, exit conditions, and learned rules. |
| **Tool** | Performs a deterministic action with a constrained interface. |
| **Runner** | Coordinates the state machine, task memory, artifacts, and the next decision. |
| **Adapter** | Isolates Kubernetes and Kunlun P800 platform differences (safety-gated writes, pod exec, file push). |
| **Validator** | Applies an independent acceptance gate and determines the verdict. |
| **Catalog** | Stores model, runtime/hardware, tool, and support capability facts. |

The building blocks are *what* something is; three orthogonal axes — **Runtime** (vllm-kunlun today, sglang-kunlun planned), **Hardware** (kunlun-p800), and **Capability** (model adaptation today; performance and memory analysis planned) — describe *what it is about*. `docs/architecture.md` carries the full ownership map and the refactor plan that makes the axes explicit.

The repository versions contracts, schemas, workflows, skills, tools, adapters, validators, tests, and documentation. Logs, traces, benchmark outputs, model caches, Pod state, and temporary worktrees belong in an external artifact root.

## Execution path

<p align="center">
  <a href="docs/assets/agent-workflow.excalidraw">
    <img src="docs/assets/agent-workflow.png" alt="Hand-drawn workflow showing task contract through deployment manifest, graph runner, tool or adapter, validator, artifact manifest, and final PASS, REWORK, NO_GO, or NEEDS_HUMAN verdict" width="100%">
  </a>
</p>

The rendered diagram is backed by an editable [Excalidraw source](docs/assets/agent-workflow.excalidraw). It shows the main contract-to-verdict path, the persistent Task Memory loop, and the explicit outcomes that prevent unverified progress from becoming a capability fact.

### Three gates before anything expensive

A registry entry proves a name is mapped, not that the code behind it loads, and loading it is not the same as running it. Both gaps used to be closed by launching the model, which on a 744B checkpoint means a 700 GiB read per error message.

- **MAT-027 Runtime Drift** imports every module of the installed plugin against the installed engine in one pass, and indexes the engine's own source to say where each missing symbol lives now. No model, no weights, no XPU. A version label is not evidence of a matching install: vLLM-Kunlun `v0.25.1-dev` pairs with a wheel labelled `0.25.1` whose internals are months newer, and every MLA module failed to import while the label matched.
- **MAT-028 Toy Bring-up** derives a few-layer copy of the real config, points the engine at dummy weights, and runs prefill and one decode step. Depth, expert count and MTP shrink; every dimension that selects a kernel stays at real size, so it is the same code path. It catches what sits between "imports" and "serves" — an abstract method the engine now requires, a factory whose return shape changed, a KV-cache tensor whose rank the layer slices wrongly — and it cannot see a wrong number, because the weights are random.

### The gate that turns torch shims into operator requests

A torch shim standing in for a vendor kernel is the right way to keep bring-up moving — and a silent way to ship un-optimised arithmetic forever. GLM-5.2's adaptation produced three of them; post-hoc inspection showed only one (`kv_spans_from_batches`) was actually on a call path — the other two were dead on arrival because the live paths already called the vendor kernels through `torch.ops.xspeedgate_ops` — but nothing in the loop could have told the difference. The dispatch path (MAT-024) was fed only by the static gap classification, and nothing connected the place shims are born to it, so all three sat unregistered and unexamined.

**MAT-029 Shim Handoff** re-enters the graph from every fix edge, nets shim candidates out of the installed plugin (the `kunlun_` prefix, docstrings that admit to replacing a triton/CUDA kernel), refuses any signal the shim registry cannot explain, and immediately dispatches one durable operator request per non-waived shim through the same `operator_lifecycle` requests MAT-026 integrates. Its first question to the adapter is the one GLM-5.2 never got asked: is this shim even wired in? A waiver is allowed — but it needs a reason someone can audit; "nobody got to it" is not one.

### MiniMax-M3 real adaptation loop

<p align="center">
  <a href="docs/assets/minimax-m3-adaptation-loop.excalidraw">
    <img src="docs/assets/minimax-m3-adaptation-loop.png" alt="Hand-drawn MiniMax-M3 P800 adaptation loop showing intake, P800 proof, capability evaluation fan-out, a service-to-golden-reference diagnosis loop, baseline freeze, candidate integration gates, rollback, and delivery" width="100%">
  </a>
</p>

This is the real MiniMax-M3 engineering loop: a static runtime match is only the start; each high-risk dimension is exercised against a CPU float32 reference with relative L2 and a discriminating negative control. A service mismatch enters a trace-to-golden-reference loop before patch placement sends the candidate back through service proof.

The editable [Excalidraw source](docs/assets/minimax-m3-adaptation-loop.excalidraw) distinguishes unit-level evidence from served-model accuracy. Only an accurate service can freeze a baseline. Every generated candidate must then pass kernel, dispatch, service-regression, and accuracy-regression gates; a failure is explicitly rejected and rolled back before the next candidate is evaluated.

## Execution model

The graph runner keeps the Task Graph for cross-task routing and stores the current and completed Loop Blocks in Task Memory. A successful fact can be reused only when its Journal environment fingerprint matches the current execution context.

### A task lifecycle

<p align="center">
  <img src="docs/assets/task-lifecycle-sequence.png" alt="Sequence diagram showing the Runner, Journal, Tool or Adapter, Validator, and Artifact root exchanging inputs, authorized actions, evidence, validation results, and reusable facts" width="100%">
</p>

Each task is a bounded evidence loop. The Runner selects the contract and method, the Tool or Adapter performs the authorized action, the Validator applies an independent acceptance gate, and the Journal receives a reusable fact only after that gate passes.

```bash
python3 runners/graph_runner.py \
  --subject Qwen3-8B \
  --artifact-root /path/to/artifacts \
  --journal /path/to/journal.jsonl \
  --env hardware=P800 \
  --env stack_commit=<commit> \
  --resume --json
```

`--resume` skips nodes whose successful artifact and state file are still available. `--json` emits a compact result for each decision with `status`, `reason_code`, `next_task`, and artifact paths. Task Memory is written to `<artifact-root>/task_memory.json` by default and can be overridden with `--loop-state`.

The tool capability index in [`catalog/tool_catalog.yaml`](catalog/tool_catalog.yaml) is the first lookup for an Agent choosing a deterministic tool. Task-specific contracts and validators remain authoritative for inputs and acceptance. The Skill registry in [`catalog/skill_catalog.yaml`](catalog/skill_catalog.yaml) maps every workflow `task_type` to a method unit with its preconditions, tools, verification, exit conditions, and prior P800 adaptation rules. The graph runner records the selected Skill in Task Memory and JSON summaries.

When a Task exposes a more specific fact, the resolver selects the narrower method automatically:

```bash
--set issue=cache_layout       # cache-layout-validation
--set dimension=quantization   # quantization-differential
--set issue=runtime_state      # runtime-state-triage
--set issue=p800_kernel        # p800-fallback-selection
```

The generic Skill remains the fallback when no specialized fact is present. Specialized Skills are activated by observed context, not by a model name or an unverified guess.

### Autonomous failure recovery

A new model fails by default, and the interesting part of bring-up is what
happens next. `--auto-recover` replaces the "stop and wait for a person" step
with a bounded decide/act/rerun loop:

```bash
python3 runners/graph_runner.py \
  --subject Qwen3-8B --artifact-root /path/to/artifacts \
  --execute --auto-recover --brain agent
```

When a node fails, the runner packages the failure evidence, the repair
history, the remaining budget, and the node's current context into a
`decision_request.json`. A decider — an LLM agent session, a model API behind
`--decide-command`, or anything else that can write JSON — answers with a
`decision.json` naming one of `RETRY`, `RETRY_WITH_PARAMS`, `RUN_TRIAGE`,
`PLACE_PATCH`, `DISPATCH_OPERATOR_TASK`, `REDISCOVER`, `ROLLBACK`, or
`BLOCKED`. The controller executes the action, reruns the node, and only the
node's own validator can declare the failure gone. Every failed node gets
`--recovery-budget` attempts (default 3); a malformed decider answer is
re-asked once and then degrades to `BLOCKED`, never to a guessed action.
`--brain rule` swaps in a deterministic classifier for environments without a
decider. The triage and placement steps the loop can invoke are sequenced
executors (`runners/triage_executor.py`, `runners/patch_executor.py`), not
prompts: mat-006's instrument/capture/isolate/restore sequence and mat-007's
apply/validate/compare/reject sequence run as written, and the independent
validators still decide their verdicts.

The three correctness gates are sequenced the same way
(`runners/correctness_executor.py`): mat-021 grades a platform kernel against
an independent CPU reference through `tools/tensor_diff.py`, mat-022 packages
the integrated-serving-path differential into case-level evidence, and mat-023
exercises sparse selection beyond `block_size * topk` with a shifted-block
control. In all three, a control that cannot fail makes the verdict AMBIGUOUS,
never a pass. With these, every task type in the workflow has an executor and
none stops for a person.

## Operator integration loop

`model_adaptation` dispatches confirmed `CAPABILITY_MISSING` gaps to durable `xpu-op-gen` requests and continues model bring-up. After service and independent accuracy both pass, it freezes a baseline. Generated candidates are then tested one at a time against that baseline. A failed kernel, dispatch, service, or accuracy gate is rejected and must be rolled back before the next candidate is considered.

The GLM-5.2 candidate integration recorded what the four failed swaps before a green one taught, now checks in MAT-026's contract: build the candidate **in the target environment** (a host-built wheel failed on glibc), from the **commit the operator team ships** (local HEAD had silently diverged and lost eight operators), enumerate **every operator the plugin references** against the new package, **restore side-car modules** the old wheel owned (`cocopod` vanished with the uninstall), and reconcile the **version metadata** a rebuild without git metadata breaks. The verification ladder is fixed: exact numeric equality against the shim on the target device, then worker logs proving every rank took the new path, then service health and a real completion.

The lifecycle tool can also be driven directly:

```bash
python3 tools/operator_lifecycle.py integrate \
  --baseline /path/to/baseline_manifest.json \
  --candidate /path/to/candidate_manifest.json \
  --subject Qwen3-8B \
  --out /path/to/integration
```

The candidate manifest must reference independent reports for `KERNEL_PASS`, `DISPATCH_CONFIRMED`, service regression, and accuracy regression. The tool does not mutate the running service; integration and rollback remain explicit Adapter actions.

## Repository map

Cluster and deployment resource profiles are kept separately under
[`config/clusters/`](config/clusters/). Add one profile per target cluster;
keep kubeconfig contents and credentials outside the repository.

| Directory | Responsibility |
| --- | --- |
| [`config/`](config/) | Harness defaults, cluster resource profiles, and deployment manifests. |
| [`contracts/`](contracts/) | Stable, machine-readable schemas. |
| [`workflows/`](workflows/) | Business orchestration and Task Graphs (model adaptation is fully equipped; performance and test-release are stubs pending their capability work). |
| [`tasks/`](tasks/) | Verifiable task contracts, per-model instances, and manifests. |
| [`skills/`](skills/) | Domain methods, rules, and human engineering knowledge. |
| [`engine/`](engine/) | Core: the adaptation-run scheduler, operator contracts, discovery, and recovery loop — platform-neutral. |
| [`tools/`](tools/) | Deterministic action interfaces, pod-side probes (`probe/`), and replayable runtime patches (`patches/`). |
| [`runners/`](runners/) | Execution, state-machine, and artifact coordination (graph runner, deployment proof, triage/patch/correctness executors, evidence and watch). |
| [`adapters/`](adapters/) | Kubernetes and Kunlun P800 platform differences. |
| [`validators/`](validators/) | Independent acceptance gates. |
| [`catalog/`](catalog/) | Model, runtime/hardware, tool, and support facts. |
| [`openwiki/`](openwiki/) | Layered references, including capability-axis experience homes (`openwiki/harness/experiences/`). |
| [`archive/`](archive/) | Frozen artifacts from completed adaptation eras (MiniMax-M3). |
| [`tests/`](tests/) | Unit, contract, fake-runner, failure-edge, integration, and path-resolution guard tests (`test_common.py`). |
| [`docs/`](docs/) | Architecture, contribution guidance, and visual documentation assets. |

## Run the checks

Run the documented local validation commands before proposing a change:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python -m compileall runners validators tools
```

The P800 integration suite is opt-in. It must never run against a shared cluster without explicit environment configuration, including the namespace, image digest, model revision, hardware, and cleanup policy.

## Reference material

The [`openwiki/`](openwiki/) directory is a layered reference base: [`vllm-core/`](openwiki/vllm-core/) documents the upstream vLLM hardware-backend integration contract, [`vllm-kunlun/`](openwiki/vllm-kunlun/) records the Kunlun P800 plugin implementation, and [`harness/`](openwiki/harness/) contains Infer-Forge engineering practice. Their separate provenance and evidence baselines are registered in [`openwiki/SOURCE.md`](openwiki/SOURCE.md). Use these as runtime and architecture reference material; platform contracts and executable Task definitions in this repository remain the source of truth for the Agent platform.

## Contributing

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull request. Changes should identify their affected layer, include the relevant validation evidence, and keep runtime artifacts, model weights, tokens, private endpoints, raw production traffic, and large traces outside the repository.

## Status

The harness runs real adaptation end to end on the P800 + vLLM-Kunlun target stack: three model bring-ups have produced durable evidence, and the current refactor track ([docs/architecture.md](docs/architecture.md), target axes) is separating the Runtime / Hardware / Capability axes so that SGLang-Kunlun support, performance, and memory-analysis capabilities can be added without touching the core.

It does **not** ship model weights, provide a hosted inference endpoint, or create a turnkey multi-node production deployment. Runtime artifacts, model caches, credentials, private endpoints, raw production traffic, and large traces remain outside the repository.
