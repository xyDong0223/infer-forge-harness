# Infer-Forge Agent Protocol

This file is the execution protocol for any Agent working in this repository.
Read it before changing code, running an adaptation, or reporting a result. The
repository is a durable orchestration harness: an Agent must use its task
contracts and scheduler instead of treating the project as a collection of
unrelated scripts.

## Mission and source of truth

The objective is to adapt an inference model to a vLLM backend and target XPU
through a reproducible, evidence-backed loop:

```text
deployment environment proof (prepared Pod + code + XPU)
  -> model identity intake
  -> runtime/toy bring-up and scan inside that Pod
  -> missing-operator discovery
  -> OperatorSpec
  -> PyTorch reference task
  -> independent reference validation
  -> XPU implementation task
  -> device correctness and dispatch validation
  -> integration and service regression
  -> functional-ready
  -> optional benchmark and profiling optimization
```

The Task contracts, validators, and actual execution evidence are authoritative.
The `openwiki/` material is engineering reference material and may explain why
an approach is appropriate, but it does not override a task contract or prove
that a result passed.

## Roles

### Main Agent

The Main Agent owns one `AdaptationRun` and its decisions. It must:

1. Create or resume the run in the scheduler.
2. Run the deployment environment-proof Task before runtime investigation. No
   model scan, capability evaluation, shim scan, or operator adaptation may
   start until its status is bound to the run.
3. Read the relevant `openwiki/vllm-core/`, `openwiki/vllm-kunlun/`, and
   `openwiki/harness/` material before choosing an implementation path.
4. Run intake, static/runtime checks, and toy bring-up before declaring a model
   usable.
5. Convert every confirmed gap into an `OperatorSpec` and a durable task.
6. Keep independent operator branches moving in parallel; do not wait on one
   failed operator before investigating other gaps.
7. Dispatch work to the appropriate child Agent and consume only its persisted
   result and evidence.
8. Use diagnosis conclusions to decide whether to rediscover, repair, retry,
   fall back explicitly, or mark a branch blocked.
9. Declare functional completion only after integration and service regression
   have passed for the required operators.

The Main Agent may change repository code when the task requires it, but it
must still record the affected task, evidence, and validation outcome.

### PyTorch Agent

The PyTorch Agent receives an `OperatorSpec` and the failure/call evidence that
produced it. It produces a reference implementation, tests, and a candidate
manifest. It must not invent tensor shapes, dtypes, layouts, or semantics that
are absent from evidence; uncertainty is a diagnosis or blocked outcome.

### XPU Agent

The XPU Agent receives the same `OperatorSpec` plus a PyTorch reference that
already passed independent validation. It implements, builds, registers, and
tests the target XPU path in the target environment. A successful compilation
alone is never sufficient.

### Diagnosis Agent

The Diagnosis Agent receives a structured `BugReport`, the source task input and
output, and all referenced artifacts. It returns a root-cause analysis, repair
conclusion, evidence references, confidence, and a concrete `next_action` such
as `REDISCOVER_OPERATOR`, `DISPATCH_TORCH_FIX`, `DISPATCH_XPU_FIX`, `RETRY`, or
`BLOCKED`. It must distinguish observed facts from hypotheses.

### Validator and Integration Agents

Validators are independent of the Agent that produced an implementation. They
re-run the relevant checks and write reports. Integration work proves the real
model/service path, not only an isolated unit test.

## Mandatory execution protocol

### Codex interactive entry

For an authorized adaptation, use the project skill
`.agents/skills/infer-forge-adaptation/SKILL.md` and read
[the interaction protocol](docs/migration/codex-interaction.zh-CN.md).
Configure the first connected Graph with `--interaction-mode codex`, then use
`cli/adaptation.py context --run-id ...` and `advance --run-id ...` against the
same external state database. Do not combine this with automatic recovery.
Graph decision handoffs use `submit-decision`; operator failures retain the
durable diagnosis protocol below. Never replay an accepted decision whose
execution has no completion receipt. P1 does not provide Pod resource locks or
replace the required independent execution of validation.

### Source ownership

Put host-side argument parsing and command entry points under `cli/`. Implement
task behavior under the matching `operations/` domain: `intake`, `discovery`,
`deployment`, `validation`, or `operators`. The CLI delegates execution; it does
not duplicate task logic or acceptance rules.

`engine/` owns scheduling and recovery; `engine/state/` owns Journal and Task
Memory. `runners/` owns executable workflow/task sequences, without command-line
parsers. Shared contracts, errors, versioned-resource paths and runtime storage
belong under `core/`; import repository resource roots from `core.paths`.
Libraries must not import `cli` or parse `sys.argv`.

`tools/` is reserved for `probe/`, replayable `patches/`, and portable `torch/`
references. Do not add host task commands or coordination state modules there.
The old host script paths are removed, not compatibility aliases. Update
catalogs, contracts, workflows, tests and documentation whenever an entry moves.
Run `python3 cli/maintenance/check_repo_references.py` to check references and
dependency direction. See [source layout](docs/architecture/source-layout.zh-CN.md).

### Capability regression scenarios

Every newly supported capability must have a mandatory local E2E scenario using
its production CLI/workflow, real scheduler and validators, and persisted
evidence. Replace only external cluster/runtime/Agent dependencies. Cover
completion, rejection and restart; never prepopulate success reports or modify
scheduler state to make a scenario pass. Model adaptation is the first template
under `tests/e2e/`. Real-device smoke and real-model regression are optional,
explicitly authorized tiers. Local `SIMULATION_PASS` is not hardware readiness.
See [scenario protocol](tests/e2e/README.md).

### Runtime write ownership

Repository files are versioned source, not a run workspace. Managed entry points
reject runtime output, Journal, Task Memory, and SQLite paths inside this source
checkout. Use `INFER_FORGE_STATE_ROOT` to select an external directory; the default
is `$XDG_STATE_HOME/infer-forge`, or `~/.local/state/infer-forge`.

Each run owns a directory identified by `run.json`. Each execution or retry owns
a fresh `tasks/<task-id>/attempts/<attempt-id>/` directory. The scheduler's claim
payload exposes its paths in `input.workspace`:

- `input/`: the dispatched task snapshot and copied execution inputs.
- `scratch/`: temporary investigations; excluded from the formal inventory.
- `output/`: the current attempt's candidate results and formal evidence.
- `logs/`: execution logs.

Never reuse a previous attempt as a writable output directory. Worker evidence
must reside in the claimed attempt's `output/`, not a previous attempt or an
arbitrary external directory. The scheduler records `result.json` and
`manifest.json`; a manifest inventories files and hashes but does not prove
correctness or authorize promotion. Preserve previous attempts for diagnosis.

Graph/deployment/performance runners allocate managed attempts automatically.
Ordinary standalone task commands under `cli/` retain their exact explicit `--out`
directory, but require it to be external and fresh. Read the returned
`artifact_root` or task workspace instead of predicting output paths.

These are cooperating-tool controls, not an OS sandbox. Arbitrary shell commands
can still write elsewhere. Agent-authored experiments belong in `scratch/`;
repository changes must be intentional source changes with task/evidence context.
Do not invent a new repository directory for every runtime investigation.
See [runtime write ownership](docs/migration/runtime-write-policy.zh-CN.md).

### 1. Start or resume an adaptation run

Use a durable state database outside temporary source files. Never create a
second run because a previous command was interrupted; resume the existing
`run_id`.

```bash
python3 cli/adaptation.py \
  --state /path/to/adaptation.db \
  create-run \
  --run-id <run-id> \
  --model <model-id> \
  --model-revision <revision> \
  --plugin-revision <revision> \
  --backend <backend>
```

Before doing expensive work, record the model revision, plugin revision, target
hardware, runtime versions, and artifact root in the run context.

### 2. Prove the deployment environment before discovery

Before discovery, the Main Agent must run the deployment environment-proof Task
and bind its result to the run. Discovery is rejected until the proof records a
ready Pod, importable runtime, ready vLLM-Kunlun code worktree, and visible XPU
devices. All later investigation and child tasks must use the Pod and code
context recorded by this handoff; a new throwaway Pod is not equivalent.

```bash
python3 cli/adaptation.py \
  --state /path/to/adaptation.db \
  environment \
  --run-id <run-id> \
  --user-id <user-supplied-id>
```

`user_id` is the resource owner's ID supplied by the user. If it is unknown,
ask the user before execution; never guess it from the host login, repository
path, example contracts, or an existing resource name. Pass it via `--user-id`
(Graph: `--set user_id=<user-supplied-id>`), or `execution.user_id` in the
contract. Legacy explicitly configured `USER_ID` remains supported. Environment
attempts record the ID, and scheduler retries reuse the recorded value.

The harness generates the environment contract from the cluster profile and the
environment Task, persisting it in the attempt. Do not copy model-specific YAML
examples or hand-author launch settings. Target service contracts come from the
validated MAT-005 DeploymentPlan in the Journal; Graph rejects a manually supplied
`contract_instance`. Direct proof replay may use a previously generated external
contract. Generated YAML and runtime outputs never belong in versioned source.

Environment retries automatically reuse the Pod recorded in the run (including
failed proofs), and direct proof execution attaches to the existing deployment
before applying any manifest. Use `--attach-pod` to select a prepared Pod
explicitly. A non-ready Pod is retained for diagnosis; it is not replaced.

```bash
python3 cli/adaptation.py \
  --state /path/to/adaptation.db \
  environment \
  --run-id <run-id> \
  --attach-pod <prepared-pod>
```

If the deployment proof was already executed by a separate workflow step, its
validated `status.json` may be imported instead:

```bash
python3 cli/adaptation.py \
  --state /path/to/adaptation.db \
  environment \
  --run-id <run-id> \
  --status /path/to/environment/status.json
```

The proof must leave one prepared Pod with an importable runtime, a pinned
vLLM-Kunlun code worktree, and visible target XPU devices. All runtime
investigation must execute in that Pod and cite its code/environment fingerprint.

**Keep one debug environment.** Import failures, operator errors, timeout and
server crashes are reasons to inspect logs and repair/restart the affected
process in the same Pod. Do not delete/recreate a Pod, roll the deployment, or
reinstall a working stack to retry a model. Replacement requires a diagnosed
Pod/node failure or an explicit environment change, with evidence preserved.

**MiniMax-M2.5 is the environment gate.** Every `--phase environment` invocation,
including a replay of an external contract, derives the base model and
smoke command from `config/clusters/p800-cluster.yaml`. Target weights are not a
substitute. Require the base model identity, health, prefill/decode and backend
evidence before model investigation.
Launch this known-good baseline directly, without MAT-028/toy bring-up. Stop
after the environment proof; only then run target intake, scan, capability
matching, gap discovery and target toy bring-up. Never launch the target service
as part of proving the environment. The deployment CLI defaults to `environment`;
`--phase all` is an explicit legacy standalone deployment, not this workflow.

**Diagnose before repairing.** Deployment no longer auto-discovers or executes
`tools/patches/patch_*.py`, including the commit-specific Kunlun drift repair.
The drift precheck remains read-only. Repair only an observed incompatibility
against the installed revisions, record the diff and evidence, and keep the
repair replayable in versioned source. No blanket drift-patch step is required.

**Toy before target weights.** Run MAT-028 with dummy weights and require engine
construction, prefill and at least two decoded tokens. After any repair, repeat
toy bring-up, then shim handoff, then target service proof. The deployment
executor also runs a fresh toy probe before a new target server launch, including
direct CLI calls; failure retains the Pod and blocks the full checkpoint load.
The known-good MiniMax environment smoke is the deliberate real-weight baseline.

**Reuse the engine's model network when it exists (hard rule).** Before
considering any out-of-tree model implementation, check the capability
match: if the architecture resolves through vLLM's ModelRegistry or the
plugin's registered models (`REGISTRY` / `MODULE` verdicts), the network
already exists — do not implement a model layer. Every remaining gap is
then an operator-level gap and enters the OperatorSpec path directly. An
OOT model is only for an architecture nothing registers (`ABSENT`) — and
even the plugin treats its own OOT models as temporary state (upstream
`models/__init__.py` carries a "Remove all of models registration" TODO;
Gemma4 already ships without Kunlun-specific model files). Run
glm52-int-w8a8-p800-001 took the reuse path end to end:
GlmMoeDsaForCausalLM resolved to the deepseek_v2 network and all the work
happened at the operator layer.

**Operator integration ladder (prefer the top).** When replacing or adding
an operator:

1. **Decorator registration** — `@register_oot("LayerName")` or
   `direct_register_custom_op` into the torch dispatcher. Resolved when
   vLLM builds the layer, import-order insensitive, no file edits.
2. **Post-import wrap** — wrap the existing symbol at runtime (the
   apply_torch_decode_patch pattern). Reversible, but import-order
   sensitive and invisible to source inspection.
3. **Text patch** — last resort: an exact-anchor file edit under
   tools/patches/, carrying all the replayability constraints above.

The mechanics, the four op namespaces (`torch.ops._C` /
`torch.ops.xspeedgate_ops` / the `kunlun_ops` pybind facade /
`torch.ops.vllm::*`), and the known traps (PluggableLayer has no
`forward_oot` dispatch — override `forward()` or it never runs; dispatcher
registration cannot be rolled back) are documented with upstream line
numbers in `openwiki/vllm-kunlun/architecture.md`.

A report must include enough evidence to identify tensor inputs,
outputs, shape/rank, dtype, layout, semantics, call site, and failure context.
Convert the report through the orchestration entry point:

```bash
python3 cli/adaptation.py \
  --state /path/to/adaptation.db \
  discover \
  --run-id <run-id> \
  --report /path/to/gap-report.json \
  --model <model-id> \
  --model-revision <revision> \
  --plugin-revision <revision> \
  --backend <backend>
```

Discovery is strict. If a required field is unknown, preserve the uncertainty
and create a diagnosis or blocked task rather than guessing.

### 3. Dispatch and complete child tasks

Workers claim only the stage they are qualified to execute:

```bash
python3 cli/adaptation.py \
  --state /path/to/adaptation.db \
  claim --run-id <run-id> --worker <worker-id> --stage torch --limit 1
```

Use `--task-id <task-id>` with `--run-id` for a specific assignment. Both
selection and expired-lease recovery are limited to the requested scope.
Read `context --run-id <run-id> [--task-id <task-id>]` before dispatch; it is
read-only and does not authorize execution. The claim-time `input/task.json`
freezes the task context and acceptance references without a lease token.

The worker must read the complete task payload, write artifacts under the run's
artifact root, run the independent checks required for that stage, and submit a
JSON result:

```bash
python3 cli/adaptation.py \
  --state /path/to/adaptation.db \
  complete \
  --task-id <task-id> \
  --worker <worker-id> \
  --lease-token <token-from-claim> \
  --result /path/to/result.json
```

Completion requires the current unexpired claim token. A reused worker name
does not authorize an older attempt. Renew long-running work before expiry:

```bash
python3 cli/adaptation.py \
  --state /path/to/adaptation.db \
  renew-lease --task-id <task-id> --worker <worker-id> \
  --lease-token <token-from-claim> --lease-seconds 300
```

Results follow `contracts/worker_result.schema.yaml` and the evidence-binding
rules in `docs/migration/worker-results.md`. Empty results, bare PASS reports,
missing artifacts, mismatched hashes or task/environment identities, and
missing independent validation are rejected into diagnosis. Local fake-agent
runs must explicitly declare `metadata.evidence_mode: simulation`; they cannot
satisfy an environment-backed real run.

The scheduler creates the next operator stage only after the previous task
passes. The normal chain is `torch -> xpu -> integration`.

### 4. Handle every failure through diagnosis

Do not hide an exception, edit the task status directly, or silently retry a
failed implementation. Report the failure so the scheduler creates a durable
diagnosis task:

```bash
python3 cli/adaptation.py \
  --state /path/to/adaptation.db \
  fail \
  --task-id <task-id> \
  --worker <worker-id> \
  --lease-token <token-from-claim> \
  --error '<structured error or concise original message>'
```

The Diagnosis Agent claims `--stage diagnosis`, examines the `BugReport` and
artifacts, then submits its structured conclusion with `resolve-diagnosis`.
That command also requires the current `--lease-token`.
The Main Agent must consume that conclusion before choosing the next action.

### 5. Prove integration before delivery

At minimum, functional delivery requires:

- PyTorch reference correctness and independent validation.
- XPU build/registration evidence and device correctness across relevant
  shapes, dtypes, and layouts.
- Evidence that the real model path dispatched to the intended XPU operator.
- Service health and a real model request/regression check.
- No unreported CPU/Torch shim fallback on a path declared XPU-ready.
- Persisted artifacts and a final run/task summary.

## Evidence gates

An Agent must not report `PASS`, `READY`, or `PROMOTED` solely because a command
returned exit code 0. Results must point to durable evidence files.

| Stage | Required evidence |
| --- | --- |
| Discovery | Reproducible call site/failure, complete `OperatorSpec`, input/output contract, and source references. |
| PyTorch | Reference source, focused tests including edge cases, and an independent numerical validation report. |
| XPU | Target-environment build/registration record, device tests, dispatch proof, and an independent device validation report. |
| Integration | Real model/service request, output or accuracy regression, health proof, and fallback check. |
| Diagnosis | Original error/traceback, reproduction evidence, root cause, repair conclusion, confidence, and `next_action`. |

If evidence is missing, use `BLOCKED` or `REWORK`; do not promote the task.

## Performance phase boundary

Performance is a separate post-functional phase. Do not block initial
correctness work on optimization, and do not claim performance from a fake run,
one request, or a profiler run whose overhead was measured as throughput.

After functional readiness, the Main Agent may choose whether to enable:

```text
fixed benchmark matrix
  -> baseline report
  -> optional profiler capture
  -> trace analysis
  -> optimization child task
  -> correctness regression
  -> benchmark regression
  -> promote or rollback
```

Benchmark and profiler artifacts must identify model/plugin revisions, hardware,
dtype, parallelism, input/output lengths, concurrency or request rate, warmup,
seed, and tool versions. Keep benchmark/profiler work separate from the
functional task verdict.

## Prohibited shortcuts

- Do not bypass `TaskScheduler` or edit SQLite state by hand.
- Do not modify the model/plugin first and reconstruct evidence afterward.
- Do not guess an operator's shape, dtype, layout, semantics, or tolerance.
- Do not mark a task complete with only prose or `{"status":"PASS"}`.
- Do not promote an XPU task based only on compilation or import success.
- Do not silently replace an XPU path with a Torch/CPU fallback.
- Do not let one operator failure terminate unrelated operator discovery.
- Do not report fake-agent or simulator results as real hardware evidence.
- Do not repair the runtime environment (plugin site-packages, pinned
  worktree, in-pod state) without a committed, idempotent, replayable patch
  under `tools/patches/`. A fix that lives only in a pod dies with the pod.
- Do not place model weights, credentials, PATs, private endpoints, raw traffic,
  or large traces in the repository.

## Handoff checklist

Before handing work back to the Main Agent, a child Agent must provide:

1. The claimed task id and operator key.
2. A machine-readable result with an explicit verdict.
3. Absolute or run-relative paths for every evidence artifact.
4. The exact command/environment used, including revisions and target device.
5. Known limitations, unresolved hypotheses, and recommended next action.

The Main Agent should report the run id, task ids, verdicts, and evidence paths
so another Agent can resume without relying on chat history.
