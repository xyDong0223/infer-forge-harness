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

### 1. Start or resume an adaptation run

Use a durable state database outside temporary source files. Never create a
second run because a previous command was interrupted; resume the existing
`run_id`.

```bash
python3 tools/run_adaptation.py \
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
python3 tools/run_adaptation.py \
  --state /path/to/adaptation.db \
  environment \
  --run-id <run-id> \
  --contract tasks/kdp-001-deployment-proof/instances/<model>.yaml
```

If the deployment proof was already executed by a separate workflow step, its
validated `status.json` may be imported instead:

```bash
python3 tools/run_adaptation.py \
  --state /path/to/adaptation.db \
  environment \
  --run-id <run-id> \
  --status /path/to/environment/status.json
```

The proof must leave one prepared Pod with an importable runtime, a pinned
vLLM-Kunlun code worktree, and visible target XPU devices. All runtime
investigation must execute in that Pod and cite its code/environment fingerprint.
A report must include enough evidence to identify tensor inputs,
outputs, shape/rank, dtype, layout, semantics, call site, and failure context.
Convert the report through the orchestration entry point:

```bash
python3 tools/run_adaptation.py \
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
python3 tools/run_adaptation.py \
  --state /path/to/adaptation.db \
  claim --worker <worker-id> --stage torch --limit 1
```

The worker must read the complete task payload, write artifacts under the run's
artifact root, run the independent checks required for that stage, and submit a
JSON result:

```bash
python3 tools/run_adaptation.py \
  --state /path/to/adaptation.db \
  complete \
  --task-id <task-id> \
  --worker <worker-id> \
  --result /path/to/result.json
```

The scheduler creates the next operator stage only after the previous task
passes. The normal chain is `torch -> xpu -> integration`.

### 4. Handle every failure through diagnosis

Do not hide an exception, edit the task status directly, or silently retry a
failed implementation. Report the failure so the scheduler creates a durable
diagnosis task:

```bash
python3 tools/run_adaptation.py \
  --state /path/to/adaptation.db \
  fail \
  --task-id <task-id> \
  --worker <worker-id> \
  --error '<structured error or concise original message>'
```

The Diagnosis Agent claims `--stage diagnosis`, examines the `BugReport` and
artifacts, then submits its structured conclusion with `resolve-diagnosis`.
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
