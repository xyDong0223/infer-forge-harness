# Capability scenarios

Model adaptation is the first executable capability scenario. Its registry entry
is [scenarios/model_adaptation.yaml](scenarios/model_adaptation.yaml), referenced
by the production workflow. Performance optimization and context/memory tuning
are not declared working scenarios yet.

## Required local tier

```bash
python -m pip install -e '.[test]'
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -m local_e2e tests/e2e \
  --basetemp /external/fresh-e2e-directory \
  --junitxml /external/model-adaptation-e2e.xml
```

Use a fresh external `--basetemp`: pytest owns and may clear that directory.
Never point it at an existing adaptation run. Torch, a cluster, credentials and
model weights are not required. The GitHub Actions local-scenarios job runs this
tier on pull requests and pushes, retaining the process logs, SQLite databases,
Journal, Task Memory, attempt manifests and JUnit report. Making that job a
required branch-protection check is a separate repository setting.

The scenario runs the **unchanged production graph topology**, not a shortened
test-only workflow:

```text
create-run CLI
  -> Graph CLI: MiniMax environment (no toy), intake, scan, classification
  -> persistent operator dispatch
  -> evaluation, plan, toy bring-up, shim handoff, service, accuracy, baseline
  -> WAITING_FOR_OPERATORS
  -> separate simulated Agent processes: claim -> compute -> validate -> complete
  -> new Graph process with --resume
  -> fresh combined-model service/accuracy regression, live integration gate
  -> memory/API checks, support matrix
  -> persisted SIMULATION_PASS delivery receipt
```

Only external cluster/runtime observations, remote revision resolution, Agent
implementations and the external capacity-planner provider are doubled. CLIs, operations, task validators, graph edges,
scheduler leases/transitions, result validation and artifact storage run for
real. The subprocess bootstrap is test-only and fails closed on unexpected
external actions. Worker arithmetic and independent evidence gates run locally;
no validator is replaced with a success function.

The required cases cover completion, missing worker evidence and durable
diagnosis, an abandoned claim followed by lease expiry and restart, rejection
of the old lease, rejection of pre-operator service evidence even after every
worker succeeds, incomplete operator contracts, environment evidence corruption,
stale accuracy after a fresh service proof, and refusal to import a simulation
proof into a real run. Each case checks that the scenario did not modify
repository source files.

The environment-input case starts without a seed YAML, checks the generated
contract and its source metadata, and removes legacy `USER_ID`, checks the persisted
missing-input rejection against the shared status schema before cluster access,
supplies `--user-id`, and resumes
the same run in a new process using its recorded ID and Pod. A separate case
rejects a manual Graph contract override before cluster access. Full delivery
asserts that service execution consumes the real MAT-005 generated contract.

All persisted deployment proof states are checked against the shared status
schema, including success and failure. A pre-plan intake failure exercises the
contract-free triage path, preservation of original console logs, resuming triage
from the Journal, and retrying intake in the same Pod. The resulting UNKNOWN
diagnosis remains NEEDS_HUMAN; it cannot enter patch placement or vendor handoff.
Changing an established run's owner is rejected before external operations and
leaves its generated plan and prepared environment unchanged.
Environment CLI retries and imports enforce that owner across successful and
failed handoffs, even if ambient USER_ID changes. Generated-plan tampering blocks
service, triage and patch placement before external operations; intact plans
remain reusable after restart.

The failed environment handoff case injects a runtime import error at the
external command boundary, checks the persisted INSTALL_FAILED proof and import
log, rejects an owner change, and retries with the same owner and Pod.
Environment failures stop at a diagnostic terminal; model triage requires a
successful environment.
Workflow intake requires EnvironmentProof and never creates its standalone
ephemeral probe Pod.

The completion case checks persisted fact order and actual adapter commands:
MiniMax baseline launch precedes intake; target toy follows discovery and precedes
target service launch. Boundary cases check that omitted `--phase` and explicit
`--phase environment` both launch only MiniMax and never invoke a toy probe.

**SIMULATION_PASS proves process integration, not device correctness, actual
model accuracy or performance.** Read the receipt's evidence mode and Journal
context; an intermediate node's `*_READY` state alone is not hardware evidence.

The same production scenarios check progress explanations across process
boundaries: an unclaimed task, an expired lease without read-side recovery,
queued diagnosis, blocked measured contracts, a ready-to-resume graph and the
final simulation delivery. Status queries retain original states and report
the next responsible actor without treating explanations as evidence.

## Production Graph/scheduler handoff

Create the run using `cli/adaptation.py`, then connect the graph explicitly:

```bash
python cli/workflow/graph.py \
  --scheduler-state /external/state.sqlite \
  --run-id my-adaptation \
  --artifact-root /external/runs/my-adaptation \
  --subject MyModel \
  --operator-report /external/measured-operators.json \
  --shim-registry /external/shim-registry.json \
  --env hardware=P800 \
  --env stack_commit=PINNED_PLUGIN_COMMIT \
  --set model_path=/mounted/model \
  --set user_id=<user-supplied-id> \
  --execute --resume --json
```

Use the run's exact model, revisions, environment and artifact root. Omit
`--operator-report` only when the classification already carries complete
operator contracts or genuinely reports no actionable gaps. A coarse capability
name is not permission to invent a tensor contract. `--shim-registry` supplies
the measured declarations used by MAT-029; an old file-only request does not
satisfy a scheduler task.

Without `--scheduler-state`, the older graph-only mode is preserved and does
not issue the new scheduler-backed functional delivery receipt. In connected
mode, exit `3` means workers are still pending: continue claiming/completing
tasks through `cli/adaptation.py`, then resume the same graph and `run_id`.
Pass opaque leases as `--lease-token=TOKEN` when constructing CLI arguments;
a valid URL-safe token can begin with `-`.
Exit `2` means blocked/rework. A vendor ticket or a waiting candidate is not
functional readiness. `--until-node` intentionally stops a partial walk; exit
`0` from that command is not a delivery claim.

The bridge revalidates the environment handoff, performs idempotent discovery,
and rechecks persisted worker evidence before delivery. Dispatch and integration
are not skipped merely because an old Journal fact says they succeeded.
Once workers are ready, service and accuracy are rerun: the frozen pre-candidate
baseline is not evidence for the combined final model. Scheduler snapshots
recorded before service and accuracy execution bind both regressions to completed
tasks; accuracy also binds the exact final service proof and its hash. Recovery
attempts use the same capture hook. Final comparison and receipt publication
share a scheduler transaction, so concurrent discovery cannot overtake delivery.
An accepted scheduler proof can be imported into the Graph Journal. A known
prepared Pod is attached, not replaced by a new deployment.

## Optional real tiers

The hardware tests are skipped unless their individual authorization variable
is exactly `1`. They never activate the local bootstrap and require an existing
**real** run and its already-proven Pod.

Set `INFER_FORGE_HARDWARE_SCENARIO` to an external JSON configuration containing:

| Field | Requirement |
| --- | --- |
| `state`, `run_id`, `artifact_root`, `subject` | Existing scheduler run and recorded root/model |
| `pod`, `namespace` | The proven Pod, in the configured adapter namespace |
| `image_digest`, `hardware` | Recorded run environment identity |
| `model_revision`, `plugin_revision` | Pinned revisions matching the run |
| `environment` | The exact Graph `--env` mapping used by the run |
| `context` | Graph node arguments such as model path, user_id, port and served-model name |
| `cleanup_policy` | `retain_prepared_pod`; these tests never delete the prepared environment |

`KUBECONFIG` and any runtime credentials remain in the external execution
environment, never in the scenario file or repository. The scheduler run's
environment must already record `hardware` and `image_digest`.

```bash
# Re-proves the existing Pod, including base-model prefill/decode.
INFER_FORGE_RUN_DEVICE_SMOKE=1 \
  python -m pytest -q -m device_smoke tests/e2e/test_model_adaptation_hardware.py

# Requires completed operator stages and upstream Graph facts.
# Starts at service proof WITHOUT --resume, forcing fresh service and accuracy work.
INFER_FORGE_RUN_REAL_MODEL=1 \
  python -m pytest -q -m real_model tests/e2e/test_model_adaptation_hardware.py
```

Missing context fails the opted-in test; it does not silently skip or create a
replacement run. A real-model run only completes with a scheduler-backed
`FUNCTIONAL_READY` receipt. Neither optional tier is executed by local CI.

## Adding a capability

Add a scenario registry entry linked from its actual production workflow and a
`local_e2e` test that invokes its real entry points. Include success, rejection
and process restart cases, assert persisted outcomes and evidence, and double
only external dependencies. Keep runtime fixtures outside the checkout and
retain process logs for diagnosis. The registry guard checks that required
case names exist and that optional real tiers remain explicitly opt-in.

A future performance scenario must additionally demonstrate fixed-workload
comparison, correctness preservation and promote/rollback decisions. A future
context/memory scenario must exercise its measured boundary and failure cases.
Do not register either as implemented by copying a PASS fixture.
