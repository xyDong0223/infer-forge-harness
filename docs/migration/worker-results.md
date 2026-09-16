# Evidence-backed worker completion

This migration makes scheduler gates enforce the existing evidence protocol.
Old success-shaped result dictionaries are intentionally no longer sufficient.
Existing run and task rows remain readable; no state database rewrite is needed.

## Claims and leases

Keep the `lease_token`, `lease_expires` and `attempt` returned by `claim`.
`complete`, `fail`, and `resolve-diagnosis` require that token, even if the worker
name is unchanged. An expired or superseded token cannot publish a result.
`renew-lease` extends a live lease without changing the attempt or token.
After expiry, claim again and produce a report for the new attempt.

The Python API also requires `lease_token`; the concise positional token form
remains available. `complete(task_id, worker_name, result)` without a token is
no longer accepted. Both CLI entry points expose the token and renewal options.

Completion/failure, their events, and creation of the successor/diagnosis task
commit in one SQLite transaction. Infrastructure exceptions are propagated and
roll back the entire transition rather than leaving a half-completed pipeline.
For historical interruptions, `run_adaptation.py --state <db> reconcile
--run-id <run>` repairs missing diagnosis tasks and missing successors.
Successors are created only after revalidating the persisted evidence; legacy
bare verdicts are returned as `blocked` with reasons, not silently promoted.

## Result envelope

[`worker_result.schema.yaml`](../../contracts/worker_result.schema.yaml)
describes the serialized envelope. The scheduler additionally verifies the
files and their binding through
[`engine/result_validation.py`](../../engine/result_validation.py).

Every result records:

| Field | Source |
| --- | --- |
| `schema_version` | `1` |
| `status` or `verdict` | Explicit `PASS`; conflicting failure fields are rejected |
| `task_id`, `operator_key`, `stage`, `attempt` | The current claimed task |
| `environment_fingerprint` | The bound run's environment proof |
| `evidence_mode` | `real`, or explicitly configured `simulation` |
| `evidence` | Object mapping evidence roles to nonempty readable files |
| `evidence_sha256` | Object mapping the same roles to SHA-256 digests |

Evidence paths are absolute, or relative to `run.metadata.artifact_root`.
They are never implicitly relative to the worker's current directory.
New managed claims expose `input.workspace`; submitted evidence must resolve
inside that attempt's `output/`. Reclaiming work allocates a new directory and
does not reuse the previous attempt's files. See
[runtime write ownership](runtime-write-policy.zh-CN.md) for layout and migration.

| Stage | Required evidence roles |
| --- | --- |
| `torch` | `reference_artifact`, `focused_tests`, `independent_validation` |
| `xpu` | `build_record`, `registration_record`, `device_test`, `dispatch_report`, `independent_validation` |
| `integration` | `integration_report`, `service_regression`, `accuracy_regression`, `fallback_report`, `independent_validation` |
| `diagnosis` | `diagnosis_report` |

An independent validation report is a JSON object with the same task, attempt,
environment and evidence-mode fields as the result, a `PASS` verdict, a
nonempty list of named checks with `passed: true`, and a recorded `validator`
identity different from the producing worker. Its `evidence_sha256` maps every
submitted evidence role except the validation report itself to the exact bytes
that were validated. Hash the report last and include that digest in the result.

The scheduler verifies this recorded provenance. It does not authenticate
external Agent identities or execute numerical/device validation itself.
Independent execution remains mandatory; fabricated reports are not evidence.

A diagnosis report binds the same identity and must contain `diagnosis`,
`repair_conclusion`, finite `confidence` from 0 to 1, and `next_action`.
These fields must match the submitted result. Allowed actions are
`REDISCOVER_OPERATOR`, `DISPATCH_TORCH_FIX`, `DISPATCH_XPU_FIX`, `RETRY`, and
`BLOCKED`. `PASS` means the diagnosis is complete, not that its source task
has recovered; `next_action: BLOCKED` is a valid completed diagnosis.

Rejected results are persisted as task failures with `validation_errors` in
the BugReport metadata, and create diagnosis work for ordinary operator stages.
CLI rejection returns a nonzero exit code. Lease errors leave the current task
untouched.

## Environment imports and simulations

Real runs require an environment proof even when created through the Python
API. Imports validate the actual state, readiness/service checks and required
artifact files. `environment --status` resolves evidence beside the imported
status file; contract executions return an explicit `artifact_root`.

The bound fingerprint is the SHA-256 digest of the actual
`environment_fingerprint.txt`, not an unverified field supplied in a status
dictionary. Required environment artifact hashes are retained with the run.
Operator specifications must reference that fingerprint and the run's model
and backend.

A different fingerprint cannot silently replace a proof after operator tasks
have been discovered. Such an import is rejected without altering the run;
existing work must not inherit evidence from a changed environment. This
migration does not implement automatic invalidation/rebasing of existing tasks
onto a replacement environment.

For local orchestration rehearsals only, explicitly create a run with
`metadata.evidence_mode: simulation`. Simulation results may use a null
environment fingerprint, but retain all task, evidence and lease checks.
FakeAgentHarness refuses real or environment-required runs. Simulation results
must not be represented as target hardware readiness or performance evidence.
