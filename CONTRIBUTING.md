# Contributing

## Change boundaries

Every change must identify whether it modifies a Contract, Workflow, Task, Skill, Tool, Runner, Adapter, Validator, Catalog, or documentation. Do not mix unrelated layers in one change.

Contract changes require a migration note and regression fixtures. Skill changes require a Golden Task or an explicit reason why no fixture is possible. Tool changes require a fake adapter or deterministic contract test. Adapter changes require environment-scoped integration evidence. Validator changes must preserve rejection behavior for known bad states.

## Source ownership

Host command entry points belong in `cli/`, with task implementations in the
matching `operations/` domain. Scheduling belongs in `engine/`, coordination
state in `engine/state/`, and executable sequences in `runners/`. Libraries
must not import `cli` or parse command-line arguments.

Reserve `tools/` for portable probes, replayable patches, and Torch references.
Use `core.paths` for versioned resource locations and `core.storage` for external
runtime writes. No compatibility script should be added at a removed path.
See [source layout](docs/architecture/source-layout.zh-CN.md) for the directory
map and required caller updates.

## Runtime and secrets

Never commit model weights, tokens, private endpoints, raw production traffic, or large traces. Use an external artifact root and commit only sanitized summaries and checksums.

## Pull requests

Use a focused PR title such as `[Contract]`, `[Workflow]`, `[Task]`, `[Skill]`, `[Tool]`, `[Adapter]`, `[Validator]`, `[Test]`, or `[Docs]`. Include the task ID, changed layer, tests run, known limitations, and reproducibility command.

## Local checks

Every newly supported capability must include an executable local E2E scenario
linked from its production workflow. Use the real CLI, workflow, scheduler,
validators and persistent artifacts; double only external cluster/runtime/Agent
boundaries. Cover successful delivery, rejection and process restart. Real-device
smoke and real-model regression are separate opt-in tiers, not prerequisites for
the local scenario. See [capability scenarios](tests/e2e/README.md).

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python cli/maintenance/check_repo_references.py
python -m pytest -q -m local_e2e tests/e2e
```

P800 integration tests are opt-in and must identify the namespace, image digest, model revision, hardware, and cleanup policy before execution.
