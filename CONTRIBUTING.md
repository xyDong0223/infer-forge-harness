# Contributing

## Change boundaries

Every change must identify whether it modifies a Contract, Workflow, Task, Skill, Tool, Runner, Adapter, Validator, Catalog, or documentation. Do not mix unrelated layers in one change.

Contract changes require a migration note and regression fixtures. Skill changes require a Golden Task or an explicit reason why no fixture is possible. Tool changes require a fake adapter or deterministic contract test. Adapter changes require environment-scoped integration evidence. Validator changes must preserve rejection behavior for known bad states.

## Runtime and secrets

Never commit model weights, tokens, private endpoints, raw production traffic, or large traces. Use an external artifact root and commit only sanitized summaries and checksums.

## Pull requests

Use a focused PR title such as `[Contract]`, `[Workflow]`, `[Task]`, `[Skill]`, `[Tool]`, `[Adapter]`, `[Validator]`, `[Test]`, or `[Docs]`. Include the task ID, changed layer, tests run, known limitations, and reproducibility command.

## Local checks

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python -m compileall runners validators tools
```

P800 integration tests are opt-in and must identify the namespace, image digest, model revision, hardware, and cleanup policy before execution.
