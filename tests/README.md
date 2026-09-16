# Tests

Run commands from the repository root in an activated virtual environment.
Use pytest: the suite includes pytest functions and TestCase classes that rely
on pytest fixtures. `cli/maintenance/validate_scaffold.py` also invokes pytest.

| Tier | Dependencies and purpose |
| --- | --- |
| [unit/](unit/README.md) | Local component rules; some numerical modules require Torch |
| [integration/](integration/README.md) | Local cross-module scheduler, leases and worker evidence |
| [e2e/](e2e/README.md), `local_e2e` | Production topology and real internal components; external boundaries are simulated |
| E2E `device_smoke` / `real_model` | Explicit opt-in, prepared real environment and authorization |

## First run

Install `.[test]` and follow the [local quickstart](../docs/guides/quickstart.zh-CN.md).
The required local E2E tier needs no Torch, model weights, cluster or Agent API.
Its successful delivery is `SIMULATION_PASS`, not hardware readiness.

Use a fresh external `--basetemp`: pytest may clear it. Artifact inspection and
the exact command are in the quickstart; scenario requirements and hardware
authorization variables are maintained in the [E2E protocol](e2e/README.md).

## Development

Select the smallest relevant files, for example:

```bash
python cli/maintenance/check_repo_references.py
python -m pytest -q tests/unit/test_storage.py tests/unit/test_scheduler.py \
  tests/integration/test_scheduler_e2e.py
```

For the full local suite, install a Torch build compatible with your Python and
operating system, then run `python -m pytest -q tests`. Torch is not included in
`.[test]`. A `-k` filter cannot reliably avoid a missing Torch import during
collection; without Torch, select dependency-light files or `tests/e2e`.
Real tiers stay disabled unless explicitly authorized as described above.

New capabilities require their production CLI/workflow, real scheduler and
validators, persisted evidence, and completion, rejection and restart cases.
Only external dependencies may be doubled. Do not prepopulate success reports
or edit scheduler state to make a scenario pass.
