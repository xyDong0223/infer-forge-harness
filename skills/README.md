# Skills

Engineering method units: preconditions, decision rules, verification, and exit
conditions for one class of work. Each subdirectory holds its own `SKILL.md`
(authoritative, so it needs no separate README) plus `skill.yaml`.

Registered in [`../catalog/skill_catalog.yaml`](../catalog/skill_catalog.yaml),
which maps every workflow `task_type` to its method unit. Catalog entries that
use a packaged method declare `method_package`; the graph runner snapshots the
resolved catalog contract and `SKILL.md` into every attempt's
`input/skill.json`, exports its path as `INFER_FORGE_SKILL_CONTRACT`, records
the routing in Task Memory, and includes the same packet in recovery Agent
requests. Package links are validated before a workflow executes.

A skill enters only after a Golden Task and an independent validator pass.
Failures are classified before the method is changed, so a skill is corrected
by evidence rather than by the first plausible explanation.
