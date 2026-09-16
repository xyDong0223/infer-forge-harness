# Skills

Engineering method units: preconditions, decision rules, verification, and exit
conditions for one class of work. Each subdirectory holds its own `SKILL.md`
(authoritative, so it needs no separate README) plus `skill.yaml`.

Registered in [`../catalog/skill_catalog.yaml`](../catalog/skill_catalog.yaml),
which maps every workflow `task_type` to its method unit. The graph runner
records the selected skill in Task Memory.

A skill enters only after a Golden Task and an independent validator pass.
Failures are classified before the method is changed, so a skill is corrected
by evidence rather than by the first plausible explanation.
