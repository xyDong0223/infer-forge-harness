# Migration

Guidance for bringing existing engineering work into the harness:

- [`worker-results.md`](worker-results.md): migrating worker submissions to
  lease-token ownership, atomic transitions and evidence-backed stage gates.
- [`migrate-existing-workflow.md`](migrate-existing-workflow.md): the mapping
  from shell scripts, Kubernetes YAML, launch commands, patches, health checks,
  benchmark scripts, profilers, human judgement, and logs onto harness
  constructs — plus the rule to keep the original workflow as a golden
  reference and migrate one task at a time.

Migration keeps the legacy entry points working during the transition; a
half-migrated workflow must not start answering for both paths.
