# Guides

## Running the harness

- [Local quickstart](quickstart.zh-CN.md): installation, local E2E, artifact
  inspection and an optional numerical tool example.
- [Real model adaptation](../../README.md#运行真实模型适配): prepared cluster,
  pinned model/runtime and scheduler-backed execution.
- [Troubleshooting](troubleshooting.zh-CN.md): preserve evidence, identify
  rejections and choose the appropriate recovery mechanism.

## Extending the harness

- [`add-workflow.zh-CN.md`](add-workflow.zh-CN.md): how to onboard an existing
  engineering process or a new capability, from Workflow/Task decomposition
  through scheduler integration, evidence gates, and mandatory local E2E.
- [`migrate-legacy-skill.zh-CN.md`](migrate-legacy-skill.zh-CN.md): how to
  decompose a legacy Skill that mixes goals, scripts, and operational knowledge,
  then migrate it through Golden Tasks and shadow validation.
- [`add-platform.md`](add-platform.md): the ordered checklist for wiring a new
  hardware or runtime target, including the rule that `planned` becomes
  `supported` only after real evidence exists.
- [`performance-analysis.md`](performance-analysis.md): what the performance
  workflow shares with model adaptation and what must be recorded in a report.

The configuration-first extension order these guides assume is defined in
[`../architecture/adapter-and-contracts.md`](../architecture/adapter-and-contracts.md).
