# Documentation

> 中文译本见 [README.zh-CN.md](README.zh-CN.md);如有不一致,以本英文版为准。

- [Local quickstart](guides/quickstart.zh-CN.md): installation, the complete local
  scenario, artifact inspection and an optional tensor comparison example.
- [CLI guide](../cli/README.md) and [configuration](../config/README.md):
  command selection, worker handoff and connecting your own environment.
- [Troubleshooting](guides/troubleshooting.zh-CN.md): common rejections,
  diagnosis recovery and CPU reference boundaries.
- [`architecture/source-layout.zh-CN.md`](architecture/source-layout.zh-CN.md):
  current source ownership, command/implementation separation, and migration rules.
- [`architecture/implementation-layers.zh-CN.md`](architecture/implementation-layers.zh-CN.md):
  current technical implementation guide in Chinese, explaining each abstraction
  layer, target resolution, execution flow, and implementation boundaries. Start
  here for architecture; use the quickstart above for a first run.
- [`architecture.md`](architecture.md): the earlier layer contract, target
  axes, and component ownership map, including historical migration notes.
- [`architecture/`](architecture/): the platform-neutral core model, the
  adapter/contract rules, and target resolution.
- [`guides/`](guides/): how to add a platform and how performance analysis is
  structured, including the Chinese
  [new workflow onboarding guide](guides/add-workflow.zh-CN.md) and
  [legacy Skill migration guide](guides/migrate-legacy-skill.zh-CN.md).
- [`migration/`](migration/): moving an existing script-based workflow into the
  harness.
- [`migration/runtime-write-policy.zh-CN.md`](migration/runtime-write-policy.zh-CN.md):
  external run directories, per-attempt write ownership, and artifact inventories.
- [`assets/`](assets/): diagrams, including editable sources.

Repository rules are authoritative in the root `AGENTS.md`; this directory
explains why they are shaped that way.
