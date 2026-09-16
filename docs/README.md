# Documentation

- [`architecture/implementation-layers.zh-CN.md`](architecture/implementation-layers.zh-CN.md):
  current technical implementation guide in Chinese, explaining each abstraction
  layer, target resolution, execution flow, and implementation boundaries. Start here.
- [`architecture.md`](architecture.md): the earlier layer contract, target
  axes, and component ownership map, including historical migration notes.
- [`architecture/`](architecture/): the platform-neutral core model, the
  adapter/contract rules, and target resolution.
- [`guides/`](guides/): how to add a platform and how performance analysis is
  structured.
- [`migration/`](migration/): moving an existing script-based workflow into the
  harness.
- [`migration/runtime-write-policy.zh-CN.md`](migration/runtime-write-policy.zh-CN.md):
  external run directories, per-attempt write ownership, and artifact inventories.
- [`assets/`](assets/): diagrams, including editable sources.

Repository rules are authoritative in the root `AGENTS.md`; this directory
explains why they are shaped that way.
