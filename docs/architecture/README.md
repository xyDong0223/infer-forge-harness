# Architecture

- [`codex-integration-plan.zh-CN.md`](codex-integration-plan.zh-CN.md): proposed
  incremental plan for Codex-led model adaptation, including task handoffs,
  independent validation, shared-Pod execution, and migration acceptance.
- [`implementation-layers.zh-CN.md`](implementation-layers.zh-CN.md): current
  implementation guide in Chinese, covering abstraction layers, their purpose,
  target axes, execution flow, and migration boundaries.
- [`system-overview.md`](system-overview.md): what the Core owns versus what
  adapters own, the declared platform scope, and directory ownership.
- [`adapter-and-contracts.md`](adapter-and-contracts.md): the three platform
  seams, the executable Protocol objects, and the configuration-first extension
  order.
- [`target-resolution.md`](target-resolution.md): how a target is parsed, why
  it must pass the compatibility gate, and what each status means.

The earlier layer-level contract and component ownership map live one level up
in [`../architecture.md`](../architecture.md). Its phase-based implementation
notes predate the target abstraction; use the implementation guide above for
the current snapshot.
