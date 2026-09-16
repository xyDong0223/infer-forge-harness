# System overview

Infer Forge is a platform-neutral orchestration core. A target is the
combination of `hardware`, `engine`, `backend`, and optional `plugin`; support
is explicit in `compatibility/matrix.yaml`.

The core owns workflows, task state, journals, artifacts, evidence, gates, and
reports. Adapters own platform behavior. Performance analysis and model
adaptation share the same run context and evidence model.

Directory ownership is documented by package-level READMEs. `core/` is
platform-neutral, `adapters/` owns hardware and cluster behavior,
`runtimes/` owns engine/backend behavior, `runners/` owns orchestration, and
`tools/` owns deterministic actions. Legacy paths remain during migration and
are not a second source of architectural truth.

Current declared scope:

- Kunlun P800: vLLM-Kunlun and SGLang-Kunlun.
- NVIDIA B200: SGLang only.

An unlisted combination is unknown; an explicitly listed unsupported
combination must fail before execution.
