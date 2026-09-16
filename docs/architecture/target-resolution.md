# Target resolution

New workflows should load a target through `core.target.load_target()` and
call `require_supported()` before creating resources. The compatibility matrix
is the gate; it is not inferred from Python import paths.

`supported` may execute, `planned` is visible but intentionally blocked,
`unsupported` is an explicit no-go, and `unknown` means the combination has
not been declared. This distinction prevents a missing adapter from silently
selecting the default P800/vLLM-Kunlun path.

The legacy CLI and runners remain unchanged during this migration. A later
phase can translate legacy `--backend` values into a `TargetContext` at the
boundary, then pass the context inward.

`graph_runner.py` now accepts `--target <yaml>`. When supplied, it resolves the
target and applies the compatibility gate before loading or walking the task
graph. Existing invocations without `--target` retain their legacy behavior.
