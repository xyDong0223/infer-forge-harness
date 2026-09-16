# Target resolution

New workflows should load a target through `core.target.load_target()` and
call `require_supported()` before creating resources. The compatibility matrix
is the gate; it is not inferred from Python import paths.

`supported` may execute, `planned` is visible but intentionally blocked,
`unsupported` is an explicit no-go, and `unknown` means the combination has
not been declared. This distinction prevents a missing adapter from silently
selecting the default P800/vLLM-Kunlun path.

Legacy P800 defaults remain at the deployment boundary, but an explicit
target can no longer silently disagree with a deployment contract. Hardware
aliases are normalized; model, runtime axes and declared revisions must agree.
Checkpoint revisions in `context.model.revision` participate in this check.
An environment-only proof may intentionally use a different base model.

`graph_runner.py --target <yaml>` binds the requested model to `--subject`,
checks compatibility and contract consistency before the graph walk, and
forwards the target and subject to deployment task commands. The task runner
checks them again before execution. Conflicting `--env` identities are rejected,
and declared revisions become part of the graph's Journal scope.

After an environment proof, graph runtime facts also carry the digest of the
actual `environment_fingerprint.txt` and the prepared Pod. Intake and the
environment proof itself are looked up without that downstream fingerprint,
since they establish it rather than depend on it.

Journal queries distinguish explicit unfiltered inspection (`environment=None`)
from a missing execution identity (`environment={}`, which yields no hits).
Graph recovery and input resolution reject failed, malformed, changed or
validator-rejected status files. New facts bind the status digest and command
return code; environment proof reuse also checks its mandatory artifacts and
fingerprint. This is not a live-Pod liveness probe, nor does a declared revision
alone prove which version was installed; runtime evidence is still required.
