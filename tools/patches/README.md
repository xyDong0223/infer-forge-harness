# Runtime patches

The replayable, idempotent repair set for the vLLM-Kunlun stack. These are
stack-specific on purpose: a repair to runtime state lives here as an
exact-anchor text edit that skips already-applied replacements, so a
reinstalled pod self-heals instead of silently regressing.

- `patch_vllm_kunlun_drift.py`: engine/plugin drift repairs.
- `apply_torch_decode_patch.py`: the post-import decode wrap.
- `vllm_kunlun_flashmla_sparse.py`: companion module the patch set deploys
  alongside its patches.

Hard rules, from `AGENTS.md`:

- A repair written into a pod but not committed here is an incident scheduled
  for the next reinstall, not a repair.
- Patch order can matter; do not assume one step's anchor exists until a prior
  step created it.
- A patch must be verified on a fresh install, not only on a pod that was
  already repaired by hand.
