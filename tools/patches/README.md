# Runtime repairs

Repairs must be reviewable and replayable against their recorded runtime revisions.
Deployment and environment proof never discover or execute this directory automatically.
Diagnose the installed source and validate a focused repair in the existing Pod,
then repeat the drift precheck and toy bring-up before loading target weights.

- `patch_vllm_kunlun_drift.py`: historical, version-specific repair reference;
  excluded from the adaptation workflow. It is not a prerequisite for environment
  or service proof.

- `apply_torch_decode_patch.py`: explicit post-import decode wrap.
- `vllm_kunlun_flashmla_sparse.py`: companion sparse-MLA implementation.

Record compatible revisions and exact source anchors for any manual repair.
Stage changes before writing and roll back partial failures. Prefer registered
operator overrides over text edits, and validate a repair with device evidence.
