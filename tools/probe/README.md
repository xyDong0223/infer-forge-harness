# Probes

Scripts pushed into a prepared pod and executed where the vendor operations
actually exist. A probe answers one bounded question and prints structured
output for the caller to persist as evidence.

Groups by purpose:

- Model and config: `model_support_probe.py`, `model_fingerprint_probe.py`,
  `capability_match_probe.py`, `parser_conformance_probe.py`.
- Runtime health: `engine_core_drift_precheck.py`, `runtime_drift_probe.py`,
  `torch_shim_probe.py`.
- Bring-up: `toy_bringup_probe.py`, `layer0_golden.py`, `layer0_xpu.py`,
  `layer_swap.py`.
- Operator-level: `moe_layer_probe.py`, `sliding_window_decode_probe.py`,
  `block_sparse_attention_probe.py`, `qknorm_rope_probe.py`,
  `qknorm_rope_insert_probe.py`, `quantized_linear_probe.py`,
  `swiglu_oai_probe.py`, `kernel_ut_replay.py`.
- Reference: `cpu_reference_logits_probe.py`, which stays conservative so a
  reference run can never quietly share the candidate's device.

Probes run inside the recorded pod of the environment proof; they cite that
pod's fingerprint rather than opening a new one.
