# vLLM 0.25.1 API Drift Map for Plugin Ports

> **What this is**: every symbol move, signature change, and contract shift
> encountered when porting vllm-kunlun's fork (written against an older
> engine) onto vllm 0.25.1, with the fix that worked. Recorded during the
> GLM5.2-Int-W8A8 adaptation run `glm52-int-w8a8-p800-001` (2026-09-14).
>
> **How to use it**: before porting a plugin file, grep it for the "old
> symbol" column; each row says where the symbol went and what else the new
> location changes. Entries marked ⚠ change control flow, not just imports.

## Engine package removals

The entire `vllm.attention` package was deleted; its symbols scattered:

| Old import | New location |
| --- | --- |
| `vllm.attention.backends.abstract.AttentionBackend` | `vllm.v1.attention.backend` |
| `vllm.attention.backends.abstract.AttentionLayer` | `vllm.v1.attention.backend` |
| `vllm.attention.backends.abstract.AttentionMetadata` | `vllm.v1.attention.backend` |
| `vllm.attention.backends.abstract.AttentionType` | `vllm.v1.attention.backend` |
| `vllm.attention.backends.abstract.MLAAttentionImpl` | `vllm.v1.attention.backend` |
| `vllm.attention.backends.utils.get_mla_dims` | `vllm.model_executor.layers.attention.mla_attention` |
| `vllm.attention.ops.common.cp_lse_ag_out_rs` | `vllm.v1.attention.ops.common` |
| `vllm.attention.ops.merge_attn_states.merge_attn_states` | `vllm.v1.attention.ops.merge_attn_states` |
| `vllm.attention.utils.fa_utils.get_flash_attn_version` | `vllm.v1.attention.backends.fa_utils` |

Other moves inside the surviving packages:

| Old symbol | New location |
| --- | --- |
| `vllm.utils.cdiv` / `round_down` | `vllm.utils.math_utils` |
| `vllm.v1.attention.backends.utils.AttentionCGSupport` | `vllm.v1.attention.backend` |
| `vllm.v1.attention.backends.utils.AttentionMetadataBuilder` | `vllm.v1.attention.backend` |
| `vllm.model_executor.layers.mla.MultiHeadLatentAttention` | removed — see ⚠ below |
| `vllm.model_executor.layers.rotary_embedding.get_rope(rotary_dim=, base=, rope_scaling=)` | single `rope_parameters` dict kwarg — see ⚠ below |
| `FusedMoE` (class, with `.make_expert_params_mapping`) | factory function; use `fused_moe_make_expert_params_mapping(model, ...)` |
| `FusedMoE(...).maybe_all_reduce_tensor_model_parallel(...)` | removed — MoERunner all-reduces internally |
| `vllm.v1.attention.backends.mla.indexer.kv_spans_from_batches` | removed — reimplement locally (see harness practice note) |
| `vllm.v1.attention.backends.mla.indexer.split_prefill_chunks(seq_lens, workspace, offset)` | `split_indexer_prefill_chunks(seq_lens, query_lens, workspace, max_logits_bytes, request_offset)` |

## Contract shifts (⚠ control flow changes)

1. **MLA layer now owns q expansion, cache write, and v up-projection.**
   The old `MLACommonImpl.forward(layer, q, k_c_normed, k_pe, kv_cache,
   metadata, output, ...)` did all three inside the impl. In 0.25.1 the
   engine's `MLAAttention` (in `vllm/model_executor/layers/mla.py`) calls
   `impl.do_kv_cache_update(...)` (default implementation provided), passes
   the expanded query to `forward_mqa` as `(ql_nope, q_pe)`, and applies
   `_v_up_proj` itself. Port an old impl by:
   - extending `SparseMLAAttentionImpl` (sparse/decode-only; only
     `forward_mqa` is abstract) or `MLAAttentionImpl`,
   - concatenating the q tuple and dispatching to the old kernel call,
   - returning the *latent* output — the layer up-projects.

2. **`MultiHeadLatentAttention` → `MultiHeadLatentAttentionWrapper`.**
   Same constructor arguments (new optional `skip_topk`), and
   `forward(positions, hidden_states, llama_4_scaling=None)` — the third
   argument is now optional, so two-argument callers keep working.

3. **`get_rope` folds base/rotary_dim/scaling into `rope_parameters`.**
   ⚠ transformers config classes can carry a *class-level default*
   `rope_scaling` (e.g. `GlmMoeDsaConfig`:
   `{'rope_type': 'default', 'rope_theta': 8000000}`) even when the
   checkpoint sets `rope_scaling: null`. `rope_type == "default"` means no
   scaling and must not be forced to `deepseek_yarn` — that branch requires
   yarn fields and raises `KeyError: 'factor'`.

4. **Sparse MLA prefill moved to the `MLAPrefillBackend` registry**
   (`vllm/v1/attention/backends/mla/prefill/`, override via
   `register_mla_prefill_backend`). On P800 no registered backend is
   CUDA-free; see the harness note below for what that means in practice.

5. **Model interface: `embed_input_ids` is now required** by the runner
   support detection (`interfaces_base.is_vllm_model`). Add it to both the
   inner model (`self.embed_tokens(input_ids)`) and the ForCausalLM wrapper
   (delegate), or the registry reports the architecture as not supporting
   `--runner generate`.

6. **FusedMoE fuses shared experts and returns a single tensor.** Pass
   `shared_experts=` at construction; the old `(shared_output, final)`
   tuple unpack crashes (it unpacks the tensor's first dimension). Weight
   loading uses the `fused_moe_make_expert_params_mapping` helper with the
   model as first argument.

7. **Custom ops receive the KV cache tensor, not a list element.**
   `DeepseekV32IndexerCache.kv_cache` starts as an empty `torch.tensor([])`
   and `bind_kv_cache` later assigns the bound tensor. Eager
   `kv_cache[0]` indexing crashes during the pre-allocation profile run;
   pass the attribute directly and let the fake-path guard handle the
   unbound case.

## P800-specific landmines (device, not engine)

- `KunlunPlatform.is_cuda_alike()` returned False while `device_type` is
  `"cuda"` (torch_xmlir maps the XPU onto the torch.cuda API). Generic
  engine code that gates on cuda-alike — `bind_kv_cache`'s multi-layer
  group check, the fp8 quant dispatcher — then rejects or skips the
  platform. Override it to True; the emulated CUDA API is the platform's
  real contract.
- `vllm.envs.VLLM_ATTENTION_BACKEND` and the `vllm.attention.ops.flashmla`
  import were both removed upstream; the plugin's platform file read them
  unconditionally and killed every MLA model at config time.
- The torch.compile act-quant fusion pass references
  `torch.ops._C.silu_and_mul_per_block_quant`, which this stack does not
  register — shape-dynamic MLA decode spans require `--enforce-eager`
  (the same requirement `tools/torch/paged_decode.py` records).
  A missing dispatcher registration should be diagnosed separately from
  shape-dynamic execution constraints. Register the expected schema,
  device implementation and fake implementation before the fusion pass
  imports, then validate dispatch, numerical results and compilation
  independently. Resolving the import does not by itself clear other eager
  requirements. See [operator registration](../vllm-kunlun/op-registration.md)
  for the general procedure; this page does not ship an implementation.
- See `catalog/xpu_specs.yaml#numerical` for the int8-only constraint and
  its evidence; the engine's generic MLA-sparse path (fp8e4m3 + deepgemm)
  cannot run on this device at all.

## Provenance

All entries verified on the cluster during run `glm52-int-w8a8-p800-001`:
plugin worktree `/workspace/vLLM-Kunlun` (v0.25.1-dev, engine commit
`ccb4f0e4f8cf88c534a9ac115cdcac9ba6b7d031`), each fix validated by
reinstalling into `/opt/vllm_kunlun` and re-running the mat-028 toy
bring-up (final: `BRINGUP_PASS reached=DECODE_OK`, TP1 dummy weights) and
the kdp-001b service proof (744B INT8, TP8, 771 s weight load).
