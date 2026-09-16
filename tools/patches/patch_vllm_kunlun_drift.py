#!/usr/bin/env python3
"""Re-apply the v0.25.1 drift repairs the vllm_kunlun plugin needs on GLM5.2.

The engine wheel labelled 0.25.1 moved or removed three symbols the pinned
plugin (vllm_kunlun @ ccb4f0e) still imports. A fresh install of the pinned
tree therefore cannot import ``vllm_kunlun.models.deepseek_v2`` at all, which
kills the vLLM registry inspection before any weights load
("Model architectures ['GlmMoeDsaForCausalLM'] failed to be inspected",
run glm52-int-w8a8-p800-001, 2026-09-14).

Repairs (each documented in openwiki/harness/vllm-0251-drift-map.md):
  1. vllm.model_executor.layers.mla.MultiHeadLatentAttention was removed;
     MultiHeadLatentAttentionWrapper has the same constructor signature.
     -> import alias in vllm_kunlun/models/deepseek_v2.py
  2. vllm.v1.attention.backends.mla.indexer.kv_spans_from_batches was removed;
     torch.ops.xspeedgate_ops.kv_spans_from_batches is the exact-equal vendor
     replacement (verified by glm52/verify_kv_spans_pod.py).
     -> local wrapper in vllm_kunlun/v1/attention/backends/mla/indexer.py
  3. split_prefill_chunks(seq_lens, workspace, offset) became
     split_indexer_prefill_chunks(seq_lens, query_lens, workspace,
     max_logits_bytes, request_offset) and now returns (req_slice, query_slice).
     -> adapted call site in kunlun_build

Idempotent: every edit matches its exact old text and skips when already
applied, so re-running after a reinstall is safe.

Applicability is content-based rather than commit-based. A commit allowlist
would reject equivalent cherry-picks and rebuilt wheels, while version strings
are already known to be unreliable for this stack. All anchors are checked and
staged first; no file is written unless the complete patch set matches.

Usage (inside the prepared pod):
    python3 patch_vllm_kunlun_drift.py

Exit codes:
    0  applied or already applied
    1  execution failure
    2  not applicable to this engine/plugin source pair
"""

from __future__ import annotations

import sys
from pathlib import Path

SITE = Path("/opt/vllm_kunlun/lib/python3.10/site-packages")
NOT_APPLICABLE = 2


class PatchTransaction:
    """Stage a complete patch set before changing runtime files."""

    def __init__(self) -> None:
        self._staged: dict[Path, str] = {}
        self._originals: dict[Path, str | None] = {}

    def read(self, path: Path) -> str:
        if path in self._staged:
            return self._staged[path]
        return path.read_text(encoding="utf-8")

    def stage(self, path: Path, content: str) -> bool:
        current = self.read(path) if path.exists() or path in self._staged else None
        if current == content:
            return False
        if path not in self._originals:
            self._originals[path] = current
        self._staged[path] = content
        return True

    def commit(self) -> list[Path]:
        attempted: list[Path] = []
        try:
            for path, content in self._staged.items():
                attempted.append(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
        except OSError as error:
            rollback_errors: list[str] = []
            for path in reversed(attempted):
                try:
                    original = self._originals[path]
                    if original is None:
                        path.unlink(missing_ok=True)
                    else:
                        path.write_text(original, encoding="utf-8")
                except OSError as rollback_error:
                    rollback_errors.append(f"{path}: {rollback_error}")
            if rollback_errors:
                raise RuntimeError(
                    f"patch commit failed ({error}); rollback also failed: "
                    + "; ".join(rollback_errors)
                ) from error
            raise
        return list(self._staged)


def patch(
    path: Path,
    replacements: list[tuple[str, str]],
    *,
    transaction: PatchTransaction | None = None,
) -> bool | None:
    text = transaction.read(path) if transaction else path.read_text(encoding="utf-8")
    changed = False
    for old, new in replacements:
        if new in text:
            continue  # already applied
        if old not in text:
            print(f"FAIL {path.name}: expected text not found:\n{old[:120]}")
            # None, not "unchanged so far": the caller must distinguish
            # "nothing to do" from "the plugin file no longer matches the
            # expected state" — a different exit code lets the replay record
            # this as SKIPPED (non-matching pair) instead of a silent zero
            # that would read as "repair applied".
            return None
        # All occurrences: the drifted call sites repeat verbatim (two
        # get_rope calls in deepseek_v2) and leaving the second one behind
        # just moves the failure to the next layer's init.
        text = text.replace(old, new)
        changed = True
    if changed:
        if transaction:
            transaction.stage(path, text)
        else:
            path.write_text(text, encoding="utf-8")
    if transaction is None:
        print(f"{'PATCHED' if changed else 'SKIP'} {path}")
    return changed


PREFILL_STUB = '''\
# XPU MLA prefill registration for vllm_kunlun.
#
# Per the drift map: "Sparse MLA prefill moved to the MLAPrefillBackend
# registry ... On P800 no registered backend is CUDA-free". The engine
# instantiates an MLA prefill backend in MLAAttention unconditionally, but
# the sparse XPU attention impl handles prefill inside forward_mqa, so the
# MHA prefill path never runs for sparse models. This registration exists
# so construction has a valid target; reaching the run methods means a
# dense-MLA model hit an unsupported path on XPU and must fail loudly
# instead of silently dispatching to CUDA kernels.
from __future__ import annotations

import torch

from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend


class XPUMLAPrefillStub(MLAPrefillBackend):

    @staticmethod
    def get_name() -> str:
        return "XPU_MLA_PREFILL_STUB"

    @classmethod
    def supports_compute_capability(cls, device_capability) -> bool:
        return True

    @classmethod
    def supports_dtype(cls, dtype: torch.dtype) -> bool:
        return dtype in (torch.bfloat16, torch.float16)

    @classmethod
    def is_available(cls) -> bool:
        return True

    def __init__(
        self,
        *,
        num_heads: int,
        scale: float,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        vllm_config=None,
    ) -> None:
        self.num_heads = num_heads
        self.scale = scale
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.vllm_config = vllm_config

    def run_prefill_new_tokens(self, *args, **kwargs):
        raise NotImplementedError(
            "dense MLA prefill is not implemented on XPU; the sparse XPU "
            "MLA impl handles prefill inside forward_mqa"
        )

    def run_prefill_context_chunk(self, *args, **kwargs):
        raise NotImplementedError(
            "dense MLA prefill is not implemented on XPU; the sparse XPU "
            "MLA impl handles prefill inside forward_mqa"
        )
'''


def main() -> int:
    transaction = PatchTransaction()
    deploy_ported = (
        SITE / "vllm_kunlun" / "v1" / "attention" / "backends" / "mla" / "flashmla_sparse.py"
    )
    ported = Path(__file__).parent / "vllm_kunlun_flashmla_sparse.py"
    if ported.exists():
        content = ported.read_text(encoding="utf-8")
        transaction.stage(deploy_ported, content)
    else:
        print(f"WARN {ported} not found; flashmla_sparse port not deployed")
    deepseek_v2 = SITE / "vllm_kunlun" / "models" / "deepseek_v2.py"
    ok1 = patch(deepseek_v2, [
        (
            "from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttention\n",
            "from vllm.model_executor.layers.mla import (\n"
            "    MLAModules,\n"
            "    MultiHeadLatentAttentionWrapper as MultiHeadLatentAttention,\n"
            ")\n",
        ),
        #  4. The engine's ModelConfig now validates the runner interface:
        #     is_text_generation_model requires the VllmModel protocol
        #     (embed_input_ids, compute_logits). The plugin's class predates
        #     that protocol, so the server dies with "This model does not
        #     support `--runner generate`" before any weights load. Port the
        #     two thin methods from the engine's DeepseekV2ForCausalLM.
        (
            "class DeepseekV2ForCausalLM(nn.Module, SupportsPP, MixtureOfExperts, SupportsLoRA):\n"
            "    packed_modules_mapping = {\n"
            '        "gate_up_proj": ["gate_proj", "up_proj"],\n'
            "    }\n",
            "class DeepseekV2ForCausalLM(nn.Module, SupportsPP, MixtureOfExperts, SupportsLoRA):\n"
            "    packed_modules_mapping = {\n"
            '        "gate_up_proj": ["gate_proj", "up_proj"],\n'
            "    }\n"
            "\n"
            "    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:\n"
            "        return self.model.embed_tokens(input_ids)\n"
            "\n"
            "    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:\n"
            "        return self.logits_processor(self.lm_head, hidden_states)\n",
        ),
        #  7. get_rope folded rotary_dim/base/rope_scaling into the single
        #     rope_parameters dict kwarg. The plugin has two verbatim
        #     call sites (MLA module and decoder attention); the patch
        #     replaces both. Per the drift map: rope_type == "default"
        #     (GlmMoeDsaConfig carries a class-level default) must not be
        #     forced to deepseek_yarn — that branch requires yarn fields
        #     and raises KeyError: 'factor'.
        (
            "        if rope_scaling:\n"
            "            rope_scaling[\"rope_type\"] = \"deepseek_yarn\"\n",
            "        if rope_scaling and rope_scaling.get(\"rope_type\", \"default\") != \"default\":\n"
            "            rope_scaling[\"rope_type\"] = \"deepseek_yarn\"\n"
            "        elif rope_scaling:\n"
            "            rope_scaling = None\n",
        ),
        (
            "        if rope_scaling:\n"
            "            mscale_all_dim = rope_scaling.get(\"mscale_all_dim\", False)\n"
            "            scaling_factor = rope_scaling[\"factor\"]\n",
            "        if rope_scaling and rope_scaling.get(\"rope_type\") == \"deepseek_yarn\":\n"
            "            mscale_all_dim = rope_scaling.get(\"mscale_all_dim\", False)\n"
            "            scaling_factor = rope_scaling[\"factor\"]\n",
        ),
        (
            "        self.rotary_emb = get_rope(\n"
            "            qk_rope_head_dim,\n"
            "            rotary_dim=qk_rope_head_dim,\n"
            "            max_position=max_position_embeddings,\n"
            "            base=rope_theta,\n"
            "            rope_scaling=rope_scaling,\n"
            "            is_neox_style=False,\n"
            "        )\n",
            "        rope_parameters = {\n"
            "            \"rope_dim\": qk_rope_head_dim,\n"
            "            \"rope_theta\": rope_theta,\n"
            "            **(rope_scaling or {}),\n"
            "        }\n"
            "        self.rotary_emb = get_rope(\n"
            "            qk_rope_head_dim,\n"
            "            max_position=max_position_embeddings,\n"
            "            rope_parameters=rope_parameters,\n"
            "            is_neox_style=False,\n"
            "        )\n",
        ),
        #  9. The new MultiHeadLatentAttentionWrapper takes the indexer rope
        #     from MLAModules.indexer_rotary_emb; the plugin never built one
        #     because the old MLA layer constructed it internally, so the
        #     Indexer received None and the first profile run died on
        #     "'NoneType' object is not callable". Build it the way the
        #     engine's attention does and pass it through MLAModules.
        (
            "        if self.is_v32:\n"
            "            self.indexer = Indexer(\n",
            "        if self.is_v32:\n"
            "            self.indexer_rope_emb = get_rope(\n"
            "                qk_rope_head_dim,\n"
            "                max_position=max_position_embeddings,\n"
            "                rope_parameters=config.rope_parameters,\n"
            "                is_neox_style=not getattr(config, \"indexer_rope_interleave\", False),\n"
            "            )\n"
            "            self.indexer = Indexer(\n",
        ),
        (
            "        else:\n"
            "            self.indexer = None\n"
            "\n"
            "        mla_modules = MLAModules(\n",
            "        else:\n"
            "            self.indexer_rope_emb = None\n"
            "            self.indexer = None\n"
            "\n"
            "        mla_modules = MLAModules(\n",
        ),
        (
            "            indexer=self.indexer,\n"
            "            is_sparse=self.is_v32,\n",
            "            indexer=self.indexer,\n"
            "            indexer_rotary_emb=self.indexer_rope_emb,\n"
            "            is_sparse=self.is_v32,\n",
        ),
        # 10. DeepseekV32IndexerCache.kv_cache is the tensor itself now (it
        #     starts empty and bind_kv_cache assigns the bound tensor); the
        #     old list-style [0] indexing crashed the pre-allocation profile
        #     run with "index 0 is out of bounds for dimension 0 with size
        #     0". Pass the attribute and let the op's fake-path guard handle
        #     the unbound case.
        (
            "            self.k_cache.kv_cache[0],\n",
            "            self.k_cache.kv_cache,\n",
        ),
        # 11. FusedMoE now fuses the shared-expert add internally
        #     (MoERunner.forward combines shared_output + fused_output
        #     before returning) and returns one tensor, not the old
        #     (shared_output, final_hidden_states) pair; unpacking it
        #     crashed the first profile run with "too many values to
        #     unpack (expected 2)" — one tensor row was read as two
        #     values. shared_experts= at construction is unchanged and
        #     already wires the addition; drop the plugin's own combine.
        (
            "        router_logits, _ = self.gate(hidden_states)\n"
            "        fused_moe_out = self.experts(\n"
            "            hidden_states=hidden_states, router_logits=router_logits\n"
            "        )\n"
            "\n"
            "        if self.shared_experts is not None:\n"
            "            shared_output, final_hidden_states = fused_moe_out\n"
            "        else:\n"
            "            shared_output = None\n"
            "            final_hidden_states = fused_moe_out\n"
            "\n"
            "        # Fix FP16 overflow\n"
            "        # See DeepseekV2DecoderLayer for more details.\n"
            "        if hidden_states.dtype != torch.float16:\n"
            "            final_hidden_states *= self.routed_scaling_factor\n"
            "        elif self.shared_experts is not None:\n"
            "            assert shared_output is not None\n"
            "            shared_output *= 1.0 / self.routed_scaling_factor\n"
            "\n"
            "        if self.shared_experts is not None:\n"
            "            assert shared_output is not None\n"
            "            final_hidden_states += shared_output\n"
            "\n",
            "        router_logits, _ = self.gate(hidden_states)\n"
            "        final_hidden_states = self.experts(\n"
            "            hidden_states=hidden_states, router_logits=router_logits\n"
            "        )\n"
            "\n"
            "        # Fix FP16 overflow\n"
            "        # See DeepseekV2DecoderLayer for more details.\n"
            "        if hidden_states.dtype != torch.float16:\n"
            "            final_hidden_states *= self.routed_scaling_factor\n"
            "\n",
        ),
        # 12. MoERunner all-reduces internally now (the drift map: "FusedMoE
        #     fuses shared experts and returns a single tensor" — the same
        #     change also folded in the old TP all-reduce this method used
        #     to perform); calling it crashed with "'MoERunner' object has
        #     no attribute 'maybe_all_reduce_tensor_model_parallel'". Drop
        #     the now-redundant call.
        (
            "        elif self.tp_size > 1:\n"
            "            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(\n"
            "                final_hidden_states\n"
            "            )\n",
            "",
        ),
        # 14. FusedMoE became a factory function, so its old classmethod
        #     make_expert_params_mapping is gone; weight loading now uses
        #     the fused_moe_make_expert_params_mapping helper with the
        #     model as the first argument (the drift map's item 6, second
        #     half). The toy bring-up passes with dummy weights and never
        #     reaches load_weights; the real 707 GiB load died here.
        (
            "        expert_params_mapping = FusedMoE.make_expert_params_mapping(\n"
            "            ckpt_gate_proj_name=\"gate_proj\",\n"
            "            ckpt_down_proj_name=\"down_proj\",\n"
            "            ckpt_up_proj_name=\"up_proj\",\n"
            "            num_experts=self.config.n_routed_experts,\n"
            "            num_redundant_experts=self.num_redundant_experts,\n"
            "        )\n",
            "        from vllm.model_executor.layers.fused_moe import (\n"
            "            fused_moe_make_expert_params_mapping,\n"
            "        )\n"
            "\n"
            "        expert_params_mapping = fused_moe_make_expert_params_mapping(\n"
            "            self,\n"
            "            ckpt_gate_proj_name=\"gate_proj\",\n"
            "            ckpt_down_proj_name=\"down_proj\",\n"
            "            ckpt_up_proj_name=\"up_proj\",\n"
            "            num_experts=self.config.n_routed_experts,\n"
            "            num_redundant_experts=self.num_redundant_experts,\n"
            "        )\n",
        ),
    ], transaction=transaction)
    indexer = SITE / "vllm_kunlun" / "v1" / "attention" / "backends" / "mla" / "indexer.py"
    ok2 = patch(indexer, [
        # -- import block: engine symbols that moved ------------------------
        (
            "from vllm.v1.attention.backends.mla.indexer import (\n"
            "    DeepseekV32IndexerMetadataBuilder,\n"
            "    kv_spans_from_batches,\n"
            "    split_prefill_chunks,\n"
            ")\n",
            "from vllm import envs\n"
            "from vllm.v1.attention.backends.mla.indexer import (\n"
            "    DeepseekV32IndexerMetadataBuilder,\n"
            "    split_indexer_prefill_chunks,\n"
            ")\n",
        ),
        # 15. The pinned xspeedgate_ops (1.5.1+87067b3) does not register
        #     kv_spans_from_batches — the old pod ran an upgraded wheel — so
        #     the first real request crashed every worker with "'_OpNamespace'
        #     'xspeedgate_ops' object has no attribute 'kv_spans_from_batches'"
        #     (health stays 200; only the first forward dies). Reversible
        #     fallback: call the vendor op when present, else the torch
        #     reference proven exact-equal against it (verify_kv_spans_pod.py).
        #     Recorded as operator debt: the vendor op should be the eventual
        #     path once the upgraded wheel is pinned.
        (
            "def kv_spans_from_batches(start_seq_loc, seq_len_per_batch, device):\n"
            "    # The engine removed this helper; the vendor op is the\n"
            "    # exact-equal replacement (verify_kv_spans_pod.py).\n"
            "    starts = start_seq_loc.to(torch.int64).to(device).contiguous()\n"
            "    lens = seq_len_per_batch.to(torch.int64).to(device).contiguous()\n"
            "    return torch.ops.xspeedgate_ops.kv_spans_from_batches(starts, lens)\n",
            "def kv_spans_from_batches(start_seq_loc, seq_len_per_batch, device):\n"
            "    # The engine removed this helper. The pinned xspeedgate_ops does\n"
            "    # not register the vendor op, so use the torch reference\n"
            "    # (exact-equal: verify_kv_spans_pod.py) when the op is absent.\n"
            "    try:\n"
            "        op = torch.ops.xspeedgate_ops.kv_spans_from_batches\n"
            "    except AttributeError:\n"
            "        op = None\n"
            "    if op is not None:\n"
            "        starts = start_seq_loc.to(torch.int64).to(device).contiguous()\n"
            "        lens = seq_len_per_batch.to(torch.int64).to(device).contiguous()\n"
            "        return op(starts, lens)\n"
            "    query_start_loc = start_seq_loc.to(torch.long).cpu()\n"
            "    seq_lens = seq_len_per_batch.to(torch.long).cpu()\n"
            "    num_reqs = seq_lens.numel()\n"
            "    query_counts = query_start_loc[1:] - query_start_loc[:-1]\n"
            "    num_tokens = int(query_start_loc[-1].item())\n"
            "    kv_starts_per_batch = torch.cumsum(seq_lens, dim=0) - seq_lens\n"
            "    batch_id = torch.repeat_interleave(torch.arange(num_reqs), query_counts)\n"
            "    row_starts = kv_starts_per_batch[batch_id]\n"
            "    pos_within_query = (\n"
            "        torch.arange(num_tokens)\n"
            "        - torch.repeat_interleave(query_start_loc[:-1], query_counts)\n"
            "        + 1\n"
            "    )\n"
            "    context_len = torch.repeat_interleave(seq_lens - query_counts, query_counts)\n"
            "    row_ends = row_starts + context_len + pos_within_query\n"
            "    return (\n"
            "        row_starts.int().to(device),\n"
            "        row_ends.int().to(device),\n"
            "    )\n",
        ),
        # -- local kv_spans_from_batches backed by the vendor op ------------
        (
            "from vllm.v1.attention.backends.utils import (\n"
            "    CommonAttentionMetadata,\n"
            "    split_decodes_and_prefills,\n"
            ")\n",
            "from vllm.v1.attention.backends.utils import (\n"
            "    CommonAttentionMetadata,\n"
            "    split_decodes_and_prefills,\n"
            ")\n"
            "\n"
            "\n"
            "def kv_spans_from_batches(start_seq_loc, seq_len_per_batch, device):\n"
            "    # The engine removed this helper; the vendor op is the\n"
            "    # exact-equal replacement (verify_kv_spans_pod.py).\n"
            "    starts = start_seq_loc.to(torch.int64).to(device).contiguous()\n"
            "    lens = seq_len_per_batch.to(torch.int64).to(device).contiguous()\n"
            "    return torch.ops.xspeedgate_ops.kv_spans_from_batches(starts, lens)\n",
        ),
        # -- call site: new signature, caller now slices the prefill window --
        (
            "        chunk_seq_ids = split_prefill_chunks(\n"
            "            common_attn_metadata.seq_lens_cpu,\n"
            "            self.max_prefill_buffer_size,\n"
            "            num_decodes,\n"
            "        )\n",
            "        prefill_query_lens_cpu = torch.diff(\n"
            "            query_start_loc_cpu[\n"
            "                num_decodes : num_decodes + num_prefills + 1\n"
            "            ]\n"
            "        )\n"
            "        chunk_seq_ids = split_indexer_prefill_chunks(\n"
            "            common_attn_metadata.seq_lens_cpu[num_decodes:],\n"
            "            prefill_query_lens_cpu,\n"
            "            self.max_prefill_buffer_size,\n"
            "            envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024,\n"
            "            request_offset=num_decodes,\n"
            "        )\n",
        ),
        (
            "        chunks = [\n"
            "            self.build_one_prefill_chunk(\n"
            "                reqs_start,\n"
            "                reqs_end,\n"
            "                query_start_loc_cpu,\n",
            "        chunks = [\n"
            "            self.build_one_prefill_chunk(\n"
            "                req_slice.start,\n"
            "                req_slice.stop,\n"
            "                query_start_loc_cpu,\n",
        ),
        # the comprehension binds (reqs_start, reqs_end); the new chunks are
        # (req_slice, query_slice) pairs.
        (
            "            for reqs_start, reqs_end in chunk_seq_ids\n",
            "            for req_slice, _query_slice in chunk_seq_ids\n",
        ),
    ], transaction=transaction)

    #  5. The engine removed VLLM_ATTENTION_BACKEND from vllm.envs entirely;
    #     the platform's FlashMLA default check died with AttributeError at
    #     VllmConfig creation (before any weights load). Read the raw
    #     variable so the unset default still selects FlashMLA.
    kunlun_platform = SITE / "vllm_kunlun" / "platforms" / "kunlun.py"
    #  8. The engine instantiates an MLA prefill backend in MLAAttention
    #     unconditionally and no engine-registered backend is
    #     XPU-compatible ("No valid MLA prefill backend found ...").
    #     The sparse XPU impl handles prefill inside forward_mqa, so the
    #     plugin registers the prefill stub (prefill_xpu.py) under CUSTOM
    #     in _run_startup_stages — which runs at plugin activation in every
    #     process, because worker processes unpickle VllmConfig and never
    #     re-run the platform check — and the platform then selects CUSTOM.
    plugin_init = SITE / "vllm_kunlun" / "__init__.py"
    prefill_stub = (
        SITE / "vllm_kunlun" / "v1" / "attention" / "backends" / "mla" / "prefill_xpu.py"
    )
    ok4 = True
    transaction.stage(prefill_stub, PREFILL_STUB)
    ok5 = patch(plugin_init, [
        (
            "    # 7. Add torch_xmlir's missing memory-info API.\n"
            "    bootstrap.patch_memory_info(logger)\n",
            "    # 7. Add torch_xmlir's missing memory-info API.\n"
            "    bootstrap.patch_memory_info(logger)\n"
            "    # 8. Register the XPU MLA prefill stub for the engine's\n"
            "    #    prefill registry: sparse XPU MLA handles prefill inside\n"
            "    #    forward_mqa, the stub makes MLAAttention construction\n"
            "    #    valid on XPU and fails loudly if the MHA prefill path is\n"
            "    #    ever reached. Runs at plugin activation so every worker\n"
            "    #    process sees the registration.\n"
            "    from vllm.v1.attention.backends.mla.prefill.registry import (\n"
            "        MLAPrefillBackendEnum,\n"
            "        register_mla_prefill_backend,\n"
            "    )\n"
            "\n"
            "    register_mla_prefill_backend(\n"
            "        MLAPrefillBackendEnum.CUSTOM,\n"
            "        \"vllm_kunlun.v1.attention.backends.mla.prefill_xpu.\"\n"
            "        \"XPUMLAPrefillStub\",\n"
            "    )\n",
        ),
    ], transaction=transaction)
    ok3 = patch(kunlun_platform, [
        # 13. Per the drift map's P800 landmine note: KunlunPlatform is
        #     PlatformEnum.OOT, so is_cuda_alike() (gated on
        #     CUDA/ROCM only) returns False even though torch_xmlir maps the
        #     XPU onto the torch.cuda API end to end. Generic engine code
        #     that branches on cuda-alike then takes the "else" path meant
        #     for genuinely exotic backends: bind_kv_cache's multi-layer
        #     group check raised bare NotImplementedError on every TP rank
        #     during KV-cache initialization (the first place two attention
        #     modules per layer — MLA + indexer — share a layer_index).
        #     Override it to True; the emulated CUDA API is this platform's
        #     real contract.
        (
            "    def is_cuda_alike(self) -> bool:\n"
            '        """Stateless version of [torch.cuda.is_available][]."""\n'
            "        return self._enum in (PlatformEnum.CUDA, PlatformEnum.ROCM)\n",
            "    def is_cuda_alike(self) -> bool:\n"
            '        """Stateless version of [torch.cuda.is_available][].\n'
            "\n"
            "        torch_xmlir maps the XPU onto the torch.cuda API end to end, so\n"
            "        this platform's real contract is CUDA-alike even though its\n"
            "        PlatformEnum is OOT, not CUDA/ROCM.\n"
            '        """\n'
            "        return True\n",
        ),
        (
            "import psutil\nimport torch\nimport vllm.envs as envs\n",
            "import os\n\nimport psutil\nimport torch\nimport vllm.envs as envs\n",
        ),
        (
            "            use_flashmla = (\n"
            "                envs.VLLM_ATTENTION_BACKEND is None\n"
            "                or envs.VLLM_ATTENTION_BACKEND == \"FLASHMLA\"\n"
            "            )\n",
            "            attention_backend = os.getenv(\"VLLM_ATTENTION_BACKEND\")\n"
            "            use_flashmla = (\n"
            "                attention_backend is None\n"
            "                or attention_backend == \"FLASHMLA\"\n"
            "            )\n",
        ),
        #  6. vllm.attention.ops.flashmla moved to
        #     vllm.v1.attention.ops.flashmla and is_flashmla_supported was
        #     split into is_flashmla_dense_supported / is_flashmla_sparse_supported.
        (
            "            from vllm.attention.ops.flashmla import is_flashmla_supported\n"
            "\n"
            "            if (\n"
            "                use_flashmla\n"
            "                and is_flashmla_supported()[0]\n"
            "                and cache_config.block_size != 64\n"
            "            ):\n",
            "            from vllm.v1.attention.ops.flashmla import (\n"
            "                is_flashmla_dense_supported,\n"
            "            )\n"
            "\n"
            "            if (\n"
            "                use_flashmla\n"
            "                and is_flashmla_dense_supported()[0]\n"
            "                and cache_config.block_size != 64\n"
            "            ):\n",
        ),
        #  8. The engine instantiates an MLA prefill backend in MLAAttention
        #     unconditionally and no engine-registered backend is
        #     XPU-compatible ("No valid MLA prefill backend found ...").
        #     The sparse XPU impl handles prefill inside forward_mqa, so the
        #     plugin registers the prefill stub (prefill_xpu.py) under CUSTOM
        #     and the platform selects it; the stub raises loudly if the MHA
        #     prefill path is ever reached.
        (
            "                logger.info(\n"
            '                    "Forcing kv cache block size to 64 for FlashMLASparse " "backend."\n'
            "                )\n"
            "\n"
            "        from vllm.config import CUDAGraphMode\n",
            # The deployed, serving-validated variant of repair 8: register
            # the stub here AND select CUSTOM when attention_config exists.
            # An earlier new-text (enum import + bare selection) never
            # matched the file the run actually deployed; the replay's
            # honest exit code caught it on 2026-09-15.
            "                logger.info(\n"
            '                    "Forcing kv cache block size to 64 for FlashMLASparse " "backend."\n'
            "                )\n"
            "\n"
            "        from vllm.v1.attention.backends.mla.prefill.registry import (\n"
            "            MLAPrefillBackendEnum,\n"
            "            register_mla_prefill_backend,\n"
            "        )\n"
            "        register_mla_prefill_backend(\n"
            "            MLAPrefillBackendEnum.CUSTOM,\n"
            '            "vllm_kunlun.v1.attention.backends.mla.prefill_xpu."\n'
            '            "XPUMLAPrefillStub",\n'
            "        )\n"
            "        if vllm_config.attention_config is not None:\n"
            "            vllm_config.attention_config.mla_prefill_backend = (\n"
            "                MLAPrefillBackendEnum.CUSTOM\n"
            "            )\n"
            "\n"
            "        from vllm.config import CUDAGraphMode\n",
        ),
    ], transaction=transaction)

    results = [ok1, ok2, ok3, ok5]
    if any(result is None for result in results):
        print("PATCH NOT APPLICABLE: the plugin tree does not match the "
              "expected anchors; no runtime files were changed")
        return NOT_APPLICABLE
    changed_paths = transaction.commit()
    for path in changed_paths:
        print(f"PATCHED {path}")
    if not changed_paths:
        print("nothing to do: all repairs already applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
