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

Usage (inside the prepared pod):
    python3 patch_vllm_kunlun_drift.py
"""

from __future__ import annotations

import sys
from pathlib import Path

SITE = Path("/opt/vllm_kunlun/lib/python3.10/site-packages")


def patch(path: Path, replacements: list[tuple[str, str]]) -> bool:
    text = path.read_text(encoding="utf-8")
    changed = False
    for old, new in replacements:
        if new in text:
            continue  # already applied
        if old not in text:
            print(f"FAIL {path.name}: expected text not found:\n{old[:120]}")
            return changed
        text = text.replace(old, new, 1)
        changed = True
    if changed:
        path.write_text(text, encoding="utf-8")
    print(f"{'PATCHED' if changed else 'SKIP'} {path}")
    return changed


def main() -> int:
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
    ])

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
    ])

    if not (ok1 or ok2):
        print("nothing to do: all repairs already applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
