---
type: experience
title: attention-block-sparse：MSA 三算子与 index cache 写入
summary: MiniMax-M3 Sparse Attention 的三个 vendor 算子、cache 布局分歧与写后回读验证法。
first_seen:
  model: MiniMax-M3
  findings: tasks/mat-020-vendor-handoff/minimax-m3-findings.yaml
sources:
- repo://tools/probe/block_sparse_attention_probe.py
- repo://tools/probe/qknorm_rope_insert_probe.py
- repo://tools/probe/qknorm_rope_probe.py
---

# attention-block-sparse

## 问题类别

MSA（**MiniMax-M3 Sparse Attention**）的 block-sparse 路径：选块 → topk 变换 →
稀疏 attention 三个 vendor 算子，加上 index cache 的写入侧。

## 已验证的约定（踩过的坑）

### 三个算子、两个维度

- 读侧（mat-008 `block_sparse` 维度）：`kunlun_ops::msa_block_score`、
  `msa_block_score_topk_transform`、`msa_sparse_attention`，孤立验证 + 合成 cache。
- 写侧（`fused_qknorm_rope_insert` 维度）：`_C::fused_minimax_m3_qknorm_rope_kv_insert`
  一个 op 做三件事（norm、RoPE、scatter 进两个 paged cache）——**写错了服务照样
  出流畅的错文本**，所以必须写后回读（readback_at_the_written_slots），不能信
  "write 成功"。

### 已知 vendor 约束

- `msa_block_score` 只接受 `head_num_kv == 1`，其他值返回错误。
- `msa_block_score` 在 `score_type=0` 时可选 attention 输出保持全零（合法，不是 bug）。

### cache 布局分歧（真实且属于模型代码）

- 插件平台布局：`(2, num_blocks, num_kv_heads, block_size, head_size)`
  （`vllm_kunlun/ops/paged_attn.py`），与 `msa_*` kernel 一致。
- 上游 M3 假设 `(num_blocks, 2, 128, num_kv_heads, head_dim)`。
- 两者不等价，分歧归模型代码处理，不归 cache。

## 工具与证据

| 工具 | 用途 |
|---|---|
| `tools/probe/block_sparse_attention_probe.py` | 三算子孤立验证 + 合成 cache + 负控制 |
| `tools/probe/qknorm_rope_insert_probe.py` | 写侧：回读验证 + 逐分支 float32 参考 + 去 RoPE 控制 |
| `tools/probe/qknorm_rope_probe.py` | pod 侧实现 stand-in（probe 评分部署将加载的同一文件） |

- 写侧探针用散乱 slot mapping 的几何（token 数不整除 block_size），整块几何会
  藏住 offset 错误。
