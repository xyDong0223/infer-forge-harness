---
type: experience
title: attention-mla：MLA 族层协议与逐层对齐
summary: DeepSeekV2 风格 MLA（含 DSA indexer）的层协议、权重布局约定与逐层对齐方法。
first_seen:
  model: GLM5.2-Int-W8A8
  run: glm52-int-w8a8-p800-001
  family: DeepSeekV2-style MLA（DeepSeek V2/V3、GLM5.2+）
sources:
- repo://tools/probe/layer0_golden.py
- repo://tools/probe/layer_swap.py
---

# attention-mla

## 问题类别

DeepSeekV2 风格的 MLA 层（q/kv lora 投影、q_a/kv_a layernorm、rope 共享 k_pe）
加上 DSA indexer/topk 稀疏路径。同族模型（DeepSeek V2/V3、GLM5.2）共享同一套
布局约定，换模型不换约定。

## 已验证的约定（踩过的坑）

### deferred-residual 层协议（最高频坑）

- `forward(positions, hidden_states, residual)` — **args[0] 是 positions**，hidden
  在 args[1]；输出是 `(hidden, residual)` 二元组。
- **有效输出 = hidden + residual**。只看 hook 的返回值第一项会得到错误的范数和
  错误的结论（见 layer_swap 的 effective_out 处理）。
- hook 捕获用 `captured[f"{name}.in"] = args[1]`，对比时用
  `out[0] + out[1]`。

### 布局细节（从 checkpoint 反推验证）

- `kv_a_layernorm` 只对 512 维 latent 部分 norm，不含 rope 64 维（per DeepseekV2）。
- `kv_a_proj_with_mqa` 输出切 `[latent 512 | rope 64]`；`q_b_proj` per head 是
  `[nope 192 | rope 64]`。
- rope：`is_neox_style=False` → **interleaved even/odd pair 旋转**，位置用
  `arange(T)`；θ=8e6 量级。
- 跨层 kv 共享：部分层可能没有 `kv_b_proj`，取拥有它的最低层并记录
  `kv_shared_from_layer`。
- attention score 除以 `sqrt(192)`（nope 维数），不是 head_dim 全长。

### DSA indexer

- 插件路径 `v1/attention/backends/mla/indexer.py` + `flashmla_sparse.py`，是
  drift 预检的 hard-gate 面（见 [engine-drift](engine-drift.md)）。

## 工具

| 工具 | 用途 |
|---|---|
| `tools/probe/layer0_golden.py` | CPU fp32 直算第 0 层（不经 transformers），逐 stage dump .pt，含扰动控制 |
| `tools/probe/layer_swap.py` | build（真层+dummy MoE 缩小模型）/ capture（in-process hook）/ compare（含 tokens 协议守卫） |
| `tools/probe/layer0_xpu.py` | 早期版本的 on-disk 换层捕获（layer_swap 前身，保留供对照） |

## 证据入口

- 逐 stage .pt dump 与服务侧 trace 的 stage-by-stage diff：mat-008/mat-013 的
  probe 输出。
- 精度二分入口：`layer_swap build --real-layers N`（GLM run 的 layer-3 MoE
  嫌疑即此法）。

## 未决

- GLM5.2 layer-3 MoE 嫌疑（垃圾输出 bisect）暂停中，结论落点在本轴或 [moe](moe.md)。
