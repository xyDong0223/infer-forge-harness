---
type: experience
title: moe：W8A8 expert、FusedMoE 工厂参数与 dummy-expert 造模
summary: MoE 层在 P800 上的三个问题面：int8 expert 权重布局、工厂参数映射、缩小模型时的专家伪造。
first_seen:
  model: GLM5.2-Int-W8A8（int8 expert）与 MiniMax-M3（工厂参数映射）
sources:
- repo://tools/probe/layer_swap.py
- repo://tools/patches/patch_vllm_kunlun_drift.py
- repo://tasks/mat-008-capability-evaluation/task.yaml
---

# moe

## 问题类别

MoE 层的三个独立问题面：权重布局（int8 expert + scale 成对）、工厂参数映射
（FusedMoE）、诊断造模（dummy expert）。

## 已验证的约定（踩过的坑）

### W8A8 expert 权重是成对的

- int8 expert 权重与 `weight_scale` 键**平铺相邻**（`X.weight` +
  `X.weight_scale`，不是嵌套结构）。
- 克隆/拷贝 expert 权重时 **weight 和 scale 必须成对复制**：layer_swap 早期版本
  过滤掉 scale 键导致权重无法反量化（自伤 bug，已修）。MoE 键按前缀整体跳过，
  attention 键按对克隆。

### FusedMoE 工厂参数

- 插件 0.25.1 与引擎的 FusedMoE 工厂 helper 参数名有 drift（expert 参数映射），
  修复见 `tools/patches/patch_vllm_kunlun_drift.py`。

### dummy-expert 造模（诊断用）

- 插件 model init **拒绝纯 dense 模型**：swap 模型必须保留至少一个 MoE 层。
- 标准做法：`first_k_dense_replace=1`（只有第 0 层 dense），其余层用
  `n_routed_experts=8` 的 dummy expert（scale 1e-3，成对克隆 donor）。
- `num_nextn_predict_layers`（MTP）必须从 swap config 中 **DROP**——dummy 模型
  没有 nextn 层（见 [spec-decode-mtp](spec-decode-mtp.md)）。
- config 手术（num_hidden_layers、first_k_dense_replace、n_routed_experts、
  DROP keys）必须记录进 manifest，证据可回放。

### 覆盖边界（诚实的未覆盖项）

- mat-008 的 moe 维度只覆盖 tensor-parallel 路径：`fused_moe_ep`（专家并行）
  未被 exercise。
- 该维度用合成 bf16 expert（独立于量化维度）；**int8 MoE 走
  `compressed_tensors_moe.py`，是另一条路径，未覆盖**。

## 工具与证据

| 工具 | 用途 |
|---|---|
| `tools/probe/layer_swap.py` | `build --real-layers N`：二分某层嫌疑的标准入口 |
| `tools/probe/moe_layer_probe.py` | mat-008 moe 维度 probe |
