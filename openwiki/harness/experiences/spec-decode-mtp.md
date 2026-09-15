---
type: experience
title: spec-decode-mtp：MTP nextn 层的配置与 decode 路由
summary: multi-token prediction（num_nextn_predict_layers）对模型加载和诊断造模的影响。
first_seen:
  model: GLM5.2-Int-W8A8
sources:
- repo://tools/probe/layer_swap.py
- repo://tools/patches/apply_torch_decode_patch.py
---

# spec-decode-mtp

## 问题类别

MTP / speculative decode：模型带 `num_nextn_predict_layers`（nextn 预测层）时的
配置处理、模型 init 差异、decode 路由。

## 已验证的约定（踩过的坑）

### config 处理

- 交换/缩小的诊断模型**没有 nextn 层**，config 里必须 DROP
  `num_nextn_predict_layers`，否则插件按 nextn 结构去加载不存在的权重
  （layer_swap 的 `DROP_CONFIG_KEYS`）。
- 完整模型加载时 nextn 层是 config 事实，扫描阶段如实报告，不猜测。

### decode 路由

- `kunlun_ops.speculative_attention` 是常规 decode 与 speculative 的共入口：
  按 `qlen` 分流，`qlen == 1`（常规 decode）才允许走 torch fallback
  （见 [attention-swa](attention-swa.md#decode-路由)）。
- `KDP_DECODE_KERNEL=speculative` 可在运行时关闭路由、复现 vendor 失败，
  无需卸载 fallback（可逆调试）。

## 未决

- GLM5.2 的 MTP 层本身未被逐层数值验证（精度问题停在 MoE 嫌疑，MTP 未证清白）。
