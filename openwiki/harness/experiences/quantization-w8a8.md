---
type: experience
title: quantization-w8a8：scale 布局与门禁度量
summary: W8A8 INT8 动态量化的 scale 布局约定，以及为什么 cosine 不能做门禁。
first_seen:
  model: GLM5.2-Int-W8A8 / MiniMax M2.5 W8A8
sources:
- repo://tools/probe/quantized_linear_probe.py
- repo://tasks/mat-008-capability-evaluation/task.yaml
---

# quantization-w8a8

## 问题类别

W8A8（int8 权重 + 动态激活量化，per-out-channel weight scale / per-token
activation scale）的线性层。compressed-tensors 键约定与数值验证方法。

## 已验证的约定（踩过的坑）

### 键布局

- scale 键平铺：`X.weight` 旁边就是 `X.weight_scale`，per-out-channel。
- scale 形状可能是 `[out_features]` 或 `[out_features, 1]`——两种都要处理
  （layer0_golden 的第一个坑）。
- embed_tokens 可能不量化（fp 直存），也可能是 int8+scale：读的时候按有无
  scale 键分支。

### cosine 相似度不能当门禁（惨痛教训）

- cosine 对**均匀 per-channel 因子不敏感**——而 W8A8 的 scale-to-max 转换错误
  恰好就是一个均匀因子：错答案曾打出 **0.9999999**。
- **门禁必须用 relative_l2**（量级在门里），mat-008 已写死
  `forbid_cosine_only_gate: true`。
- 负控制：故意去掉 scale-to-max 转换再算一遍，确认门禁能红。

### 反量化参考

- CPU fp32 参考直接从 safetensors 反量化（`w_fp32 = w_int8.to(float32) * scale`），
  不经任何 runtime 的量化层——参考的独立性才成立。

## 工具与证据

| 工具 | 用途 |
|---|---|
| `tools/probe/quantized_linear_probe.py` | mat-008 quantization 维度 probe（含负控制） |
| `tools/probe/layer0_golden.py` | 整层反量化 + 逐 stage dump |

- probe 几何用**非整除 tile 的 token 数**：整块几何会藏住 offset 类错误。
