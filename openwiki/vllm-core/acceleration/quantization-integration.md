---
type: concept
title: 量化接入与平台白名单
generated:
  by: comate-agent/1.0
  at: 2026-09-09
verified:
  by: comate-agent/1.0
  at: 2026-09-09
sources:
  - repo://vllm/model_executor/layers/quantization/__init__.py
  - repo://vllm/config/model.py
  - repo://vllm/platforms/interface.py
  - repo://vllm/platforms/rocm.py
  - repo://vllm/platforms/xpu.py
  - repo://vllm/model_executor/layers/quantization/fp8.py
related:
  - "[自定义算子](ops-custom-kernels.md)"
---

# 量化接入与平台白名单

## 1. 全局量化方法注册表（`vllm/model_executor/layers/quantization/__init__.py`）

- `QuantizationMethods` Literal @L15-L49：`awq, auto_awq, fp8, fbgemm_fp8, fp_quant, modelopt, modelopt_fp4, modelopt_mxfp8, modelopt_mixed, auto_gptq, gptq, gptq_marlin, awq_marlin, humming, compressed-tensors, experts_int8, quark, moe_wna16, torchao, inc, mxfp4, gpt_oss_mxfp4, deepseek_v4_fp8, online` + 在线量化 shorthand（`fp8_per_tensor, fp8_per_block, fp8_per_channel, int8_per_channel_weight_only, nvfp4_per_token, mxfp8`）。
- `QUANTIZATION_METHODS = list(get_args(...))` @L50；`DEPRECATED_QUANTIZATION_METHODS = ["fbgemm_fp8", "fp_quant"]` @L52-L55。
- `get_quantization_config` @L111-L183：校验方法名后**延迟导入**各 Config（避免过早触发 torch.compile，@L115），映射表 @L142-L171；自定义方法最后覆盖 @L181。

## 2. 树外自定义量化注册

`@register_quantization_config(quantization)` 装饰器 @L61-L108：

1. 把方法名追加进全局 `QUANTIZATION_METHODS`（@L96）；
2. **自动追加进 `current_platform.supported_quantization`**（@L97-L99）——绕开平台白名单拦截；
3. 存入 `_CUSTOMIZED_METHOD_TO_QUANT_CONFIG`（@L105），`get_quantization_config` 优先返回它（@L181）。

## 3. 平台白名单与校验链

- 基类字段 `supported_quantization: list[str] = []`（`repo://vllm/platforms/interface.py#L170`），**空 = 不限制**。
- 校验入口 `Platform.verify_quantization` @L983-L991：非空且 quant 不在列表 → `ValueError`。
- 调用链：`ModelConfig._verify_quantization`（`repo://vllm/config/model.py#L1255-L1342`）——先校验方法在全局注册表内（@L1336-L1341），再调 `current_platform.verify_quantization`（@L1342）；同函数还按优先级探测 checkpoint 的 `override_quantization_method`（@L1268-L1322）。
- 平台实例：
  - **ROCm** 白名单 20+ 项（awq/gptq/fp8/mxfp4/quark/modelopt 等，`repo://vllm/platforms/rocm.py#L513-L537`）；并覆写 `verify_quantization` 强制 awq 走 `VLLM_USE_TRITON_AWQ`（@L939-L947）。
  - **XPU** 白名单（awq/gptq/moe_wna16/inc/fp8/mxfp4/mxfp8/online/compressed-tensors，`repo://vllm/platforms/xpu.py#L113-L130`）。
  - CUDA 与基类不设白名单（`[]`），全放行。

## 4. 平台能力如何影响量化实现

- FP8 dtype 变体：`Platform.is_fp8_fnuz()`（AMD MI300/MI325 为 True，`repo://vllm/platforms/rocm.py#L976-L985`）→ 量化层分派 `float8_e4m3fnuz`（`repo://vllm/model_executor/layers/quantization/fp8.py#L757`）；`Platform.fp8_dtype()` @L1114-L1121（XPU 用 e4m3fn）。
- `Platform.supports_fp8()` @L1093-L1098、`supports_mx()` @L1086-L1091。
- 算子级：量化 kernel 走 `torch.ops._C.*`，平台能力（`get_device_capability`）与 `hasattr` 门控见 [ops-custom-kernels.md](ops-custom-kernels.md)。

## 5. OOT 平台的量化接入路径

1. checkpoint 用现有格式 → 只需在 `supported_quantization` 白名单里放行（或留空全放行）。
2. 自有量化格式 → 实现 `QuantizationConfig` 子类 + `@register_quantization_config("my_quant")`（自动进白名单）。
3. 早期注册（`--quantization my_quant` 在 CLI 就出现）→ 在 `Platform.pre_register_and_update(parser)` 中注册（`repo://vllm/platforms/interface.py#L545-L559`）。

## 相关页面

- [ops-custom-kernels.md](ops-custom-kernels.md) — 量化 kernel 的注册与门控
- [new-backend-guide.md](../guides/new-backend-guide.md) — 接入清单中的量化部分
