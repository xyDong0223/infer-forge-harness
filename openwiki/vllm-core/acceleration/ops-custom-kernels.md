---
type: concept
title: 自定义算子注册与设备分发
generated:
  by: comate-agent/1.0
  at: 2026-09-09
verified:
  by: comate-agent/1.0
  at: 2026-09-09
sources:
  - repo://vllm/_custom_ops.py
  - repo://vllm/_xpu_ops.py
  - repo://vllm/utils/torch_utils.py
  - repo://vllm/model_executor/custom_op.py
  - repo://vllm/model_executor/layers/activation.py
  - repo://vllm/model_executor/layers/layernorm.py
  - repo://vllm/model_executor/layers/rotary_embedding/base.py
related:
  - "[量化接入](quantization-integration.md)"
  - "[Platform 抽象层](../platform/platform-abstraction.md)"
---

# 自定义算子注册与设备分发

vLLM 的算子体系有两层：**kernel 导入层**（平台决定加载哪个 C++ 扩展）和 **Python 分发层**（`CustomOp` 按 `current_platform` 绑定 forward）。

## 1. kernel 导入：`Platform.import_kernels()`

模块 `vllm/_custom_ops.py` 被 import 时第一件事是 `current_platform.import_kernels()`（@L20）。各平台的覆写：

| 平台 | 导入内容 | 位置 |
|---|---|---|
| 基类 | `vllm._C`（失败仅告警）+ 可选 `_moe_C_stable_libtorch` | `repo://vllm/platforms/interface.py#L362-L370` |
| CUDA | `_C_stable_libtorch`、`_moe_C_stable_libtorch`、可选 `_qutlass_C` | `repo://vllm/platforms/cuda.py#L229-L239` |
| ROCm | `super().import_kernels()` + `vllm._rocm_C` | `repo://vllm/platforms/rocm.py#L539-L548` |
| XPU | **刻意不导入 `vllm._C`**，只导入 `vllm._moe_C` | `repo://vllm/platforms/xpu.py#L132-L136` |
| CPU | 按 ISA 三分：AVX512_BF16→`vllm._C`，AVX512→`_C_AVX512`，否则 `_C_AVX2`（都注册到 `torch.ops._C`，Python 层不感知差异） | `repo://vllm/platforms/cpu.py#L552-L574` |

**模式**：所有平台差异收敛到 `import_kernels` 这一处；OOT 平台覆写它导入自有扩展即可。

## 2. Python 包装层与门控（`vllm/_custom_ops.py`）

- 绝大多数 op 是 `torch.ops._C.*` 的 Python 包装（如 `rotary_embedding` @L211-L235、`rms_norm` @L239-L245、`fused_add_rms_norm` @L322-L329）；ROCm 专用走 `torch.ops._rocm_C.*`（如 `paged_attention_rocm` @L105-L148）。
- **旧机制 `Platform.use_custom_ops` 已删除**。现在的门控是两级：
  1. **编译产物存在性**：`if hasattr(torch.ops._C, "awq_dequantize"):` 才注册 `register_fake`（meta 实现，供 torch.compile）——见 @L91-L101、@L561-L575；
  2. **运行期平台/能力检查**：如 `cutlass_scaled_mm` 内部 ROCm 回退（@L820-L828）；`scaled_int8_quant` 对 XPU 直接走 torch 参考实现（@L2008-L2023）；FP4 系列入口 `assert not current_platform.is_rocm()`（@L1557 等）。
- 能力门控：`cutlass_scaled_mm_supports_fp8(...)` 等接收 `cuda_device_capability` 转发给 `torch.ops._C`；能力查询来自 `Platform.get_device_capability`（`repo://vllm/platforms/interface.py#L419-L490`）。

## 3. `direct_register_custom_op`（`repo://vllm/utils/torch_utils.py#L1035-L1074`）

绕开 `torch.library.custom_op` 的 dispatch 开销的直接注册：

```python
vllm_lib = Library("vllm", "FRAGMENT")                      # @L1032
def direct_register_custom_op(op_name, op_func, ...):
    dispatch_key = dispatch_key or current_platform.dispatch_key   # @L1062-L1065
    schema_str = torch.library.Library.infer_schema(op_func)       # @L1067
    my_lib.define(op_name + schema_str, tags=tags)                 # @L1069
    my_lib.impl(op_name, op_func, dispatch_key=dispatch_key)       # @L1070
    my_lib._register_fake(op_name, fake_impl)                      # @L1071 可选
```

使用范例：XPU 扩展算子批量注册 `vllm/_xpu_ops.py@L1231-L1324`（`xpu_ops.register_ops_once()`，尾部 @L1324 调用、`_OPS_REGISTERED` 防重复）；ROCm AITER（`vllm/_aiter_ops.py@L2141-L2197`）；FlashInfer 包装（`vllm/utils/flashinfer.py@L745`）。

## 4. `CustomOp`：构造期设备分发（`repo://vllm/model_executor/custom_op.py`）

- 注册表：`op_registry` / `op_registry_oot` 两个全局 dict（@L21-L22）。
- `CustomOp` 基类 @L103：
  - `__init__` @L130-L133：`self._forward_method = self.dispatch_forward()` ——**构造期一次性绑定**，forward 直接调（@L135-L136），运行期零分支。
  - 默认实现层级：`forward_native` @L138（纯 PyTorch，可被 compile）；`forward_cuda` @L146（子类实现）；`forward_hip` @L149 回落 `forward_cuda`；`forward_xpu/cpu/tpu/oot` @L153-L172 默认回落 `forward_native`。
  - `dispatch_forward` @L174-L207：注释明确"假设只为单一后端构建，不支持动态 dispatch"。顺序：`enabled()` 不过 → native；否则按 `is_rocm/is_cpu/is_tpu/is_xpu/is_out_of_tree` 依次绑定，否则 `forward_cuda`。
  - `enabled()` @L271-L293：依据 `CompilationConfig.custom_ops` 的 `+op_name/-op_name`。
- **OOT 整类替换**：`@CustomOp.register_oot(name=...)` @L329-L360 把树外子类登记进 `op_registry_oot`；`CustomOp.__new__` @L109-L128 实例化时自动替换。官方示例：HPU 用 `@UnquantizedFusedMoEMethod.register_oot` 提供 `HPUUnquantizedFusedMoEMethod`（@L333-L337）。`PluggableLayer` @L32-L100 是有子模块组合的层的等价物。

## 5. 典型层的分发模式

| 层 | forward_xpu 策略 | 位置 |
|---|---|---|
| `SiluAndMul` | 委托 `forward_cuda`（复用 `_C` 同名 kernel） | `repo://vllm/model_executor/layers/activation.py#L149-L150` |
| `SiluAndMulWithClamp` | 回落 `forward_native` | @L252-L253 |
| `RMSNorm` | 委托 `forward_cuda`；而 CUDA 主路径已改走 **vLLM IR 层**（`vllm.ir.ops.rms_norm`，`repo://vllm/ir/ops/layernorm.py#L9-L66`，经 `vllm/ir/op.py@L106-L120` 的 `register_op` 包成 `vllm_ir` torch library） | `repo://vllm/model_executor/layers/layernorm.py#L117-L122` |
| `GemmaRMSNorm` | 独立 kernel：`xpu_gemma_rms_norm`（`vllm/_xpu_ops.py@L1236-L1247` 注册，仅当宿主有 `_C.gemma_rms_norm`），否则回落 native | layernorm.py@L169-L182 |
| 普通 `LayerNorm` | 不参与 CustomOp（torch 原生 `F.layer_norm` 足够） | layernorm.py@L337 |
| `RotaryEmbeddingBase` | `forward_xpu` @L273-L296：key 为 None 回落 native，否则调 `ops.rotary_embedding`（依赖 `import_kernels` 载入的扩展提供同名 `_C` 算子） | `repo://vllm/model_executor/layers/rotary_embedding/base.py` |
| `ApplyRotaryEmb` | `forward_hip` 针对 HIP grid 上限 65535 检测超限回落 native（@L246-L270）；`forward_cpu` 回落 native | `repo://vllm/model_executor/layers/rotary_embedding/common.py` |

## 6. OOT 后端提供自有 kernel 的四个层次

1. **平台类**：覆写 `import_kernels` 导入自有扩展；设好 `dispatch_key`、`get_device_capability`。
2. **算子级**：仿 `_xpu_ops.py` 用 `direct_register_custom_op` 注册 `vllm::` 命名空间算子，配 `hasattr(torch.ops._C, ...)` 门控 + fake impl。
3. **层/类替换**：`@CustomOp.register_oot` / `@PluggableLayer.register_oot` 整类替换重量级 method（MoE 等）。
4. **量化注册**：`@register_quantization_config("my_quant")`（见 [quantization-integration.md](quantization-integration.md)）。

## 相关页面

- [platform-abstraction.md](../platform/platform-abstraction.md) — `import_kernels` / `dispatch_key` 的宿主
- [quantization-integration.md](quantization-integration.md) — 量化层如何消费平台能力
