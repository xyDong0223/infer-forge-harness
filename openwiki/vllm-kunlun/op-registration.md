---
type: reference
title: 自定义算子注册与验证清单
summary: >-
  Kunlun 侧自定义算子的注册时机、命名空间判断、schema 兼容层与分层验证顺序；
  页内 `vllm/...` 路径仅作源码导航提示，审计结论以旁挂 claims 为准。
generated:
  by: hand-authored (GitHub Copilot, OpenWiki OKF v0.2 conventions)
  at: 2026-09-17T00:00:00Z
evidence_version:
  repo: https://github.com/baidu/vLLM-Kunlun
  ref: v0.25.1-dev
  commit: c53e090ff8800f586bf9e36e0d876779981bfb20
sources:
- repo://vllm_kunlun/registration/bootstrap.py#L7-L110
- repo://vllm_kunlun/schema.py#L25-L117
- repo://vllm_kunlun/registration/import_hooks.py#L45-L97
- repo://vllm_kunlun/ops/_custom_ops.py#L20-L2863
- repo://vllm_kunlun/ops/_kunlun_ops.py#L22-L47
- repo://vllm_kunlun/ops/attention/layer.py#L223-L240
claims: .claims/op-registration.json
---

# vLLM-Kunlun 自定义算子注册方法

本文说明如何将引擎需要的算子接入 Kunlun 运行时：确认接口、选择实现、
注册 dispatcher 和 fake 实现，再验证实际调用与数值结果。
方法以算子契约为单位，不依赖具体模型。`_C::silu_and_mul_per_block_quant`
仅用于演示接口和注册结构；应用到其他算子时必须重新核对契约。

源码路径以 vLLM 0.25.1 / Kunlun `v0.25.1-dev` 兼容栈为例；本页可审计结论只覆盖
Kunlun 侧已核对的注册机制，`vllm/...` 路径在本文中仅作源码导航提示，其他版本应以
已安装源码重新定锚。本文不提供可直接部署的算子实现，也不要求使用某个历史补丁脚本。

## 1. 区分注册缺失与执行失败

引擎可能在模块导入期读取算子对象，例如：

```python
FUSED_OPS[kFp8Dynamic128Sym] = torch.ops._C.silu_and_mul_per_block_quant.default
```

如果算子尚未注册，导入会出现：

```text
AttributeError: '_OpNamespace' '_C' object has no attribute 'silu_and_mul_per_block_quant'
```

排查时应读取 worker 的原始 traceback，定位首次失败的符号和调用位置。
APIServer 的 `Engine core initialization failed` 只是外层错误。
同时检查平台门控：`is_cuda_alike()` 等条件可能使引擎访问原本未覆盖的算子。

注册缺失、设备实现缺失、fake tracing 失败和数值错误需要分别验证。
注册后导入成功，只证明该符号可以解析；能否启用 compile/cudagraph，
仍需验证相应执行路径，不能仅凭符号存在就移除 eager 约束。

## 2. 注册前确认算子契约

1. **Schema 与调用点**：查引擎 wrapper（例如 `vllm/_custom_ops.py`）和所有
   调用位置，确认参数名、顺序、类型、返回值、可选参数和原地修改行为。
   kwargs 调用要求参数名一致；wrapper 中的分配和断言也是接口证据。
2. **数值语义**：确认独立参考实现、dtype、shape、layout、广播、归一化和
   量化规则。例如分块激活量化可查对应 helion 实现；路由算子应检查对应
   router/reference 的选择与缩放规则。语义缺失时先记录待确认项。
3. **注册时机**：注册必须早于引擎首次解析该算子，并在每个 worker 中生效。
   检查插件 bootstrap 到 `ops/_custom_ops.py` 的实际导入链，避免依赖
   主进程已注册或未经验证的导入顺序。

## 3. 选择接入路径

| 路径 | 适用条件 | 接入方式 |
| --- | --- | --- |
| 厂商门面直调 | `kunlun_ops` 已有符合契约的实现 | 声明引擎要求的 schema，并在设备实现中转发到厂商函数 |
| 已注册算子桥接 | `torch.ops.xspeedgate_ops` 等命名空间已有等价实现 | 对齐参数和输出，通过 dispatcher 桥接到已有算子 |
| Torch 参考实现 | 没有可复用实现，但有明确参考语义 | 注册 Torch 实现，独立验证后用于明确声明的执行路径 |

选择前检查安装版本的算子导出、schema 和实现，不要仅根据函数同名推断等价。
已有 schema 时应先确认缺少的是实现还是注册时机，避免重复定义。

可参考插件中的 `rms_norm_per_block_quant` 门面转发，以及
`registration/bootstrap.py` 中的 `register_weak_ref_tensor` 注册方式；
具体符号和位置以安装版本为准。

## 4. 注册结构示例

以下是接口结构，**不是可运行的数值实现**。设备实现故意抛出异常，
填入经过验证的实现后才能调用。示例沿用 Kunlun 栈的 `CUDA` dispatch key；
其他后端应核对自己的设备分派方式。

```python
import torch
from torch.library import custom_op, impl, register_fake


@custom_op("_C::silu_and_mul_per_block_quant", mutates_args=("output", "scales"))
def silu_and_mul_per_block_quant(
    output: torch.Tensor,
    input: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    scale_ub: torch.Tensor | None,
    is_scale_transposed: bool,
) -> None:
    raise NotImplementedError("A validated device implementation is required")


@impl("_C::silu_and_mul_per_block_quant", "CUDA")
def silu_and_mul_per_block_quant_xpu(
    output: torch.Tensor,
    input: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    scale_ub: torch.Tensor | None,
    is_scale_transposed: bool,
) -> None:
    # 根据安装版本的参考实现补齐计算和原地输出写入。
    raise NotImplementedError("Implement and validate the operator contract")


@register_fake("_C::silu_and_mul_per_block_quant")
def fake_silu_and_mul_per_block_quant(
    output, input, scales, group_size, scale_ub, is_scale_transposed
) -> None:
    # 此例只修改预分配的输出，没有返回值。
    return
```

注册时需要核对：

- **原地修改声明**：`mutates_args` 列出所有被修改的参数，使 tracing 和
  functionalization 能识别副作用。
- **Fake 实现**：表达输出的元数据契约。本例没有返回值；有返回张量的
  算子不能照抄空 fake，需要正确表达 shape、dtype 和 device。
- **别名与布局**：输出可能是非连续张量或转置视图，写回必须符合 wrapper
  的分配方式，不能假定连续布局。
- **可选参数与边界**：核对 `None`、标量、空输入以及 dtype 分支。
  示例中的 `scale_ub` 和转置 scale 布局应对照当前参考源码确认。

## 5. 分层验证与交付

按以下顺序记录证据，前一层通过不能替代后一层：

1. **注册与导入**：在干净 worker 进程中验证 schema 可解析、调用方可导入，
   并确认注册时机不依赖偶然的导入顺序。
2. **设备分派**：用最小输入实际调用，确认进入预期实现，没有静默 fallback。
3. **独立数值对照**：对比参考结果，覆盖需要支持的 shape、dtype、layout、
   可选参数与边界条件，记录误差和容差；加入能检出错误实现的负控制。
4. **编译路径**：需要 compile/cudagraph 时，再验证 fake tracing、捕获和
   执行结果，并与已验证的 eager 路径对照。
5. **调用方回归**：通过真实调用接口验证参数传递、输出写回和组合行为。

实现及修复应保存在版本化源码中，记录适用版本、契约来源、验证命令和结果。
未经验证的 dtype 或执行路径必须明确标注；Torch 参考实现也需要数值验证。
注册形式、可重放性和成功编译都不能替代正确性证据。

## 6. 源码导航

| 要确认的内容 | 参考位置 |
| --- | --- |
| 引擎 wrapper、schema 与输出分配 | `vllm/_custom_ops.py` |
| 导入期算子引用 | `vllm/compilation/passes/fusion/act_quant_fusion.py` |
| 示例算子的数值语义 | `vllm/kernels/helion/ops/silu_and_mul_per_block_quant.py` |
| 插件侧声明与设备实现 | `vllm_kunlun/ops/_custom_ops.py` |
| 插件 bootstrap 与桥接 | `vllm_kunlun/registration/bootstrap.py` |

## 相关页面

- [算子命名空间与接入架构](architecture.md)
- [引擎接口漂移排查](../harness/vllm-0251-drift-map.md)
- [已知兼容性缺口](known-gaps.md)
