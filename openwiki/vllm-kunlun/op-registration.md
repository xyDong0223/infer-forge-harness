# vLLM-Kunlun 缺失 `_C::` 算子的注册方法

> 来源：`step35-flash-p800-002`（2026-09-17）的适配总结与代码快照，
> 基于 harness `999c648`、vLLM 0.25.1、vLLM-Kunlun `ccb4f0e`。
> 本页记录该次运行报告的经验；所引用的 attempt 原始日志未随本文收录。
> 本次文档提交不包含 repair 16/17 的实现，也不表示主线已提供这些修复。
> compile/cudagraph 初始化成功仅是启动证据；该次目标模型仍停在
> `API_SMOKE_FAILED`，不能据此宣称模型数值正确或适配完成。

> 结论先行：引擎（vLLM 0.25.1）在**模块导入期**就会引用一部分
> `torch.ops._C::*` 算子；本 stack 的 `_C` 命名空间由插件
> `ops/_custom_ops.py` 在 Python 侧声明，引擎引用而插件没声明的算子，
> 会在每个 worker 进程里以 `AttributeError` 杀死 torch.compile 后端初始化，
> 表面症状是"必须 `--enforce-eager` 才能起服务"。
> 本文给出注册路径与决策规则，配真实案例
> （run `step35-flash-p800-002`，2026-09-17，`_C::silu_and_mul_per_block_quant`）。

## 1. 症状识别：导入期 AttributeError ≠ 运行期算子缺失

真实案例的失败形态（8 个 TP worker 全部在同一行死掉）：

```text
File ".../vllm/compilation/passes/fusion/act_quant_fusion.py", line 43, in <module>
    FUSED_OPS[kFp8Dynamic128Sym] = torch.ops._C.silu_and_mul_per_block_quant.default
AttributeError: '_OpNamespace' '_C' object has no attribute 'silu_and_mul_per_block_quant'
```

识别要点：

- **发生在 import 阶段**，权重加载之前。该文件的这段赋值被
  `current_platform.is_cuda_alike()` 门控——而插件的平台补丁
  （drift map 里的 `is_cuda_alike → True`）恰好打开了这扇门，
  于是"平台像 CUDA"反而暴露了引擎对 CUDA 专属算子的引用。
- **日志尾部只有 APIServer 侧的包装错误**（
  `RuntimeError: Engine core initialization failed`），
  真正的 `AttributeError` 在前面每个 `Worker_TP*` 的 traceback 里。
  读日志要 grep `AttributeError`，不要只看尾部。
- 一旦算子注册成功，同类错误消失且 `--enforce-eager` 可以移除
  （该案例实测：注册后 compile 正常、51 个 piecewise cudagraph 捕获成功）。

## 2. 注册前必须取证的三件事（不许猜）

1. **引擎期望的 schema**：找引擎侧 Python wrapper（通常在
   `vllm/_custom_ops.py`）和全部调用点（`grep -rn "torch.ops._C.<name>"`
   整个 vllm 包）。确认**参数顺序、类型、哪些参数是输出（mutation）**。
   引擎 wrapper 一般还会自带 assert/分配逻辑，等于免费的 schema 文档。
   注意调用点可能用 kwargs——`@custom_op` 的 Python 形参名必须与之一致。
2. **语义参考实现**：引擎常带 Python 参考内核。本案例的语义全部取自
   `vllm/kernels/helion/ops/silu_and_mul_per_block_quant.py`（helion 内核，
   语义即官方定义）。MoE 路由类取自
   `vllm/model_executor/layers/fused_moe/cpu_fused_moe.py::grouped_topk`。
   **找不到参考就停下来走 diagnosis，不要发明语义。**
3. **执行时机**：算子要在引擎导入那个 pass **之前**注册。插件
   `ops/_custom_ops.py` 在插件激活（每个 worker 进程 import vllm 前的
   bootstrap）时执行 `@custom_op` 声明，天然早于
   `vllm/compilation/passes/*` 的导入。不要在运行期才注册。

## 3. 三条注册路径（按优先级）

| 路径 | 适用条件 | 模板位置 |
| --- | --- | --- |
| A. kunlun_ops 门面直调 | C++ wheel 已有同名实现 | `ops/_custom_ops.py` 里 `rms_norm_per_block_quant`（`#L114-L139`）：`@custom_op` 声明 schema（函数体 `pass`）+ `@impl(..., "CUDA")` 转发 `kunlun_ops.*` |
| B. vendor 轮子桥接 | `torch.ops.xspeedgate_ops` 已有等价算子 | `registration/bootstrap.py` 的 `register_weak_ref_tensor`（`torch.library.Library("_C", "FRAGMENT")` + `define` + `impl` 转发） |
| C. torch 参考实现 | 哪都没有，引擎有 Python 参考语义 | 见下文本案例 |

案例中 `silu_and_mul_per_block_quant` 在 `xspeedgate_ops` 和 `kunlun_ops`
里都不存在（先 `dir(torch.ops.xspeedgate_ops)` / 查门面确认过），
所以走路径 C。

## 4. 路径 C 完整模板（torch 参考实现）

写入 `ops/_custom_ops.py`（文件头已有
`from torch.library import custom_op, impl, register_fake`）：

```python
# 写清三件事：为什么注册（引擎哪个文件在 import 期引用）、
# 语义出处（helion/参考文件路径）、验证状态。
@custom_op("_C::silu_and_mul_per_block_quant", mutates_args=("output", "scales"))
def silu_and_mul_per_block_quant(
    output: torch.Tensor,
    input: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    scale_ub: torch.Tensor | None,
    is_scale_transposed: bool,
) -> None:
    pass


@impl("_C::silu_and_mul_per_block_quant", "CUDA")
def silu_and_mul_per_block_quant_xpu(
    output: torch.Tensor,
    input: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    scale_ub: torch.Tensor | None,
    is_scale_transposed: bool,
) -> None:
    # 以下数学全部来自 helion 参考内核，逐行可对照：
    # x = a*sigmoid(a)*b；分块 amax；scale_ub（0-dim scalar）clamp；
    # s = amax/qmax 且 clamp(min=1/(qmax*512))；
    # int8 round / fp8 直除；结果 clamp 到 qtype 范围写回 output/scales。
    ...


@register_fake("_C::silu_and_mul_per_block_quant")
def _fake_silu_and_mul_per_block_quant(
    output, input, scales, group_size, scale_ub, is_scale_transposed
) -> None:
    return
```

模板要点（每一处都有本案例的实测依据）：

- **`mutates_args` 必须列全**：引擎把预分配的 `output`/`scales` 作为参数传入
  （`vllm/_custom_ops.py` wrapper 负责分配）。漏列会导致 torch.compile 的
  functionalization 出错。
- **`register_fake` 必须给**：piecewise compile 的 fake tracing 会摸到这个算子。
  无返回值的 mutation 算子 fake 体 `return` 即可。
- **schema 参数名要与引擎调用点对齐**：dispatcher 按 Python 形参名生成 schema，
  引擎若用 kwargs 调用（本案例 `act_quant_fusion.py` 用 `.default` OpOverload）
  会做名字校验。
- **转置输出别名写清楚**：`is_scale_transposed=True` 时 `scales` 是
  `.t()` 视图，`scales.copy_(s_ref.t())` 即可正确写回，不必特殊处理。
- **`scale_ub` 是 0-dim scalar tensor**（helion `hl.load(scale_ub, [])` 为证），
  不是逐块向量。

## 5. 注册之后的硬约束

1. **修复必须可重放**：对 site-packages 的改动应保留版本、diff 与验证证据，
   不允许只留在某个 Pod。本案例通过历史 drift 脚本的 exact-anchor repair
   落地；当时环境证明会自动重放该脚本。这是运行时的历史机制，
   不构成必须使用该脚本或恢复自动重放的要求；当前执行遵循仓库协议。
2. **数值边界必须声明**：torch 参考实现是"正确性优先"的落位。**在任何
   量化（FP8/W8A8）服务路径依赖它之前，必须先过独立数值验证**
   （kernel-correctness / kernel-grade 流程 + 负控制）。bf16 路径不会触发
   FP8 fusion 算子，但注册者有义务把这个边界写进代码注释和 drift map。
3. **drift map 同步更新**：`openwiki/harness/vllm-0251-drift-map.md`
   里原有结论（"该 stack 未注册 → 必须 --enforce-eager"）要追加更新记录
   （注册后 import 不再失败、enforce-eager 可移除），并保留旧结论的
   出处——历史结论和现行结论都要在。

## 6. 本案例的完整证据链（未来 agent 可对照）

| 证据 | 位置 |
| --- | --- |
| 导入期失败 traceback（8 worker 同行） | run `step35-flash-p800-002` attempt 000005 `server_crash_log_startup_death.txt` |
| 引擎 wrapper（schema 与分配逻辑） | `vllm/_custom_ops.py#L409-L455` |
| 语义参考（helion 内核） | `vllm/kernels/helion/ops/silu_and_mul_per_block_quant.py#L185-L255`（量化数学）、`#L134-L153`（fake/baseline 签名） |
| 注册落点（drift 补丁 repair 16） | `tools/patches/patch_vllm_kunlun_drift.py`，锚点为 `_custom_ops.py` 的 `static_scaled_fp8_quant` 块 |
| 注册后验证 | attempt 000007+：compile 后端初始化通过、权重加载、health 200 |

## 相关页面

- `architecture.md` §5（算子命名空间总表）
- `../harness/vllm-0251-drift-map.md`（本案例的 drift 结论与更新记录）
- `known-gaps.md`
