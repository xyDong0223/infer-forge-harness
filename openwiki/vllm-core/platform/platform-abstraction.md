---
type: concept
title: Platform 抽象层与内置平台
generated:
  by: comate-agent/1.0
  at: 2026-09-09
verified:
  by: comate-agent/1.0
  at: 2026-09-09
sources:
  - repo://vllm/platforms/interface.py
  - repo://vllm/platforms/cuda.py
  - repo://vllm/platforms/rocm.py
  - repo://vllm/platforms/cpu.py
  - repo://vllm/platforms/xpu.py
  - repo://vllm/platforms/zen_cpu.py
  - repo://vllm/platforms/tpu.py
related:
  - "[平台检测](platform-detection.md)"
  - "[新后端接入指南](../guides/new-backend-guide.md)"
---

# Platform 抽象层与内置平台

`vllm/platforms/interface.py`（1341 行）定义了 `Platform` 基类，是所有硬件后端的第一落点。运行期全局单例是 `vllm.platforms.current_platform`（懒加载，见 [platform-detection.md](platform-detection.md)）。

## 1. `Platform` 类的关键属性

| 属性 | 位置 | 说明 |
|---|---|---|
| `_enum: PlatformEnum` | `repo://vllm/platforms/interface.py#L135` | 枚举 `CUDA, ROCM, TPU, XPU, CPU, OOT, UNSPECIFIED`（@L70-L76）；OOT 平台通常取 `PlatformEnum.OOT` |
| `device_name` / `device_type` | @L136-L137 | 日志名 / **PyTorch 设备类型字符串**（必须与 `torch.device(...)` 一致，分布式建组直接用它拼 `f"{device_name}:{idx}"`，`repo://vllm/distributed/parallel_state.py#L535-L538`） |
| `dispatch_key` | @L142 | PyTorch dispatch key，默认 `"CPU"`；未注册进 PyTorch 的平台用它兜底（`direct_register_custom_op` 的缺省 dispatch key） |
| `ray_device_key` | @L147 | Ray 设备 key，空串=不支持 Ray |
| `device_control_env_var` | @L153 | 设备可见性控制变量（CUDA: `CUDA_VISIBLE_DEVICES`，XPU: `ZE_AFFINITY_MASK`，ROCm 也用 `CUDA_VISIBLE_DEVICES`） |
| `ray_noset_device_env_vars` | @L158 | 阻止 Ray 改写可见设备（如 `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES`） |
| `simple_compile_backend` | @L165 | torch.compile 后端，默认 `"inductor"` |
| `dist_backend` | @L168 | torch.distributed 后端：CUDA/ROCm=`nccl`，CPU=`gloo`，XPU=`xccl` |
| `supported_quantization` | @L170 | 量化白名单，空=不限制（见 [quantization-integration.md](../acceleration/quantization-integration.md)） |
| `additional_env_vars` | @L172 | Ray worker 需透传的平台环境变量 |

**属性透传**：`__getattr__`（@L1153-L1172）把未知属性转发到 `torch.<device_type>` 模块——`current_platform.synchronize()` 即 `torch.cuda.synchronize()`。OOT 平台因此可以少实现很多方法，但依赖 PyTorch 有对应 device 模块。

## 2. 平台必须/常覆盖的方法

### 配置钩子（平台生效的核心三件）

| 方法 | 位置 | 时机 |
|---|---|---|
| `pre_register_and_update(parser)` | @L545-L559 | **VllmConfig 初始化与 CLI 解析之前**；OOT 平台在此动态注册量化 config 等 |
| `apply_config_platform_defaults(vllm_config)` | @L561-L573 | CLI 解析后按平台改默认值 |
| `check_and_update_config(vllm_config)` | @L575-L586 | 检查并**就地修改**配置，可抛异常；**必须在此把 `parallel_config.worker_cls` 从 `"auto"` 改为具体类限定名**（`repo://docs/design/plugin_system.md#L105`） |

### 设备与能力

- `get_device_capability()` @L419-L430 / `has_device_capability()` @L432-L454 / `is_device_capability[_family]()` @L456-L493：返回 `DeviceCapability(major, minor)` 或 None；算子门控大量消费它（如 CUDA `supported_dtypes` 依赖 `has_device_capability(80)`，`repo://vllm/platforms/cuda.py#L245-L254`）。
- `get_device_name` @L495、`get_device_uuid` @L500、`get_device_total_memory` @L505、`num_compute_units` @L1310：基类抛 `NotImplementedError`，必须实现。
- 设备 ID 三重命名空间转换（logical / visible / physical）：`device_id_to_physical_device_id` @L284-L312、`logical_device_id_to_visible_device_id` @L314-L338、`visible_device_id_to_physical_device_id` @L340-L360；worker 发布的物理 ID 映射经 `set_assigned_physical_gpu_ids()` @L36-L47 优先消费。

### 注意力与运行时挂钩（其余钩子详见对应页面）

- `get_attn_backend_cls(selected_backend, attn_selector_config, num_heads)` @L372-L380：返回注意力后端类全限定名（[attention-backend.md](../acceleration/attention-backend.md)）。
- `get_device_communicator_cls()` @L1067-L1072：默认 `DeviceCommunicatorBase`（[distributed-comm.md](../runtime/distributed-comm.md)）。
- `import_kernels()` @L362-L370：默认 `import vllm._C` + 可选 `vllm._moe_C_stable_libtorch`；**各平台覆写它导入自有扩展**（[ops-custom-kernels.md](../acceleration/ops-custom-kernels.md)）。
- `verify_quantization` @L983-L991、`verify_model_arch` @L971-L981、`check_if_supports_dtype` @L1204-L1209：校验钩子。
- `supported_dtypes` property @L181-L187：**列表第一个是 `dtype="auto"` 的回退默认**。
- `is_sleep_mode_available()` @L224-L229：CUDA/ROCM/XPU 为 True。
- `inference_mode()` @L523-L531：不支持 `torch.inference_mode` 的平台（TPU）回退 `torch.no_grad`。

## 3. 内置平台清单

本版本 `vllm/platforms/` 只有 8 个文件；**neuron / openvino / hpu(gaudi) 已不在树内**，全部转为 OOT 插件（TPU 的外移由代理文件直接佐证）。

| 文件 | 类 | 要点 |
|---|---|---|
| `cuda.py`（1067 行） | `CudaPlatformBase` @L217、`NvmlCudaPlatform` @L767、`NonNvmlCudaPlatform` @L1017 | `CudaPlatform = NvmlCudaPlatform if nvml_available else NonNvmlCudaPlatform`（@L1064，Jetson 无 NVML 时回退）。NVML 版提供 uuid/NUMA/PCI bus ids。`_get_backend_priorities()` @L83-L173 决定注意力后端优先级。`import_kernels()` @L229-L239 导入 `_C_stable_libtorch`、`_qutlass_C` |
| `rocm.py`（1156 行） | `RocmPlatform` @L498 | **`device_type="cuda"`**（HIP 复用 CUDA dispatch），`dispatch_key="CUDA"`。大量 gfx 架构判断（`on_gfx9/gfx90a/gfx942/gfx12x/...` @L306-L384，基于 amdsmi 查 gcnArch @L178）。`is_fp8_fnuz()` MI300/MI325 为 True @L975-L981。`verify_quantization` 强制 awq 走 `VLLM_USE_TRITON_AWQ` @L939-L947 |
| `cpu.py`（656 行） | `CpuPlatform` @L98 | `dispatch_key="CPU"`、`dist_backend="gloo"`。`import_kernels()` @L552-L574 按 ISA 三分：AVX512_BF16→`vllm._C`，AVX512→`_C_AVX512`，否则 `_C_AVX2`（都注册到 `torch.ops._C` 命名空间，Python 层不感知差异）。`check_and_update_config` @L198-L452 处理线程/NUMA/dtype |
| `xpu.py`（547 行） | `XPUPlatform` @L103 | `dispatch_key="XPU"`、`dist_backend="xccl"`、`device_control_env_var="ZE_AFFINITY_MASK"`。**`import_kernels()` 刻意不导入 `vllm._C`**，只导入 `vllm._moe_C`（@L132-L136）。内置平台中最"薄"的参考实现 |
| `zen_cpu.py`（33 行） | `ZenCpuPlatform(CpuPlatform)` @L12 | AMD Zen + zentorch 专用：`supported_dtypes=[bf16, fp32]`（AMD CPU 无 fp16 计算）@L30-L32；由 CPU 检测函数在 `_is_amd_zen_cpu()` 且 `import zentorch` 成功时切换（`repo://vllm/platforms/__init__.py#L207-L221`） |
| `tpu.py`（21 行） | 无自有类 | **纯代理**：`from tpu_inference.platforms import TpuPlatform`（@L9-L20）。TPU 实现已外移到 `tpu_inference` 包 |
| `interface.py`（1341 行） | `Platform`、`UnspecifiedPlatform` @L1338 | 基类 + 未检测到平台时的兜底 |
| `__init__.py`（340 行） | — | 检测/懒加载核心（见 [platform-detection.md](platform-detection.md)） |

## 4. 与旧版本的关键差异（读旧文档须知）

- `PlatformRegistry` 类已删除（全库 grep 零匹配）——注册表职责由"**字符串限定名 + `resolve_obj_by_qualname` 懒实例化**"替代（`repo://vllm/utils/import_utils.py#L125-L131`）。
- `Platform.get_worker_class()` / `get_model_architecture()` / `check_device()` 已删除；worker 选择改走 `check_and_update_config` 设置 `worker_cls`。
- `use_custom_ops` 门控已删除，换为 `hasattr(torch.ops._C, ...)` 编译产物检查（[ops-custom-kernels.md](../acceleration/ops-custom-kernels.md)）。

## 相关页面

- [platform-detection.md](platform-detection.md) — `current_platform` 如何被选出来
- [plugin-system.md](plugin-system.md) — OOT 平台如何通过 entry point 注册
- [new-backend-guide.md](../guides/new-backend-guide.md) — 完整接入清单
