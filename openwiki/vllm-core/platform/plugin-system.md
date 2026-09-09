---
type: concept
title: 插件系统与 Entry Point 组
generated:
  by: comate-agent/1.0
  at: 2026-09-09
verified:
  by: comate-agent/1.0
  at: 2026-09-09
sources:
  - repo://vllm/plugins/__init__.py
  - repo://docs/design/plugin_system.md
  - repo://pyproject.toml
  - repo://vllm/v1/worker/worker_base.py
related:
  - "[平台检测](platform-detection.md)"
  - "[新后端接入指南](../guides/new-backend-guide.md)"
---

# 插件系统与 Entry Point 组

统一加载器：`load_plugins_by_group(group)`（`repo://vllm/plugins/__init__.py#L36-L74`）——`importlib.metadata.entry_points(group=...)` 发现、`VLLM_PLUGINS` 白名单过滤（@L40, L56-L66）、逐个 `plugin.load()` 失败只记日志不中断（@L68-L72）。返回 `dict[str, Callable]`——**值是工厂函数，不是类本身**。

## 1. 全部 6 个 Entry Point 组

| # | 组名 | 加载时机 / 进程 | 典型用途 | 位置 |
|---|---|---|---|---|
| 1 | `vllm.general_plugins` | 所有进程，每进程一次 | **OOT 模型注册**（`ModelRegistry.register_model`）、LoRA resolver | 常量 @L18；执行 @L77-L90；worker 触发点 `repo://vllm/v1/worker/worker_base.py#L269-L271` |
| 2 | `vllm.platform_plugins` | 首次访问 `current_platform` 时惰性加载 | **OOT 硬件平台注册**（返回平台类限定名或 None） | 常量 @L21-L23；裁决 `repo://vllm/platforms/__init__.py#L243-L286` |
| 3 | `vllm.io_processor_plugins` | 仅 process0 | pooling 模型的输入/输出预处理（返回 IOProcessor 类限定名） | 常量 @L20；`repo://vllm/plugins/io_processors/__init__.py#L38` |
| 4 | `vllm.stat_logger_plugins` | 仅 process0 的 async serve | 统计日志插件 | 常量 @L26 |
| 5 | `vllm.endpoint_plugins` | 仅 API server 前端；**默认不加载**，必须显式列入 `VLLM_PLUGINS` | API endpoint 扩展，按 `required_tasks` 二次门控 | 常量 @L30；加载 @L93-L158 |
| 6 | `vllm.logits_processors` | VllmConfig 侧探测 | 采样 logits processor | `repo://vllm/v1/sample/logits_processor/__init__.py#L48` |

**不存在 `vllm.model_architectures` 组**（全库 grep 无此字符串）；OOT 模型注册走 `vllm.general_plugins`（官方文档示例 `repo://docs/design/plugin_system.md#L26-L46`）。

vLLM 自身只注册了 `vllm.general_plugins` 的两个 LoRA resolver（filesystem / hf_hub）：`repo://pyproject.toml#L46-L49`。**内置平台不经 pyproject entry point 注册**——`builtin_platform_plugins` 字典直接挂 Python 函数，与 OOT 插件走同一套裁决代码。

## 2. `vllm.platform_plugins` 平台插件契约

依据 `repo://docs/design/plugin_system.md#L50-L124`：

1. **entry point 声明**（插件包 setup.py）：
   ```python
   entry_points={"vllm.platform_plugins": ["my_platform = vllm_my_hw:register"]}
   ```
2. **`register()` 函数**：当前环境不支持时返回 `None`；支持时返回平台类全限定名：
   ```python
   def register():
       if not detect_my_hardware():
           return None
       return "vllm_my_hw.platform.MyHardwarePlatform"
   ```
3. **平台类要求**：继承 `vllm.platforms.interface.Platform`；`_enum = PlatformEnum.OOT`；至少实现 `device_type`、`device_name`、`check_and_update_config`（**必须在其中设置 `worker_cls`**）、`get_attn_backend_cls`、`get_device_communicator_cls`。
4. **Worker 类要求**：继承 `vllm.v1.worker.worker_base.WorkerBase`。

## 3. 插件的幂等性要求

- worker 进程初始化会重新 `load_general_plugins()`（`repo://vllm/v1/worker/worker_base.py#L269-L271`）——平台检测、模型注册函数都会**在每个 worker 进程重跑**，必须可重入、不报错（`repo://vllm/plugins/__init__.py#L77-L81`、plugin_system.md@L52-L56）。
- `VLLM_PLUGINS` 白名单：未设置 → 除 endpoint 插件外全部加载；设为空串 → 全不加载；endpoint 插件永远需要显式列出。

## 相关页面

- [platform-detection.md](platform-detection.md) — 平台插件被裁决的过程
- [new-backend-guide.md](../guides/new-backend-guide.md) — 把这些 entry point 串成接入步骤
