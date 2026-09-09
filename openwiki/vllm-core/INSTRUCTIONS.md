# INSTRUCTIONS.md — vLLM 硬件后端接入 wiki 简报

## 范围与优先级

本 wiki 聚焦 **vLLM 的硬件后端 / Platform 接入体系**，不追求全仓库概览。优先级从高到低：

1. **Platform 抽象层**：`vllm/platforms/` 的 `Platform` 基类、平台检测与选择、内置平台清单。
2. **插件注册机制**：`vllm.platform_plugins` 等 entry-point 组、in-tree 与 out-of-tree 平台的优先级、`PlatformRegistry`。
3. **Attention 后端接入**：`vllm/attention/` 的后端选择链、`AttentionBackend` 需要实现的接口、`VLLM_ATTENTION_BACKEND` 覆盖方式。
4. **算子与量化接入**：`torch.library` 自定义算子注册、`CustomOp` 的 native/device 双路径分发、量化方法注册与平台白名单。
5. **Worker / Executor / 分布式接入点**：Worker 与 ModelRunner 的选择、分布式通信后端挂载。
6. **新后端接入实操指南**：以 XPU/CPU 等非 CUDA 平台为参照，给出接入清单（必须实现 / 可选覆盖）。

## 写作要求

- 每个事实性结论必须落到具体源码位置（`repo://路径#行号` 形式写进 sources）。
- 语言：中文，代码符号保留英文原名。
- 页面之间用标准 Markdown 链接互联，从 index.md 可达所有页面。
- 使用 Mermaid 图表达：平台选择流程、attention 后端选择链、插件加载时序。
- 证据版本：本 wiki 基于 vLLM commit `94848eda600a07c28675f5753a11b2c212c146ed`（2026-09-09, main 分支浅克隆）。
