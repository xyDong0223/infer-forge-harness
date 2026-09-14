---
type: index
title: Infer-Forge 工程实践
summary: 项目自身的模型适配、验证和算子接入经验；它与上游和插件参考材料分层维护。
---

# Infer-Forge 工程实践

> [返回 OpenWiki 根目录](../index.md)

本模块记录 `infer-forge-harness` 自身沉淀的工程方法。与 `vllm-core/` 的上游参考和 `vllm-kunlun/` 的厂商插件参考不同，本模块的结论应以本仓库的任务契约、工具和验证器为准。

| 页面 | 内容 |
|---|---|
| [模型适配实践经历](practice-experience.md) | 在 Kunlun3 P800 上适配模型时可复用的排查、验证与算子接入经验。 |
| [vLLM 0.25.1 API 漂移映射表](vllm-0251-drift-map.md) | 插件移植到 vllm 0.25.1 的符号迁移、契约变化与 P800 专属陷阱；GLM5.2 适配实测沉淀。 |
