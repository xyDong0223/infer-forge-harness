---
type: index
title: 能力轴经验索引
summary: 按能力轴（而非模型）组织的适配经验。模型是能力问题的首次发现现场，不是经验的归属单位。
---

# 能力轴经验索引

> [返回工程实践](../index.md) · [返回 OpenWiki 根目录](../../index.md)

本目录按**能力轴**组织经验，与 mat-003/mat-008 的 capability axes 使用同一套词汇。
一个模型 run 结束时，findings 里可泛化的结论提炼进对应轴的页面；模型名只作为
"首次发现现场"的引用保留。下一个同族模型（例如下一个 MLA+MoE+W8A8 模型）从这里
继承全部既有积累，而不是从零开始。

按模型命名的 run 实例与 findings（`tasks/*/instances/`、`*findings.yaml`）仍然按
模型命名——它们是事实记录，不是经验。

| 能力轴 | 内容 |
|---|---|
| [attention-mla](attention-mla.md) | MLA（DeepSeekV2 风格）+ DSA indexer + deferred-residual 层协议 |
| [attention-swa](attention-swa.md) | 滑窗注意力：prefill/decode 的窗口参数语义与 fallback 验证 |
| [attention-block-sparse](attention-block-sparse.md) | MSA（MiniMax-M3 Sparse Attention）：三个 vendor 算子与 index cache write |
| [moe](moe.md) | MoE 层：W8A8 expert、FusedMoE 工厂参数、dummy-expert 造模 |
| [quantization-w8a8](quantization-w8a8.md) | W8A8 INT8 动态量化：scale 布局与门禁度量 |
| [spec-decode-mtp](spec-decode-mtp.md) | MTP / speculative decode：nextn 层配置与 decode 路由 |
| [engine-drift](engine-drift.md) | vllm_kunlun↔vllm API 漂移：修复模式与真实运行验证 |
