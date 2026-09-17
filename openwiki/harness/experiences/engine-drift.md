---
type: experience
title: engine-drift：vllm_kunlun 对 vllm 引擎的 API 漂移
summary: 插件↔引擎 API 漂移的修复模式（可重放 patch）与真实运行验证方法。
first_seen:
  model: GLM5.2-Int-W8A8（vllm_kunlun 0.25.1.dev vs vllm 0.25.1，15 处 drift）
sources:
- repo://tools/patches/patch_vllm_kunlun_drift.py
- repo://openwiki/harness/vllm-0251-drift-map.md
---

# engine-drift

## 问题类别

`vllm_kunlun`（厂商插件）与上游 `vllm` 引擎之间没有兼容矩阵：每个版本对齐都
可能出现符号迁移、签名变化、契约改变。GLM5.2 run 实测 15 处，每处的试错成本
是 707GiB 冷启 13-15 分钟。

## 已验证的方法

### 修复模式（AGENTS.md 硬约束）

- 修复必须是**仓库内可重放的幂等 patch**：`tools/patches/patch_*.py`，
  exact-anchor 文本编辑，已应用报 SKIP，anchor 漂移报失败（不静默）。
- **只活在 pod site-packages 里的修复不是修复，是给下次重装埋雷**（GLM run 的
  十三处修复曾随 pod 蒸发过一次）。
- 部署证明不自动重放 patch；修复后必须在同一 Pod 中重新执行真实验证。

### 真实运行验证

- 静态 drift precheck 已移除：它需要持续追随上游内部 API，并会因未执行分支、
  import 顺序和 mock 签名产生误报，维护成本高于实际收益。
- 先验证 runtime/plugin 可导入，再运行 MAT-028 dummy-weight toy bring-up，要求
  engine construction、prefill 和至少两个 decode token。
- toy 通过后再走 shim handoff 与目标服务证明；真实调用路径中的失败作为修复依据，
  不从静态扫描结果推断兼容性。

## 证据入口

- 漂移全景：[vLLM 0.25.1 API 漂移映射表](../vllm-0251-drift-map.md)。
- 修复证据必须包含原始 traceback、已安装 revision、可重放 patch diff，以及 toy
  和服务路径的复验结果。
