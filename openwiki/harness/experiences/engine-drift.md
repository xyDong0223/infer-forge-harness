---
type: experience
title: engine-drift：vllm_kunlun 对 vllm 引擎的 API 漂移
summary: 插件↔引擎 API 漂移的修复模式（可重放 patch）、加载前预检方法、import-order 假阳性。
first_seen:
  model: GLM5.2-Int-W8A8（vllm_kunlun 0.25.1.dev vs vllm 0.25.1，15 处 drift）
sources:
- repo://tools/patches/patch_vllm_kunlun_drift.py
- repo://tools/probe/engine_core_drift_precheck.py
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
- 部署证明在每次 install/attach 后自动重放 patch 集，再用 drift 预检验证。

### 加载前预检（把 15 分钟一轮变成秒级一轮）

- `tools/probe/engine_core_drift_precheck.py`：权重加载前做符号 resolve +
  调用点 `inspect.signature().bind` mock 干跑。
- gate 规则按 ModelRegistry 架构 scope + init surface：**只有调用点失败才
  hard-gate**；裸属性引用（如 PlatformEnum.HPU/NEURON 这类未执行分支里的真缺失
  符号）一律 report-only——健康 pod 上有 23 个这类"合法缺失"。

### import-order 假阳性（预检自己的坑）

- 插件先 import 会污染引擎模块导入（"Duplicate op name" 崩溃）→ 引擎侧
  warmup 先行 + 干净子进程仲裁争议符号。
- alias 归属只信 asname import；import-raised 错误优先级高于 missing。

## 证据入口

- 漂移全景：[vLLM 0.25.1 API 漂移映射表](../vllm-0251-drift-map.md)。
- 预检报告字段：`summary.drift_gated / drift_report_only / drift_out_of_path`
  （互斥三桶）。
