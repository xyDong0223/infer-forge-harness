---
type: experience
title: attention-swa：滑窗注意力参数语义与 decode fallback
summary: 滑窗不是独立算子而是参数；decode qlen 路由与 torch fallback 的验证方法。
first_seen:
  model: Qwen3-8B / MiniMax 系（首个需要 SWA fallback 的 bring-up）
sources:
- repo://tools/torch/paged_decode.py
- repo://tools/probe/sliding_window_decode_probe.py
- repo://tasks/mat-008-capability-evaluation/task.yaml
---

# attention-swa

## 问题类别

sliding-window attention（SWA）。**术语**：SWA 是滑窗注意力的标准缩写；
MSA 是 MiniMax-M3 Sparse Attention（[另一轴](attention-block-sparse.md)），
两者不可混用——mat-008 契约曾把本维度键误写为 `msa`，已改 `swa`。

## 已验证的约定（踩过的坑）

### 滑窗是参数，不是算子

- kunlun_ops 413 个符号里**没有**独立的 window 算子。窗口语义分布在：
  prefill 走 `swa_left`/`swa_right`，decode 走 `max_window_size`。
- 因此本轴的工作量在 torch 参考实现的窗口语义，不在新 kernel。
- flash-attn 风格的窗口是 `(sliding_window - 1, 0)`，插件把 config 值直传，
  不做 -1——窗口边界 off-by-one 是历史坑。

### decode 路由

- vendor `speculative_attention` 的常规 decode（`qlen == 1`）与 speculative
  路径共入口：fallback 只路由常规 decode，speculative 保留 vendor kernel
  （`tools/patches/apply_torch_decode_patch.py` 的 dispatch 逻辑）。
- `KDP_DECODE_KERNEL=speculative` 环境变量可复现原始失败（可逆验证）。

### fallback 的边界必须显式拒绝

- fallback 拒绝 attention sinks 与未支持的 `max_window_size`，而不是猜一个
  近似（`paged_decode.py` 的 `UnsupportedDecode`）。fallback-validation skill
  的核心原则。

## 工具与证据

| 工具 | 用途 |
|---|---|
| `tools/torch/paged_decode.py` | torch SWA decode fallback（部署对象=评分对象，同一文件） |
| `tools/probe/sliding_window_decode_probe.py` | mat-008 swa 维度 probe，相对 float32 参考的 relative_l2 门禁 |

- 几何必须 `context_len > window`：窗口内满时窗口与无窗计算相同，probe 会
  "通过但什么都没测"。
- window_mask 的当前 token 永不被 mask；无窗时只剩 context 长度约束。
